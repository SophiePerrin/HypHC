"""Script to adownload the HypHC clustering embeddings."""

import argparse
import json
import os
import s3fs
import torch
import numpy as np
from datasets.loading import load_data
from model.hyphc import HypHC
from utils.poincare import project
import networkx as nx
import pickle
import shutil

# Ajout (pour définir le dossier dans lequel on sauvegardera les embeddings)


def get_latest_model_dir(path):
    """
    Retourne le chemin absolu du dernier dossier de modèle sauvegardé dans `path`.
    Chaque sous-dossier est supposé contenir un modèle (config.json, model_{seed}.pkl).
    """
    # Liste tous les dossiers dans path
    subdirs = [os.path.join(path, d) for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d))]

    if not subdirs:
        raise FileNotFoundError(f"Aucun sous-dossier trouvé dans {path}")

    # Trie par date de dernière modification (dossier le plus récent)
    latest_dir = max(subdirs, key=os.path.getmtime)

    subsubdirs = [os.path.join(latest_dir, d) for d in os.listdir(latest_dir)
                if os.path.isdir(os.path.join(latest_dir, d))]

    if not subdirs:
        raise FileNotFoundError(f"Aucun sous-dossier trouvé dans {latest_dir}")

    # Trie par date de dernière modification (dossier le plus récent)
    latest_subdir = max(subsubdirs, key=os.path.getmtime)

    return latest_subdir


dir = "/home/onyxia/work/HypHC/results"
model_dir = get_latest_model_dir(dir)
# fin de l'ajout


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to a directory with model_{seed}.pkl and config.json")
    parser.add_argument("--seed", type=str, default="0", help="Model seed to use")
    parser.add_argument("--use_latest", action="store_true",
                        help="Use the latest model directory from the default results path")

    args = parser.parse_args()

    if args.use_latest:
        default_base_dir = "/home/onyxia/work/HypHC/results"
        args.model_dir = get_latest_model_dir(default_base_dir)
        print(f"Using latest model directory: {args.model_dir}")

    if args.model_dir is None:
        raise ValueError("You must provide --model_dir or use --use_latest")

    # charge la config + le dataset
    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {args.model_dir}")
    config = json.load(open(config_path))
    config_args = argparse.Namespace(**config)


    # charge les données
    _, y_true, similarities = load_data(config_args.dataset)

  
    # build HypHC model
    model = HypHC(similarities.shape[0], config_args.rank, config_args.temperature, config_args.init_size,
                  config_args.max_scale)
    params = torch.load(os.path.join(args.model_dir, f"model_{args.seed}.pkl"), map_location=torch.device('cpu'))
    model.load_state_dict(params, strict=False)
    model.eval()

    # decode tree
    tree = model.decode_tree(fast_decoding=True)

    leaves_embeddings = model.normalize_embeddings(model.embeddings.weight.data)

    leaves_embeddings = project(leaves_embeddings).detach().cpu().numpy()

    # Sauvegarde des embeddings des feuilles dans le même dossier que le modèle
    # np.save(f"{model_dir}/leaves_emb.npy", leaves_embeddings)

    emb_name = f"leaves_emb_{config_args.dataset}_TRIPLES{config_args.triplets}temp{config_args.temperature}tfactor{config_args.temperature_anneal_factor}lr{config_args.learning_rate}intprob{config_args.inter_prob}.npy"
    np.save(os.path.join(model_dir, emb_name), leaves_embeddings)

    # sauvegarde l'arbre décodé dans le même dossier que le modèle
    # nx.write_gpickle(tree, f"{model_dir}/tree.gpickle")
    # sauvegarde du log correspondant dans le même dossier que le modèle

    '''
    tree_name = f"tree_{config_args.dataset}.gpickle"

    with open(os.path.join(model_dir, tree_name), "wb") as f: # adapté pour l'environnement python 3.8
        pickle.dump(tree, f)
    '''

    BUCKET = "projet-clustering-ano-graphe"
    PREFIX = "albert/"

    # Récupération de l'endpoint depuis l’environnement
    raw_endpoint = os.getenv("AWS_S3_ENDPOINT", "").strip()

    # Ajoute le protocole si absent
    if not raw_endpoint.startswith("http"):
        endpoint = f"https://{raw_endpoint}"
    else:
        endpoint = raw_endpoint

    # Affiche pour debug
    print(f"[INFO] S3 endpoint utilisé : {endpoint}")

    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': endpoint})

    # Upload vers S3
    
    files_to_upload = [
        (os.path.join(model_dir, emb_name), f"{BUCKET}/{PREFIX}{emb_name}")# ,
        # (os.path.join(model_dir, tree_name), f"{BUCKET}/{PREFIX}{tree_name}")
        ]
    # ajoute le log correspondant
    # nom du fichier de log local
    log_file_local = f"train_{config_args.seed}.log"
    log_path = os.path.join(model_dir, log_file_local)  # chemin réel du log

    # nom que tu veux sur S3
    log_file_s3 = f"log_{config_args.dataset}_TRIPLES{config_args.triplets}temp{config_args.temperature}tfactor{config_args.temperature_anneal_factor}lr{config_args.learning_rate}intprob{config_args.inter_prob}.log"
    s3_path_log = f"{BUCKET}/{PREFIX}{log_file_s3}"

    # ajouter le log dans la liste des fichiers à uploader
    if os.path.exists(log_path):
        files_to_upload.append((log_path, s3_path_log))
    else:
        print(f"⚠️ Attention : le log {log_path} n'existe pas et ne sera pas uploadé")

    # ajoute le fichier config.json
    config_file_local = os.path.join(model_dir, "config.json")
    if os.path.exists(config_file_local):
        config_file_s3 = (
            f"config_{config_args.dataset}_TRIPLES{config_args.triplets}temp{config_args.temperature}"
            f"tfactor{config_args.temperature_anneal_factor}"
            f"lr{config_args.learning_rate}"
            f"intprob{config_args.inter_prob}.json"
        )
        s3_path_config = f"{BUCKET}/{PREFIX}{config_file_s3}"
        files_to_upload.append((config_file_local, s3_path_config))
    else:
        print(f"⚠️ Attention : le fichier config.json n'existe pas dans {model_dir}")

    for local_path, s3_path in files_to_upload:
        with fs.open(s3_path, "wb") as f_out:
            with open(local_path, "rb") as f_in:
                f_out.write(f_in.read())
        print(f"  ✔ Uploaded {os.path.basename(local_path)} to s3://{s3_path}")
        