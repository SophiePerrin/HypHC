"""Script to visualize the HypHC clustering."""

import argparse
import json
import os
import torch
import numpy as np
from datasets.loading import load_data
from model.hyphc import HypHC
from utils.poincare import project
import networkx as nx

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
    return latest_dir


dir = "/home/onyxia/work/HypHC/embeddings/zoo"
model_dir = get_latest_model_dir(dir)
# fin de l'ajout


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="path to a directory with a torch model_{seed}.pkl and a config.json files saved by train.py."
                        )
    parser.add_argument("--seed", type=str, default=0, help="model seed to use")
    args = parser.parse_args()

    # load dataset
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    config_args = argparse.Namespace(**config)
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

    # np.save("leaves_emb.npy", leaves_embeddings)
    np.save(f"{model_dir}/leaves_emb.npy", leaves_embeddings)
    # sauvegarde les embeddings des feuilles dans le même dossier que le modèle

    nx.write_gpickle(tree, f"{model_dir}/tree.gpickle")