"""Train a hyperbolic embedding model for hierarchical clustering."""

import argparse
import json
import logging
import os

import numpy as np
import io # ###
from sklearn.decomposition import PCA
import torch
import torch.utils.data as data
from tqdm import tqdm

import optim
from config import config_args
from datasets.hc_dataset import HCDataset
from datasets.loading import load_data
from model.hyphc import HypHC
from utils.metrics import dasgupta_cost
from utils.training import add_flags_from_config, get_savedir


# Calcul de la similarité cosine entre features des noeuds par blocs (pour ne pas exploser la mémoire dispo)
# pour les datasets weibo et reddit de GADBench
def compute_cosine_similarity_matrix_blockwise(X, block_size=1000):
    N = X.shape[0]
    X = X.astype(np.float32)

    # Normalisation des vecteurs ligne de X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-8)  # pour éviter la division par zéro

    # Matrice de sortie
    S = np.empty((N, N), dtype=np.float32)

    for i in range(0, N, block_size):
        Xi = X[i:min(i+block_size, N)]
        for j in range(0, N, block_size):
            Xj = X[j:min(j+block_size, N)]
            S_block = np.dot(Xi, Xj.T)
            S[i:i+Xi.shape[0], j:j+Xj.shape[0]] = S_block

    # transformation de la similarité cosine en une similarité comprise entre 0 et 1 
    S = 0.5 * (1.0 + S)
    S = np.clip(S, 0.0, 1.0)
    # Diagonale à 1.0 (au cas où il y aurait un flottement numérique)
    np.fill_diagonal(S, 1.0)
    return S


# optimisation de l'hyperparamètre alpha pour calculer la matrice similarities optimale
# pour minimiser la perte de Dasgupta relaxée en prenant en compte "à la bonne proportion"
# la matrice d'adjacence du graphe et Scosine celle des cosine entre features des noeuds
def optimize_alpha_by_training(alphas, args_template):
    """
    Optimise alpha pour entraîner HypHC sur différentes matrices de similarité.

    Args:
        A, Scosine: matrices numpy (n x n)
        alphas: liste de float ∈ [0,1]
        args_template: Namespace contenant les configs de base (sans .similarities)

    Returns:
        Liste des résultats, meilleur alpha et coût associé
    """
    import copy

    results = []
    best_result = None

    for alpha in alphas:

        args = copy.deepcopy(args_template)
        args.alpha = alpha  # pour log uniquement
        args.save = False  # désactive sauvegarde

        print(f"\n[α={alpha:.2f}] Entraînement...")
        cost = train(args)
        print(f"→ Coût Dasgupta = {cost:.4f}")

        result = {'alpha': alpha, 'cost': cost}
        results.append(result)

        if best_result is None or cost < best_result['cost']:
            best_result = result

    print(f"\n✅ Meilleur alpha : {best_result['alpha']:.2f} → coût Dasgupta = {best_result['cost']:.4f}")

    # 🔽 Entraînement final avec sauvegarde activée
    best_args = copy.deepcopy(args_template)
    best_args.alpha = best_result['alpha']
    best_args.save = True  # active la sauvegarde
    print("\n📦 Réentraînement final avec sauvegarde du meilleur modèle")
    train(best_args)

    return results, best_result['alpha'], best_result['cost']

#############################################

# Fonction pour la réduction de dimension des features des noeuds

#############################################


def analyze_feature_redundancy(X, variance_thresh=1e-6, corr_thresh=0.95, pca_variance=0.95):
   
    # 1. vérifier leurs caractéristiques
    print("Min global :", X.min())
    print("Max global :", X.max())
    
    norms = np.linalg.norm(X, axis=1)
    print("Norme moyenne :", norms.mean())
    print("Norme max :", norms.max())
    print("Ecart-type :", X.std(axis=0))

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    print("Moyenne min (= 0 si déjà centrée réduite):", means.min())
    print("Moyenne max (= 0 si déjà centrée réduite):", means.max())
    print("Écart-type min (= 1 si déjà centrée réduite) :", stds.min())
    print("Écart-type max (= 1 si déjà centrée réduite) :", stds.max())

    print(f"type de X : {type(X)}")

    # Résultat : ni reddit ni weibo ne sont centrés réduits, alors qu'ils doivent l'être pour effectuer la PCA

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    print("Moyenne min vérif (= 0 si centrée réduite):", means.min())
    print("Moyenne max vérif (= 0 si centrée réduite):", means.max())
    print("Écart-type min vérif (= 1 si centrée réduite) :", stds.min())
    print("Écart-type max vérif (= 1 si centrée réduite) :", stds.max())
    print(f"2e type de X : {type(X)}")

    # 2. Calculer la variance
    variances = X.var(axis=0)
    var_idx = np.where(variances >= variance_thresh)[0]       # indices à garder
    low_var_idx = np.where(variances < variance_thresh)[0]    # indices supprimés pour traçabilité
    print(f"{len(low_var_idx)} features ont une variance < {variance_thresh} : {low_var_idx.tolist()} — elles sont supprimées avant la PCA")

    # 3. Filtrer les colonnes à faible variance
    X_clean = X[:, var_idx]
    print(f"type de X_clean : {type(X_clean)}")

    # 5. Features très corrélées (calculé sur X d'origine, pas X_clean)
    corr_matrix = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)
    
    # Matrice de corrélation absolue
    abs_corr = np.abs(corr_matrix)

    # Indices de la partie triangulaire supérieure (hors diagonale)
    triu_indices = np.triu_indices_from(abs_corr, k=1)

    # Paires (i, j) avec leur valeur de corrélation
    pair_scores = [(i, j, abs_corr[i, j]) for i, j in zip(*triu_indices)]  # 🔄 remplacé redundant_pairs

    # Trier par corrélation décroissante
    pair_scores.sort(key=lambda x: x[2], reverse=True)  # 🔄 nouveau : trie toutes les paires par corrélation

    # Affichage
    print("Top 10 des paires de features les plus corrélées :")  # 🔄 message plus clair
    for i, j, score in pair_scores[:10]:                         # 🔄 on affiche les paires réellement les plus corrélées
        print(f"  Feature {i} ↔ Feature {j} (corr = {corr_matrix[i, j]:.2f})")

    # 6. PCA sur les features nettoyées
    pca = PCA(n_components=pca_variance)
    X_pca = pca.fit_transform(X_clean)
    print(f"PCA a réduit de {X_clean.shape[1]} à {pca.n_components_} dimensions (variance expliquée : {pca_variance})")

    means = X_pca.mean(axis=0)
    stds = X_pca.std(axis=0)

    print("Moyenne min vérif (= 0 si centrée réduite):", means.min())
    print("Moyenne max vérif (= 0 si centrée réduite):", means.max())
    print("Écart-type min vérif (= 1 si centrée réduite) :", stds.min())
    print("Écart-type max vérif (= 1 si centrée réduite) :", stds.max())

    print(f"type de X_pca : {type(X_pca)}")

    # 8. Remplacer les features de x par celles transformées par la PCA
    X_pca = pca.transform(X_clean)
    print(f"2e type de X_pca : {type(X_pca)}")

    # 9. Retourner les résultats
    return {
        'features_supprimées_par_variance': low_var_idx.tolist(),  
        'top_corr_pairs': pair_scores[:10],                   
        'pca_model': pca,
        'x_pca': X_pca 
    }


# fonction pour entraîner le modèle
def train(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # get saving directory
    if args.save:
        save_dir = get_savedir(args)
        logging.info("Save directory: " + save_dir)
        save_path = os.path.join(save_dir, "model_{}.pkl".format(args.seed))
        if os.path.exists(save_dir):
            if os.path.exists(save_path):
                logging.info("Model with the same configuration parameters already exists.")
                logging.info("Exiting")
                return
        else:
            os.makedirs(save_dir)
            with open(os.path.join(save_dir, "config.json"), 'w') as fp:
                json.dump(args.__dict__, fp)
        log_path = os.path.join(save_dir, "train_{}.log".format(args.seed))
        hdlr = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    # set seed
    logging.info("Using seed {}.".format(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set precision
    logging.info("Using {} precision.".format(args.dtype))
    if args.dtype == "double":
        torch.set_default_dtype(torch.float64)

    # create dataset
    # x, y_true, similarities = load_data(args.dataset)

    # ici on modifie le programme d'origine pour introduire Scosine et l'optimisation de alpha

    x_dep, y_true, A = load_data(args.dataset)      # #####
    print(f"type de x_dep : {type(x_dep)}")
    # réduction de dimension par PCA
    result_pca = analyze_feature_redundancy(x_dep, pca_variance=0.95)        # #####
    print(f"type de x : {type(x)}")
    x = result_pca['x_pca']
    # Calcul de la similarité cosine
    Scosine = compute_cosine_similarity_matrix_blockwise(x, block_size=1000)

    # Si alpha est fourni dans args, mélange les deux
    if hasattr(args, "alpha") and args.alpha is not None:
        alpha = args.alpha
    else:
        raise ValueError("Tu dois fournir args.alpha")

    # Construction de la matrice finale
    similarities = alpha * A + (1 - alpha) * Scosine  # ######

    # ici on reprend le cours du programme d'origine
    dataset = HCDataset(x, y_true, similarities, num_samples=args.num_samples)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # create model
    model = HypHC(dataset.n_nodes, args.rank, args.temperature, args.init_size, args.max_scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # create optimizer
    Optimizer = getattr(optim, args.optimizer)
    optimizer = Optimizer(model.parameters(), args.learning_rate)

    # train model
    best_cost = np.inf
    # best_model = None
    best_model_buffer = None # ###
    counter = 0
    logging.info("Start training")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(dataloader), unit='ex') as bar:
            for step, (triple_ids, triple_similarities) in enumerate(dataloader):
                triple_ids = triple_ids.to(device)
                triple_similarities = triple_similarities.to(device)
                loss = model.loss(triple_ids, triple_similarities)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.update(1)
                bar.set_postfix(loss=f'{loss.item():.6f}')
                total_loss += loss
        total_loss = total_loss.item() / (step + 1.0)
        logging.info("\t Epoch {} | average train loss: {:.6f}".format(epoch, total_loss))

        # keep best embeddings
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            tree = model.decode_tree(fast_decoding=args.fast_decoding)
            cost = dasgupta_cost(tree, similarities)
            logging.info("{}:\t{:.4f}".format("Dasgupta's cost", cost))
            if cost < best_cost:
                counter = 0
                best_cost = cost
                # best_model = model.state_dict()
                best_model_buffer = io.BytesIO()                    # ###
                torch.save(model.state_dict(), best_model_buffer)   # ###

            else:
                counter += 1
                if counter == args.patience:
                    logging.info("Early stopping.")
                    break

        # anneal temperature
        if (epoch + 1) % args.anneal_every == 0:
            model.anneal_temperature(args.anneal_factor)
            logging.info("Annealing temperature to: {}".format(model.temperature))
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.anneal_factor
                lr = param_group['lr']
            logging.info("Annealing learning rate to: {}".format(lr))
    ######
    logging.info("Optimization finished.")

    if best_model_buffer is not None:
        logging.info("Loading best model before evaluation.")
        best_model_buffer.seek(0)
        best_model_state = torch.load(best_model_buffer)
        model.load_state_dict(best_model_state)
        model_to_save = best_model_state
    else:
        logging.warning("No best model selected during training.")
        model_to_save = model.state_dict()
    
    if args.save:
        logging.info("Saving model at {}".format(save_path))
        torch.save(model_to_save, save_path)
        logger.removeHandler(hdlr)              # ##
        hdlr.close()                            # ##
    '''
    if best_model is not None:
        logging.info("Loading best model before evaluation.")
        model.load_state_dict(best_model)
        model_to_save = best_model
    else:
    logging.warning("No best model selected during training.")

    else:
        logging.warning("No best model selected during training. Saving current model state instead.")
        model_to_save = model.state_dict()

    if args.save:
        logging.info("Saving model at {}".format(save_path))
        torch.save(model_to_save, save_path)
    '''
    '''
    logging.info("Optimization finished.")
    if best_model is not None:
        # load best model
        model.load_state_dict(best_model)

    if args.save:
        # save best embeddings
        logging.info("Saving best model at {}".format(save_path))
        torch.save(best_model, save_path)
    '''
    # evaluation
    model.eval()
    logging.info("Decoding embeddings.")
    tree = model.decode_tree(fast_decoding=args.fast_decoding)
    cost = dasgupta_cost(tree, similarities)
    logging.info("{}:\t{:.4f}".format("Dasgupta's cost", cost))

    if args.save:
        logger.removeHandler(hdlr)
    return cost  #


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    print("config_args =", config_args) #
    parser = add_flags_from_config(parser, config_args)
    args = parser.parse_args()

    if getattr(args, "optimize_alpha", False):      # ####
        
        optimize_alpha_by_training(args.alphas, args)

    else:                                           # ####
        train(args)



'''
python train.py \
  --dataset reddit \
  --num_samples 1000 \
  --alpha 0.2 \
  --epochs 2

ou 

python train.py \
  --dataset reddit \
  --num_samples 1000 \
  --optimize_alpha \
  --alphas 0.0 0.25 0.5 0.75 1.0 \
  --epochs 50

python train.py \
  --dataset weibo \
  --num_samples 1000 \
  --optimize_alpha \
  --alphas 0.0 0.25 0.5 0.75 1.0 \
  --epochs 50


python train.py \
  --dataset weibo \
  --num_samples 100 \
  --alpha 1 \
  --epochs 1 \
  --eval_every 1

'''