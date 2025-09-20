"""Triplet sampling utils."""

import numpy as np
from tqdm import tqdm


def samples_triples(n_nodes, num_samples):
    num_samples = int(num_samples)
    all_nodes = np.arange(n_nodes)
    mesh = np.array(np.meshgrid(all_nodes, all_nodes))
    pairs = mesh.T.reshape(-1, 2)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    n_pairs = pairs.shape[0]
    if num_samples < n_pairs:
        print("Generating all pairs subset")
        subset = np.random.choice(np.arange(n_pairs), num_samples, replace=False)
        pairs = pairs[subset]
    else:
        print("Generating all pairs superset")
        k_base = int(num_samples / n_pairs)
        k_rem = num_samples - (k_base * n_pairs)
        subset = np.random.choice(np.arange(n_pairs), k_rem, replace=False)
        pairs_rem = pairs[subset]
        pairs_base = np.repeat(np.expand_dims(pairs, 0), k_base, axis=0).reshape((-1, 2))
        pairs = np.concatenate([pairs_base, pairs_rem], axis=0)
    num_samples = pairs.shape[0]
    triples = np.concatenate(
        [pairs, np.random.randint(n_nodes, size=(num_samples, 1))],
        axis=1
    )
    return triples


def generate_all_triples(n_nodes):
    triples = []
    for n1 in tqdm(np.arange(n_nodes)):
        for n2 in np.arange(n1 + 1, n_nodes):
            for n3 in np.arange(n2 + 1, n_nodes):
                triples += [(n1, n2, n3)]
    return np.array(triples)


# Fonction ajoutée (pour suréchantillonner les triplets "inter-clusters")
def samples_triples_balanced(n_nodes, num_samples, similarities, inter_prob=0.5):
    """
    Génère des triplets (i,j,k) équilibrés entre triplets intra- et inter-cluster
    de manière heuristique via les similarités.
    
    similarities: matrice numpy n_nodes x n_nodes, valeurs entre 0 et 1
    inter_prob: probabilité de tirer un triplet "inter-cluster" (Plus inter_prob est élevé, 
    plus le modèle sera forcé à séparer les clusters et donc aura un coût de Dasgupta discret
    plus faible et une train loss (relaxation continue hyperbolique du coût de Dasgupta) plus haute.)
    """
    num_samples = int(num_samples)
    all_nodes = np.arange(n_nodes)
    mesh = np.array(np.meshgrid(all_nodes, all_nodes))
    pairs = mesh.T.reshape(-1, 2)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    n_pairs = pairs.shape[0]

    # sous-échantillon ou sur-échantillon pour atteindre num_samples
    if num_samples < n_pairs:
        print("Generating all pairs subset")
        subset = np.random.choice(np.arange(n_pairs), num_samples, replace=False)
        pairs = pairs[subset]
    else:
        print("Generating all pairs superset")
        k_base = int(num_samples / n_pairs)
        k_rem = num_samples - (k_base * n_pairs)
        subset = np.random.choice(np.arange(n_pairs), k_rem, replace=False)
        pairs_rem = pairs[subset]
        pairs_base = np.repeat(np.expand_dims(pairs, 0), k_base, axis=0).reshape((-1, 2))
        pairs = np.concatenate([pairs_base, pairs_rem], axis=0)

    num_samples = pairs.shape[0]
    triples = np.zeros((num_samples, 3), dtype=int)
    triples[:, :2] = pairs

    # Tirage du troisième élément avec probabilité basée sur similarité
    for idx, (i,j) in enumerate(pairs):
        if np.random.rand() < inter_prob:
            # tirer k "lointain" des i,j
            k_candidates = np.where((similarities[i] < 0.5) & (similarities[j] < 0.5))[0]
            if len(k_candidates) == 0:
                k_candidates = all_nodes
        else:
            # tirer k "proche" des i,j
            k_candidates = np.where((similarities[i] > 0.5) | (similarities[j] > 0.5))[0]
            if len(k_candidates) == 0:
                k_candidates = all_nodes
        triples[idx, 2] = np.random.choice(k_candidates)

    return triples
