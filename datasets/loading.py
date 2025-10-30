"""Dataset loading."""

import os
import s3fs
import numpy as np

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
]

GADBench_datasets = [
    "weibo",
    "reddit"
]


def load_data(dataset, normalize=True, alpha=None):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    if dataset in UCI_DATASETS:
        x, y = load_uci_data(dataset)
        
        if normalize:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        x0 = x[None, :, :]
        x1 = x[:, None, :]
        cos = (x0 * x1).sum(-1)
        similarities = 0.5 * (1 + cos)
        similarities = np.triu(similarities) + np.triu(similarities).T
        similarities[np.diag_indices_from(similarities)] = 1.0
        similarities[similarities > 1.0] = 1.0

    else:
        if dataset in GADBench_datasets:
            x = load_data_s3("x", dataset)
            y = load_data_s3("y", dataset)
            similarities = load_data_s3("A", dataset)   

        else:
            raise NotImplementedError("Unknown dataset {}.".format(dataset))
               
    return x, y, similarities


def load_uci_data(dataset):
    """Loads data from UCI repository.

    @param dataset: UCI dataset name
    @return: feature vectors, labels
    @rtype: Tuple[np.array, np.array]
    """
    x = []
    y = []
    ids = {
        "zoo": (1, 17, -1),
        "iris": (0, 4, -1),
        "glass": (1, 10, -1),
    }
    data_path = os.path.join(os.environ["DATAPATH"], dataset, "{}.data".format(dataset))
    classes = {}
    class_counter = 0
    start_idx, end_idx, label_idx = ids[dataset]
    with open(data_path, 'r') as f:
        for line in f:
            split_line = line.split(",")
            
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
    y = np.array(y, dtype=int)
    x = np.array(x, dtype=float)
    mean = x.mean(0)
    std = x.std(0)
    x = (x - mean) / std
    return x, y


def load_data_s3(name, dataset_name):
    local_path = f"/tmp/{name}_{dataset_name}.npy"

    if os.path.exists(local_path):
        return np.load(local_path)

    # Paramètres S3
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]

    # Initialiser le système de fichiers S3
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    """
    # Spécifier le chemin dans le bucket
    BUCKET = "projet-clustering-ano-graphe"
    FILE_KEY_S3 = f"albert/{name}_{dataset_name}.npy"  # Remplace par le chemin correct
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
    """

    # Spécifier le chemin complet S3 avec bucket
    FILE_PATH_S3 = "projet-clustering-ano-graphe/albert/{}_{}.npy".format(name, dataset_name)

    # Charger le fichier .npy depuis S3
    with fs.open(FILE_PATH_S3, mode="rb") as f:
        array = np.load(f)

    # Vérification (optionnelle)
    print(array.shape)
    print(array.dtype)

    # Sauvegarde en local pour la prochaine fois
    # np.save(local_path, array)

    return array
