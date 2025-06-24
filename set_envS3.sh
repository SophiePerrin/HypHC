#!/bin/bash

# Récupère le chemin absolu du dossier contenant ce script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Active l'environnement conda
source ~/miniconda3/bin/activate hyphc_env

# Définir les variables d’environnement S3
export AWS_S3_ENDPOINT=minio.lab.sspcloud.fr


export AWS_DEFAULT_REGION=us-east-1

# Définir le chemin local pour les sauvegardes
export SAVEPATH=$HOME/work/HypHC/results

# (Optionnel) Créer le dossier s’il n’existe pas
# mkdir -p "$SAVEPATH"

