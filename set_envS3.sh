#!/bin/bash

# --------------------------------------------
# set_env.sh
# Configure l'environnement Python et MinIO/S3
# --------------------------------------------

# Récupère le chemin absolu du dossier contenant ce script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# -----------------------------
# Activation de l'environnement conda
# -----------------------------
echo "Activation de l'environnement conda 'hyphc_env'"
source ~/miniconda3/bin/activate hyphc_env

# -----------------------------
# Variables MinIO / S3
# -----------------------------
export AWS_S3_ENDPOINT="https://minio.lab.sspcloud.fr"
export AWS_DEFAULT_REGION=us-east-1

export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Définir le chemin local pour les sauvegardes
export SAVEPATH=$HOME/work/HypHC/results

# (Optionnel) Créer le dossier s’il n’existe pas
mkdir -p "$SAVEPATH"


