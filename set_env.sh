#!/bin/bash

# Récupère le chemin absolu du dossier contenant ce script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Active l'environnement conda global
source ~/miniconda3/bin/activate hyphc_env


# Définit les chemins relatifs à ce dossier
export HHC_HOME="$SCRIPT_DIR"
export DATAPATH="$HHC_HOME/data"                # Path where to save the data files
export SAVEPATH="$HHC_HOME/embeddings"          # Path where to save the trained models 


