#!/bin/bash
source /Users/sophieperrin/.pyenv/versions/3.7.17/envs/hyphc_env/bin/activate
export HHC_HOME=$(pwd)
export DATAPATH="$HHC_HOME/data"                # Path where to save the data files
export SAVEPATH="$HHC_HOME/embeddings"          # Path where to save the trained models 
