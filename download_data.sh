#!/bin/bash
cd $HHC_HOME
mkdir data
for dataset in zoo iris glass; do
  mkdir data/$dataset
  if [ "$dataset" == "zoo" ]; then
    wget -P data/$dataset https://archive.ics.uci.edu/static/public/111/zoo.zip
    unzip data/$dataset/zoo.zip -d data/$dataset  # Dézipper le fichier après téléchargement
  else
    # Mettre ici les liens pour d'autres datasets (ex. iris)
    wget -P data/$dataset https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset/$dataset.data
    wget -P data/$dataset https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset/$dataset.names
  fi
done
