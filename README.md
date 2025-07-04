# Hyperbolic Hierarchical Clustering (HypHC)

This code is adapted from the official PyTorch implementation of the NeurIPS 2020 paper: 
> **From Trees to Continuous Embeddings and Back: Hyperbolic Hierarchical Clustering**\
> Ines Chami, Albert Gu, Vaggos Chatziafratis and Christopher Ré\
> Stanford University\
> Paper: https://arxiv.org/abs/2010.00402

<p align="center">
  <img width="400" height="400" src="https://github.com/HazyResearch/HypHC/blob/master/HypHC.gif">
</p>

> **Abstract.** Similarity-based Hierarchical Clustering (HC) is a classical unsupervised machine learning algorithm that has traditionally been solved with heuristic algorithms like Average-Linkage. Recently, Dasgupta reframed HC as a discrete optimization problem by introducing a global cost function measuring the quality of a given tree. In this work, we provide the first continuous relaxation of Dasgupta's discrete optimization problem with provable quality guarantees. The key idea of our method, HypHC, is showing a direct correspondence from discrete trees to continuous representations (via the hyperbolic embeddings of their leaf nodes) and back (via a decoding algorithm that maps leaf embeddings to a dendrogram), allowing us to search the space of discrete binary trees with continuous optimization. Building on analogies between trees and hyperbolic space, we derive a continuous analogue for the notion of lowest common ancestor, which leads to a continuous relaxation of Dasgupta's discrete objective. We can show that after decoding, the global minimizer of our continuous relaxation yields a discrete tree with a (1+epsilon)-factor approximation for Dasgupta's optimal tree, where epsilon can be made arbitrarily small and controls optimization challenges. We experimentally evaluate HypHC on a variety of HC benchmarks and find that even approximate solutions found with gradient descent have superior clustering quality than agglomerative heuristics or other gradient based algorithms. Finally, we highlight the flexibility of HypHC using end-to-end training in a downstream classification task.


## Installation

This code has been tested with python3.7. First, create a virtual environment (or conda environment) and install the dependencies:

```#Script à automatiser pour création environnement adapté HypHC

cd HypHC

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda create -n hyphc_env python=3.7
conda activate hyphc_env
python --version


#Puis installation de HypHC :

pip install -r requirements.txt
cd mst; python setup.py build_ext --inplace
cd ..
cd unionfind; python setup.py build_ext --inplace
cd ..

pip install s3fs

source ./set_envS3.sh```

## Datasets : for using the HypHC examples : 

```source download_data.sh```

This will download the zoo, iris and glass datasets from the UCI machine learning repository. Please refer to the paper for the download links of the other datasets used in the paper. 

## For using weibo or reddit datasets :

see Code Usage below

## Code Usage

### Train script

To use the code, first set environment variables in each shell session:

```source set_env.sh``` (or, here, ```source set_envS3.sh```)

To train the HypHC mode, use the train script:

```
python train.py
    optional arguments:
      -h, --help            show this help message and exit
      --seed SEED
      --epochs EPOCHS
      --batch_size BATCH_SIZE
      --learning_rate LEARNING_RATE
      --eval_every EVAL_EVERY
      --patience PATIENCE
      --optimizer OPTIMIZER
      --save SAVE
      --fast_decoding FAST_DECODING
      --num_samples NUM_SAMPLES
      --dtype DTYPE
      --rank RANK
      --temperature TEMPERATURE
      --init_size INIT_SIZE
      --anneal_every ANNEAL_EVERY
      --anneal_factor ANNEAL_FACTOR
      --max_scale MAX_SCALE
      --dataset DATASET
``` 

### Examples

We provide examples of training commands for the zoo, iris and glass datasets. For instance, to train HypHC on zoo, run: 

```source examples/run_zoo.sh``` 

This will create an `embedding` directory and save training logs, embeddings and the configuration parameters in a `embedding/zoo/[unique_id]` where the unique id is based on the configuration parameters used to train the model.   

### weibo and reddit datasets :

# "reddit"
python train.py \
  --seed 1234 \
  --epochs 10 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --eval_every 10 \
  --patience 20 \
  --optimizer RAdam \
  --save 1 \
  --fast_decoding 1 \
  --num_samples 1000 \
  --dtype double \
  --rank 2 \
  --temperature 0.01 \
  --init_size 0.001 \
  --anneal_every 20 \
  --anneal_factor 1.0 \
  --max_scale 0.999 \
  --dataset reddit


# "weibo"
python train.py \
  --seed 1234 \
  --epochs 10 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --eval_every 10 \
  --patience 20 \
  --optimizer RAdam \
  --save 1 \
  --fast_decoding 1 \
  --num_samples 10000 \
  --dtype double \
  --rank 2 \
  --temperature 0.01 \
  --init_size 0.001 \
  --anneal_every 20 \
  --anneal_factor 1.0 \
  --max_scale 0.999 \
  --dataset weibo

# Then, we can retrieve the hyperbolic embeddings for using it with GADBench models :

python pick_up_embeddings.py --use_latest --seed 1234

or 

python pick_up_embeddings.py --model_dir /chemin/vers/le/modele --seed 1234

## Citation

If you find this code useful, please cite the following paper:

```
@inproceedings{NEURIPS2020_ac10ec1a,
 author = {Chami, Ines and Gu, Albert and Chatziafratis, Vaggos and R\'{e}, Christopher},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {15065--15076},
 publisher = {Curran Associates, Inc.},
 title = {From Trees to Continuous Embeddings and Back: Hyperbolic Hierarchical Clustering},
 url = {https://proceedings.neurips.cc/paper/2020/file/ac10ec1ace51b2d973cd87973a98d3ab-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
