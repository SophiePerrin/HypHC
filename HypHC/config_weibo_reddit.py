export AWS_S3_ENDPOINT=s3.us-east-1.minio.lab.sspcloud.fr  # adapte si besoin

python train.py \
  --seed 1234 \
  --epochs 50 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --eval_every 10 \
  --patience 20 \
  --optimizer RAdam \
  --save 1 \
  --fast_decoding 1 \
  --num_samples -1 \
  --dtype double \
  --rank 2 \
  --temperature 0.01 \
  --init_size 0.001 \
  --anneal_every 20 \
  --anneal_factor 1.0 \
  --max_scale 0.999 \
  --dataset weibo


export AWS_S3_ENDPOINT=s3.us-east-1.minio.lab.sspcloud.fr  # adapte si besoin

python train.py \
  --seed 1234 \
  --epochs 50 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --eval_every 10 \
  --patience 20 \
  --optimizer RAdam \
  --save 1 \
  --fast_decoding 1 \
  --num_samples -1 \
  --dtype double \
  --rank 2 \
  --temperature 0.01 \
  --init_size 0.001 \
  --anneal_every 20 \
  --anneal_factor 1.0 \
  --max_scale 0.999 \
  --dataset reddit

