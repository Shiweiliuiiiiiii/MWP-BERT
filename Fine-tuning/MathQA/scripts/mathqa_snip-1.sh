#!/bin/bash
#SBATCH --job-name=MWP-BERT-mathqa-snip-before-1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 4-00:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o MWP-BERT-mathqa-snip-1.out

source /home/sliu/miniconda3/etc/profile.d/conda.sh
conda activate prune_cry


for sparsity in 0.2 0.36 0.488 0.590
do

python run_ft_before_FT.py \
    --output_dir /home/sliu/project_space/pruning_cfails/QA/mathQA/bert_snip_before/$sparsity/ \
    --fix --sparse_init snip --sparsity $sparsity --sparse \
    --bert_pretrain_path /home/sliu/project_space/pruning_fails/QA/mathQA/pretrained_models/MWP-BERT_en \
    --data_dir data \
    --train_file MathQA_bert_token_train.json \
    --finetune_from_trainset MathQA_bert_token_train.json \
    --dev_file MathQA_bert_token_val.json \
    --test_file MathQA_bert_token_test.json \
    --schedule linear \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --n_epochs 80 \
    --warmup_steps 4000 \
    --n_save_ckpt 3 \
    --n_val 5 \
    --logging_steps 100 \
    --embedding_size 128 \
    --hidden_size 768 \
    --beam_size 5 \
    --dropout 0.5 \
    --seed 17

done