#!/bin/bash
#SBATCH --job-name=MWP-BERT-mathqa-LTH-1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 3-12:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o MWP-BERT-mathqa-LTH-1.out

source /home/sliu/miniconda3/etc/profile.d/conda.sh
conda activate prune_cry

python run_ft_LTH.py \
    --output_dir /home/sliu/project_space/pruning_fails/QA/mathQA/bert_lth/ \
    --fix --sparse_init iterative_gm --sparsity 0.2 --imp_iters 11 --sparse \
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
