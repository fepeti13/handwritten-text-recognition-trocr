#!/bin/bash
#SBATCH --job-name=trocr-hungarian
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trocr

cd ~/handwritten-text-recognition-trocr

nvidia-smi

python src/train.py \
    --model_name microsoft/trocr-base-handwritten \
    --train_csv data/processed/train.csv \
    --val_csv data/processed/val.csv \
    --images_dir data/processed/images \
    --output_dir models/base-hungarian-v2 \
    --num_epochs 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --fp16

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
