#!/bin/bash
#SBATCH --job-name=eval-test
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trocr
cd ~/handwritten-text-recognition-trocr

python scripts/evaluate-test-set.py