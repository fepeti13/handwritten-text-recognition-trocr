#!/bin/bash
#SBATCH --job-name=trocr-hungarian
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=16G

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trocr

cd ~/trocr-project

nvidia-smi

python src/train.py \
    --model_name microsoft/trocr-base-handwritten \
    --train_csv data/processed/train.csv \
    --val_csv data/processed/val.csv \
    --images_dir data/processed/images \
    --output_dir models/base-hungarian \
    --num_epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --fp16

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
```

---

## **Step 3: Update requirements.txt**
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
evaluate>=0.4.0
pillow>=10.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0