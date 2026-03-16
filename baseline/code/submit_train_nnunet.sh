#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_nnunet
#SBATCH --output=logs/train_nnunet_%j.log

echo "Starting nnU-Net Training (3D Full Resolution)..."
date

# Environment Setup
source "$HOME/ish/venv_pancreas/bin/activate"
source "baseline/code/env_nnunet.sh"

# Run nnU-Net training for Dataset 007 (Pancreas)
# 3d_fullres = 3D Full Resolution U-Net
# 0 = Fold 0 of the 5-fold cross-validation
nnUNetv2_train 7 3d_fullres 0

echo "nnU-Net Training Complete."
date