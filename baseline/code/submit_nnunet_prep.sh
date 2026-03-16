#!/bin/bash
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --job-name=nnunet_prep
#SBATCH --output=logs/nnunet_prep_%j.log

echo "Starting nnU-Net Preprocessing..."
date

# Environment Setup
source "$HOME/ish/venv_pancreas/bin/activate"
source "baseline/code/env_nnunet.sh"

# Convert MSD dataset to nnU-Net v2 format
echo "Converting MSD Dataset..."
nnUNetv2_convert_MSD_dataset -i /scratch/lustre/home/kayi9958/ish/data/Task07_Pancreas

# Run Planning and Preprocessing
echo "Planning and Preprocessing Dataset007_Pancreas..."
nnUNetv2_plan_and_preprocess -d 7 --verify_dataset_integrity

echo "nnU-Net Preprocessing Complete."
date