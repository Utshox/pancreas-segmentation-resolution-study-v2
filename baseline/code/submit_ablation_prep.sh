#!/bin/bash
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --job-name=ablation_prep
#SBATCH --output=logs/ablation_prep_%j.log

echo "Starting Ablation Dataset Preparation (256 & 128)..."
date

# Environment Setup
source "$HOME/ish/venv_pancreas/bin/activate"

# Paths
RAW_DIR="/scratch/lustre/home/kayi9958/ish/data/Task07_Pancreas"
OUTPUT_DIR="/scratch/lustre/home/kayi9958/ish/ablation_data"

mkdir -p "$OUTPUT_DIR"

python baseline/code/preprocess_ablation.py \
    --raw_dir "$RAW_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Preparation Complete."
date
