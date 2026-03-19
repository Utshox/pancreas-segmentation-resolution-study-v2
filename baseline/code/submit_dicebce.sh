#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=dicebce
#SBATCH --output=logs/dicebce_%j.log

echo "=== Dice+BCE Loss Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
date

# Environment Setup
source "$HOME/ish/venv_pancreas/bin/activate"
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS
LIBDEVICE_PATH=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1)
CUDA_ROOT=$(dirname $(dirname $(dirname $LIBDEVICE_PATH)))
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

DATA_DIR="/scratch/lustre/home/kayi9958/ish/preprocessed_v5_patches"
SPLIT_JSON="baseline/code/ssl_splits.json"

# -----------------------------------------------
# 1. Supervised Baseline with Dice+BCE (~2-3 hours)
# -----------------------------------------------
echo ""
echo "=== [1/2] Training Supervised Baseline (Dice+BCE) ==="
date
python baseline/code/run_patch_training_dicebce.py \
    --data_dir "$DATA_DIR" \
    --output_dir "baseline/models/supervised_dicebce"

echo "Supervised Dice+BCE training complete."
date

# -----------------------------------------------
# 2. UA-MT 50% with Dice+BCE (~3-4 hours)
# -----------------------------------------------
echo ""
echo "=== [2/2] Training UA-MT 50% (Dice+BCE) ==="
date
python baseline/code/run_ssl_uamt_dicebce.py \
    --data_dir "$DATA_DIR" \
    --split_json "$SPLIT_JSON" \
    --ratio 50 \
    --output_dir "baseline/models/ssl_uamt_50_dicebce"

echo "UA-MT Dice+BCE training complete."
date
echo "=== All Dice+BCE experiments finished ==="
