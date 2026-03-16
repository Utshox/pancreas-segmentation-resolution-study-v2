#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_transformer
#SBATCH --output=logs/train_transformer_%j.log

echo "Starting Transformer Baseline Training..."
date

# Environment Setup
source "$HOME/ish/venv_pancreas/bin/activate"
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS
LIBDEVICE_PATH=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1)
CUDA_ROOT=$(dirname $(dirname $(dirname $LIBDEVICE_PATH)))
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

# Paths
# We use the native 512x512 SOTA patches to compare architectures fairly on the same data
DATA_DIR="/scratch/lustre/home/kayi9958/ish/preprocessed_v5_patches"
OUTPUT_DIR="baseline/models/transformer_baseline"

mkdir -p "$OUTPUT_DIR"

python baseline/code/run_transformer_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Training Complete."
date