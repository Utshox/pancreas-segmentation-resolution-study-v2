#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_ablation_128
#SBATCH --output=logs/train_ablation_128_%j.log

echo "Starting Ablation Training (128x128)..."
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
DATA_DIR="/scratch/lustre/home/kayi9958/ish/ablation_data/res_128"
OUTPUT_DIR="baseline/models/ablation_128"

mkdir -p "$OUTPUT_DIR"

python baseline/code/run_ablation_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --img_size 128

echo "Training Complete."
date