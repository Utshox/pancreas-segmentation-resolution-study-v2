#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=tcia_inf
#SBATCH --output=logs/inference_tcia_%j.log

echo "--- Running Zero-Shot 3D Inference on TCIA External Dataset ---"
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
IMAGE_DIR="/scratch/lustre/home/kayi9958/ish/data_external/imagesTs"
LABEL_DIR="/scratch/lustre/home/kayi9958/ish/data_external/labelsTs"
OUTPUT_DIR="baseline/logs/verification"

# 1. Supervised SOTA (100%)
echo "Step 1: Running Zero-Shot Inference for Supervised SOTA..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "baseline/models/model_patch_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "tcia_supervised_sota"

# 2. UA-MT (50%)
echo "Step 2: Running Zero-Shot Inference for UA-MT (50%)..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "baseline/models/ssl_uamt_50/standalone_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "tcia_uamt_50"

echo "TCIA Zero-Shot Evaluation Complete."
date
