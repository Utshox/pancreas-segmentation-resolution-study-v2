#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=btcv_inf
#SBATCH --output=logs/inference_btcv_%j.log

echo "--- Running Zero-Shot 3D Inference on BTCV External Dataset ---"
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
IMAGE_DIR="/scratch/lustre/home/kayi9958/ish/data_external_btcv/averaged-training-images"
LABEL_DIR="/scratch/lustre/home/kayi9958/ish/data_external_btcv/averaged-training-labels"
OUTPUT_DIR="baseline/logs/verification"

# Since BTCV has different suffixes (_avg vs _avg_seg), we need a slight adjustment to the inference call 
# We will use a custom script or a loop if needed, but let's try to standardize the filenames first to make it cleaner.

mkdir -p /scratch/lustre/home/kayi9958/ish/data_external_btcv/standardized_labels

echo "Standardizing BTCV label names..."
python -c "
import os
import shutil
src = '$LABEL_DIR'
dst = '/scratch/lustre/home/kayi9958/ish/data_external_btcv/standardized_labels'
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith('_avg_seg.nii.gz'):
        new_name = f.replace('_avg_seg.nii.gz', '_avg.nii.gz')
        shutil.copy(os.path.join(src, f), os.path.join(dst, new_name))
"

LABEL_DIR_STD="/scratch/lustre/home/kayi9958/ish/data_external_btcv/standardized_labels"

# 1. Supervised SOTA (100%)
echo "Step 1: Running Zero-Shot Inference for Supervised SOTA on BTCV..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR_STD" \
    --model_path "baseline/models/model_patch_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "btcv_supervised_sota"

# 2. UA-MT (50%)
echo "Step 2: Running Zero-Shot Inference for UA-MT (50%) on BTCV..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR_STD" \
    --model_path "baseline/models/ssl_uamt_50/standalone_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "btcv_uamt_50"

echo "BTCV Zero-Shot Evaluation Complete."
date
