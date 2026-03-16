#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=hu_study
#SBATCH --output=logs/hu_study_%j.log

echo "--- HU Windowing Study: [-125, 275] vs [-100, 240] ---"
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
IMAGE_DIR="/scratch/lustre/home/kayi9958/ish/data_val/imagesTr"
LABEL_DIR="/scratch/lustre/home/kayi9958/ish/data_val/labelsTr"
MODEL_PATH="baseline/models/model_patch_best.h5"
OUTPUT_DIR="baseline/logs/verification"

# 1. Run Champion Windowing [-125, 275]
echo "Running Inference with Champion Window [-125, 275]..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "hu_champion"

# 2. Run Standard Windowing [-100, 240]
echo "Running Inference with Standard Window [-100, 240]..."
python baseline/code/sliding_window_inference_hu_std.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "hu_standard"

echo "HU Study Complete."
date
