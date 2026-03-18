#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=ssl_inference
#SBATCH --output=logs/ssl_inference_%j.log

echo "--- Running Full 3D Sliding Window Inference on Final SSL Models ---"
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
OUTPUT_DIR="baseline/logs/verification"

echo "Extracting weights from SSL wrappers..."
python baseline/code/extract_weights.py --type uamt --checkpoint baseline/models/ssl_uamt_25/model_best.h5 --out baseline/models/ssl_uamt_25/standalone_best.h5
python baseline/code/extract_weights.py --type mt --checkpoint baseline/models/ssl_meanteacher_50/model_best.h5 --out baseline/models/ssl_meanteacher_50/standalone_best.h5

# 1. UA-MT 25% (The 25% Champion)
echo "Running Inference on UA-MT 25%..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "baseline/models/ssl_uamt_25/standalone_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "ssl_uamt_25_inference"

# 2. Mean Teacher 50%
echo "Running Inference on Mean Teacher 50%..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "baseline/models/ssl_meanteacher_50/standalone_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "ssl_meanteacher_50_inference"

echo "SSL Inference Complete."
date
