#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=ssl_inf50
#SBATCH --output=logs/ssl_inference_50_%j.log

echo "--- Final 50% SSL Volumetric Benchmark ---"
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

echo "Step 1: Extracting Standalone U-Net Weights from 50% SSL Wrappers..."
# Note: extract_weights.py uses --type, --checkpoint, --out arguments as seen in your previous submit script
python baseline/code/extract_weights.py --type uamt --checkpoint baseline/models/ssl_uamt_50/model_best.h5 --out baseline/models/ssl_uamt_50/standalone_best.h5
python baseline/code/extract_weights.py --type cps --checkpoint baseline/models/ssl_cps_50/model_best.h5 --out baseline/models/ssl_cps_50/standalone_best.h5

# 1. UA-MT 50%
echo "Step 2: Running Full 3D Inference on UA-MT 50%..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "baseline/models/ssl_uamt_50/standalone_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "ssl_uamt_50_inference"

# 2. CPS 50%
echo "Step 3: Running Full 3D Inference on CPS 50%..."
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "baseline/models/ssl_cps_50/standalone_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "ssl_cps_50_inference"

echo "Final 50% SSL Volumetric Benchmark Complete."
date
