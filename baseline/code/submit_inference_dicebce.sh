#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=inf_dicebce
#SBATCH --output=logs/inference_dicebce_%j.log

echo "=== Dice+BCE Model Inference (3D Volumetric) ==="
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

# Paths
IMAGE_DIR="/scratch/lustre/home/kayi9958/ish/data_val/imagesTr"
LABEL_DIR="/scratch/lustre/home/kayi9958/ish/data_val/labelsTr"
OUTPUT_DIR="baseline/logs/verification"

# -----------------------------------------------
# 1. Supervised Dice+BCE Model
# -----------------------------------------------
echo ""
echo "=== Inference: Supervised Dice+BCE ==="
date
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "baseline/models/supervised_dicebce/model_patch_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "supervised_dicebce"

# -----------------------------------------------
# 2. UA-MT 50% Dice+BCE Model
# -----------------------------------------------
echo ""
echo "=== Inference: UA-MT 50% Dice+BCE ==="
date
python baseline/code/sliding_window_inference.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --model_path "baseline/models/ssl_uamt_50_dicebce/model_best.h5" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "uamt50_dicebce"

# -----------------------------------------------
# 3. Also run on TCIA for cross-dataset comparison
# -----------------------------------------------
TCIA_IMAGE_DIR="/scratch/lustre/home/kayi9958/ish/data_tcia_ras/images"
TCIA_LABEL_DIR="/scratch/lustre/home/kayi9958/ish/data_tcia_ras/labels"

if [ -d "$TCIA_IMAGE_DIR" ]; then
    echo ""
    echo "=== Inference: Supervised Dice+BCE on TCIA ==="
    date
    python baseline/code/sliding_window_inference.py \
        --image_dir "$TCIA_IMAGE_DIR" \
        --label_dir "$TCIA_LABEL_DIR" \
        --model_path "baseline/models/supervised_dicebce/model_patch_best.h5" \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "tcia_supervised_dicebce"

    echo ""
    echo "=== Inference: UA-MT 50% Dice+BCE on TCIA ==="
    date
    python baseline/code/sliding_window_inference.py \
        --image_dir "$TCIA_IMAGE_DIR" \
        --label_dir "$TCIA_LABEL_DIR" \
        --model_path "baseline/models/ssl_uamt_50_dicebce/model_best.h5" \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "tcia_uamt50_dicebce"
else
    echo "TCIA data not found at $TCIA_IMAGE_DIR, skipping."
fi

echo ""
echo "=== All Dice+BCE inference complete ==="
date
