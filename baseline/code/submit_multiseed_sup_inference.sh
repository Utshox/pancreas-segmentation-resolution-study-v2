#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=ms_inf
#SBATCH --output=logs/multiseed_sup_inference_%j.log

echo "=== Multi-Seed Supervised Inference ==="
echo "Job ID: $SLURM_JOB_ID"
date

source "$HOME/ish/venv_pancreas/bin/activate"
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS
LIBDEVICE_PATH=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1)
CUDA_ROOT=$(dirname $(dirname $(dirname $LIBDEVICE_PATH)))
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

IMAGE_DIR="/scratch/lustre/home/kayi9958/ish/data_val/imagesTr"
LABEL_DIR="/scratch/lustre/home/kayi9958/ish/data_val/labelsTr"
OUTPUT_DIR="baseline/logs/verification"

for SEED in 42 123 456; do
    SUP_MODEL="baseline/models/multiseed_supervised/seed_${SEED}/model_patch_best.h5"
    if [ -f "$SUP_MODEL" ]; then
        echo ""
        echo "=== Supervised Seed $SEED ==="
        date
        python baseline/code/sliding_window_inference.py \
            --image_dir "$IMAGE_DIR" \
            --label_dir "$LABEL_DIR" \
            --model_path "$SUP_MODEL" \
            --output_dir "$OUTPUT_DIR" \
            --exp_name "supervised_seed${SEED}"
    else
        echo "SKIP: $SUP_MODEL not found"
    fi
done

echo ""
echo "=== All multi-seed inference complete ==="
date
