#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=ext_inf
#SBATCH --output=logs/extract_infer_%j.log

echo "=== Extract Student & Run Inference ==="
date

source "$HOME/ish/venv_pancreas/bin/activate"
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS
LIBDEVICE_PATH=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1)
CUDA_ROOT=$(dirname $(dirname $(dirname $LIBDEVICE_PATH)))
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

# Step 1: Extract student model
echo "=== Extracting student model ==="
python baseline/code/extract_student.py

# Step 2: Run inference on MSD test set
echo ""
echo "=== Inference: UA-MT 50% Dice+BCE ==="
date
python baseline/code/sliding_window_inference.py \
    --image_dir "/scratch/lustre/home/kayi9958/ish/data_val/imagesTr" \
    --label_dir "/scratch/lustre/home/kayi9958/ish/data_val/labelsTr" \
    --model_path "baseline/models/ssl_uamt_50_dicebce/standalone_best.h5" \
    --output_dir "baseline/logs/verification" \
    --exp_name "uamt50_dicebce"

echo ""
echo "=== All done ==="
date
