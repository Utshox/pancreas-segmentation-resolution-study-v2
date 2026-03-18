#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=stability
#SBATCH --output=logs/stability_analysis_%j.log

echo "--- Running 3D Stability Analysis (Dice Variance) ---"
date

# Environment Setup
source "$HOME/ish/venv_pancreas/bin/activate"
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS
LIBDEVICE_PATH=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1)
CUDA_ROOT=$(dirname $(dirname $(dirname $LIBDEVICE_PATH)))
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

# Run the python script
python baseline/code/analyze_3d_stability.py

echo "Stability Analysis Complete."
date
