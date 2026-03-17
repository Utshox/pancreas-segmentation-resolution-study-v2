#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=ssl_phase2
#SBATCH --output=logs/ssl_phase2_%j.log

# Usage: sbatch baseline/code/submit_phase2_ssl.sh <method> <ratio>
# Methods: meanteacher, cps, uamt
# Ratios: 10, 25, 50

METHOD=$1
RATIO=$2

echo "Starting Phase 2 SSL: Method=$METHOD, Ratio=$RATIO%"
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
DATA_DIR="/scratch/lustre/home/kayi9958/ish/preprocessed_v5_patches"
SPLIT_JSON="baseline/code/ssl_splits.json"
OUTPUT_DIR="baseline/models/ssl_${METHOD}_${RATIO}"

mkdir -p "$OUTPUT_DIR"

if [ "$METHOD" == "meanteacher" ]; then
    python baseline/code/run_ssl_meanteacher_v2.py --data_dir "$DATA_DIR" --split_json "$SPLIT_JSON" --ratio "$RATIO" --output_dir "$OUTPUT_DIR"
elif [ "$METHOD" == "cps" ]; then
    python baseline/code/run_ssl_cps.py --data_dir "$DATA_DIR" --split_json "$SPLIT_JSON" --ratio "$RATIO" --output_dir "$OUTPUT_DIR"
elif [ "$METHOD" == "uamt" ]; then
    python baseline/code/run_ssl_uamt.py --data_dir "$DATA_DIR" --split_json "$SPLIT_JSON" --ratio "$RATIO" --output_dir "$OUTPUT_DIR"
else
    echo "Unknown method: $METHOD"
    exit 1
fi

echo "SSL Training Complete."
date
