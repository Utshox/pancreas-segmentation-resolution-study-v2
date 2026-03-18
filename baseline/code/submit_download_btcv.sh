#!/bin/bash
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --job-name=dl_btcv
#SBATCH --output=logs/download_btcv_%j.log

echo "Starting BTCV Dataset Download via Synapse..."
date

# Activate environment
source "$HOME/ish/venv_pancreas/bin/activate"

# Create destination directory
BTCV_DIR="/scratch/lustre/home/kayi9958/ish/data_external_btcv"
mkdir -p "$BTCV_DIR"
cd "$BTCV_DIR"

# Download the specific BTCV folder containing training/testing NIfTI files
# Note: You need a Synapse account to download data. We will attempt anonymous download first, 
# but if it requires auth, it will fail and we'll need to pass credentials.
synapse get -r syn3193805

echo "Process Complete."
date