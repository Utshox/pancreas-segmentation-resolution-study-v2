#!/bin/bash
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --job-name=dl_btcv
#SBATCH --output=logs/download_btcv_%j.log

# Check if token is provided
if [ -z "$SYNAPSE_TOKEN" ]; then
    echo "Error: SYNAPSE_TOKEN environment variable is not set."
    echo "Usage: sbatch --export=ALL,SYNAPSE_TOKEN='your_token' baseline/code/submit_download_btcv.sh"
    exit 1
fi

echo "Starting BTCV Dataset Download via Synapse..."
date

# Activate environment
source "$HOME/ish/venv_pancreas/bin/activate"

# Run the python script
python baseline/code/download_btcv.py --token "$SYNAPSE_TOKEN"

echo "Process Complete."
date
