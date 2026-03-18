#!/bin/bash
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --job-name=dl_tcia
#SBATCH --output=logs/download_tcia_%j.log

echo "Starting TCIA Dataset Download and NIfTI Conversion..."
date

# Activate environment
source "$HOME/ish/venv_pancreas/bin/activate"

# Run the python script
python baseline/code/download_tcia.py

echo "Process Complete."
date
