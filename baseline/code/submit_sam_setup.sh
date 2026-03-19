#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sam_setup
#SBATCH --output=logs/sam_setup_%j.log

echo "=== SAM/MedSAM Setup and Inference ==="
echo "Job ID: $SLURM_JOB_ID"
date

source "$HOME/ish/venv_pancreas/bin/activate"

# Step 1: Install segment-anything if needed
pip install segment-anything 2>/dev/null
pip install opencv-python-headless 2>/dev/null

# Step 2: Download SAM ViT-B checkpoint (smallest, ~375MB)
SAM_DIR="$HOME/ish/sam_checkpoints"
mkdir -p "$SAM_DIR"

if [ ! -f "$SAM_DIR/sam_vit_b_01ec64.pth" ]; then
    echo "Downloading SAM ViT-B checkpoint..."
    wget -q -O "$SAM_DIR/sam_vit_b_01ec64.pth" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    echo "SAM checkpoint downloaded."
else
    echo "SAM checkpoint already exists."
fi

# Step 3: Download MedSAM checkpoint
if [ ! -f "$SAM_DIR/medsam_vit_b.pth" ]; then
    echo "Downloading MedSAM checkpoint..."
    pip install gdown 2>/dev/null
    gdown "1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_" -O "$SAM_DIR/medsam_vit_b.pth" 2>/dev/null
    if [ ! -f "$SAM_DIR/medsam_vit_b.pth" ]; then
        echo "gdown failed, trying wget with direct link..."
        echo "NOTE: MedSAM checkpoint may need manual download from Google Drive"
    fi
else
    echo "MedSAM checkpoint already exists."
fi

# Step 4: Run SAM inference
echo ""
echo "=== Running SAM Zero-Shot Inference ==="
date
python baseline/code/run_sam_inference.py \
    --image_dir "/scratch/lustre/home/kayi9958/ish/data_val/imagesTr" \
    --label_dir "/scratch/lustre/home/kayi9958/ish/data_val/labelsTr" \
    --sam_checkpoint "$SAM_DIR/sam_vit_b_01ec64.pth" \
    --medsam_checkpoint "$SAM_DIR/medsam_vit_b.pth" \
    --output_dir "baseline/logs/verification"

echo ""
echo "=== SAM inference complete ==="
date
