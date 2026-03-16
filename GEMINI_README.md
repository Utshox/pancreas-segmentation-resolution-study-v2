# Gemini CLI - Project Handover & Context

Hello! This file is for you (and future Gemini CLI instances) to quickly understand the state of this project.

## Key Facts
- **Goal:** Pancreas segmentation on CT scans.
- **SOTA Model:** Supervised Patch-based U-Net (v6).
- **Metric to Watch:** Dice Score (SOTA is ~0.85-0.91).
- **Environment:** Running on a SLURM-based GPU cluster (Tesla V100). Virtual environment is at `venv_pancreas`.

## Important Paths
- **Latest Work:** All "gold standard" scripts are in `ishFinal/`.
- **Preprocessed Data:** `preprocessed_v5_patches/`
- **Raw Data:** `data_val/` (contains imagesTr/labelsTr for validation/testing).

## Instructions for Gemini
1. **Always verify GPU availability** before suggesting training/inference runs.
2. **Prioritize Patch-based methods:** Direct 3D training showed memory issues; 2D patch-based (256x256) is the working standard here.
3. **SSL Status:** We are currently bridging the gap between SSL (FixMatch/MT) and the strong Supervised Baseline.
