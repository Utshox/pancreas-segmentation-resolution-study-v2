# Gemini CLI - Pancreas Segmentation Journey (Journal Project)

This file maintains the persistent state and context for this long-term research project. **Future Gemini instances MUST read this file first.**

## 🎯 Current Mission: Journal Publication
The ultimate goal is to transition this research into a high-impact journal paper. The core novelty is **Resolution Preservation via Patching**.

## 📊 Current State (March 16, 2026)
- **Champion Model:** `ishFinal/baseline/models/model_patch_best.h5` (Patch-based U-Net v6).
- **Verified Metrics:** Avg Dice **0.815** (Max **0.91** on pancreas_001).
- **HPC Environment:** VU MIF HPC (Lustre Scratch). 
- **GPU Setup:** Use `submit_verify_gpu_v3.sh` logic (linking `nvidia` libs from `venv`). **Critical: Use HU Windowing [-125, 275]**.

## 🧠 Strategic Knowledge
1. **Resolution vs. Complexity:** Our results prove that high-resolution 256x256 patches beat complex 3D architectures (UNETR, V-Net) that require downsampling.
2. **SSL Preference:** Mean Teacher is the preferred SSL method over FixMatch for this soft-tissue task.
3. **Data Path:** `/scratch/lustre/home/kayi9958/ish/data_val` contains the validation NIfTI files.

## 🛤️ Future Work Directions
- **Ablation Studies:** Quantify the exact impact of patch size and overlap.
- **Journal Comparisons:** Compare against newer SOTA models (nnU-Net, Swin-UNet) using the high-res patching strategy.
- **Refinement:** Investigate why `pancreas_005` (~0.69) is significantly lower than others and how to stabilize it.

## 🛠️ Operational Instructions
- **Always verify GPU** before running inference/training.
- **Maintain Unique Logs:** When running `sliding_window_inference.py`, use the `--exp_name` flag to avoid overwriting verification history.
- **Version Control:** All "Gold Standard" assets are in `ishFinal/`.
- **Continuity & Documentation:** Every session MUST proactively update `RESEARCH_LOG.md` (for raw data/status) and `JOURNAL_WALKTHROUGH.md` (for the narrative story).
- **Journal Writing:** A formal LaTeX manuscript is located at `baseline/paper/journal_manuscript.tex`. As new experiments (like Transformer or nnU-Net baselines) finish, YOU MUST update the `Methodology` and `Results` sections of this `.tex` file using a high-quality academic tone (avoiding generic "AI styling").
