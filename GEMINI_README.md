# Gemini CLI - Pancreas Segmentation Journey (Journal Project)

This file maintains the persistent state and context for this long-term research project. **Future Gemini instances MUST read this file first.**

## 🎯 Current Mission: Journal Publication
The ultimate goal is to transition this research into a high-impact journal paper. The core novelty is **Resolution Preservation via Patching**.

## 📊 Current State (March 16, 2026)
- **Champion Model:** `ishFinal/baseline/models/model_patch_best.h5` (Patch-based U-Net v6).
- **Verified Metrics:** Avg Dice **0.815** (Max **0.91** on pancreas_001).
- **HPC Environment:** VU MIF HPC (Lustre Scratch). 
- **GPU Setup:** Use `submit_verify_gpu_v3.sh` logic (linking `nvidia` libs from `venv`). **Critical: Use HU Windowing [-125, 275]**.

## 🧠 Strategic Knowledge & Q1 Journal Principles
1. **Resolution vs. Complexity:** Our results prove that high-resolution 256x256 patches beat complex 3D architectures (UNETR, V-Net) that require downsampling.
2. **SSL Preference:** Mean Teacher is the preferred SSL method over FixMatch for this soft-tissue task.
3. **Data Path:** `/scratch/lustre/home/kayi9958/ish/data_val` contains the validation NIfTI files.

### 🌟 Core Principles for the Journal Manuscript (ALWAYS Keep These in Mind)
When drafting the paper or evaluating results, future AI sessions MUST adhere to these three principles to ensure Q1/Q2 journal acceptance:
*   **Statistical Significance:** Never just report that score A is higher than score B. Always aim to run a paired t-test or Wilcoxon signed-rank test to report a p-value (e.g., $p < 0.05$) to prove the gain isn't luck.
*   **Acknowledge Trade-offs:** A mature paper admits its limitations. For example, our patch-based approach preserves resolution but increases training complexity and inference time (due to sliding windows). Acknowledging this builds credibility.
*   **Visual Proof is Everything:** Medical reviewers rely heavily on visual evidence. Always prioritize generating and discussing high-quality, multi-case comparison heatmaps (Ground Truth vs. Prediction) to accompany quantitative tables.

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

## 🚀 Next Session Start (March 17+)
1.  **Check Transformer:** Job `210615` should be complete. Extract its final validation IoU/Dice, add it to `RESEARCH_LOG.md`, update the `ablation_iou_comparison.png` to include it, and write the results into the LaTeX manuscript.
2.  **Check nnU-Net:** Job `210621` is a long-running PyTorch training job. Check its status (`squeue` and logs). If finished, extract its final metrics to see if it beat our 0.815 SOTA. Update logs and the paper accordingly.
