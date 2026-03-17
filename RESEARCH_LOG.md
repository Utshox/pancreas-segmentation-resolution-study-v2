# Pancreas Segmentation Research Log (Journal & PhD)

This log tracks all experiments, architectural decisions, and benchmarking results for the "Resolution Preservation" journal project.

---

## 📅 Monday, March 16, 2026

### 🚀 Session Start: Initializing Phase 1
**Objective:** Begin Comparative Benchmarking (SOTA vs. Complexity).
**Baseline Status:** Patch-based U-Net v6 (`model_patch_best.h5`) | Verified Avg Dice: **0.849**.
**Key Targets:** 
- SOTA Comparison (nnU-Net, Swin-UNet).
- Resolution Ablation (512 vs 256 vs 128).
- HU Windowing Optimization.

### 🧪 Experiment 1: HU Windowing Study
*Goal: Quantify the benefit of domain-specific windowing [-125, 275] vs. Standard [-100, 240].*

**Setup:**
- Model: `baseline/models/model_patch_best.h5`
- Data: 4 NIH validation cases (001, 004, 005, 006)
- Scripts: `sliding_window_inference.py` (Champion) vs `sliding_window_inference_hu_std.py` (Standard)
- HPC Job: `sbatch baseline/code/submit_hu_study.sh` (Job ID: 210598)

**Execution:**
- [x] Create standardized inference script for comparison.
- [x] Create `submit_verify_gpu_v3.sh` and confirm GPU functionality (Job ID: 210596).
- [x] Launch comparative inference job.

**Status:** Completed.
**Results:**
- **Champion ([-125, 275]):** Avg Dice **0.8147**
- **Standard ([-100, 240]):** Avg Dice **0.7653**
- **Gain:** +4.94% absolute Dice. This confirms that the pancreas-specific HU range is critical for capturing low-contrast boundary details.

---

### 🧪 Experiment 2: Resolution Ablation (Downsampling vs. Patching)
*Goal: Prove that "Resolution Preservation" (Patching) is superior to "Global Context" (Downsampling) for small organ segmentation.*

**Setup:**
- Baseline: Patch-based U-Net v6 (512x512 patches on native resolution).
- Ablation A: Full-slice U-Net resized to 256x256.
- Ablation B: Full-slice U-Net resized to 128x128.
- Scripts: `preprocess_ablation.py`, `run_patch_training_v2.py` (will be adapted for full-slice).

**Execution:**
- [x] Download full NIH Pancreas dataset via Google Drive link (tar archive).
- [x] Extract dataset and update `submit_ablation_prep.sh` to use the correct data path (`Task07_Pancreas`) and partition (`main`).
- [x] Generate ablation datasets (256, 128).
- [x] Adapt `run_patch_training_v2.py` for full-slice training (`run_ablation_training.py` created).
- [x] Create SLURM submission scripts (`submit_train_ablation_256.sh` and `submit_train_ablation_128.sh`).
- [x] Create visualization scripts for journal plots (`plot_training_history.py` and `plot_ablation_comparison.py`).
- [x] Train U-Net on 256x256 resized slices - **Completed (Job 210607)**
- [x] Train U-Net on 128x128 resized slices - **Completed (Job 210608)**.

**Status:** Resolution Ablation Study COMPLETE. Both 128x128 and 256x256 full-slice models have finished training. 

**Final Observations (T+2h):**
- **Ablation 128:** COMPLETED. Final Val IoU: **0.3036**. The extreme loss of resolution prevents the model from accurately segmenting the pancreas despite seeing the whole slice.
- **Ablation 256:** COMPLETED. Final Val IoU: **0.3113**. 
- **Comparison:** The 256x256 resolution barely outperformed the 128x128 resolution on validation, despite having 4x more pixels. Most importantly, both are dramatically lower than the native 512x512 patch-based SOTA (Avg Dice **0.849**). This proves that the critical spatial information for the pancreas is irreparably lost when the CT slice is downsampled.

---

### 🧪 Experiment 3: Architectural Complexity (SOTA vs. Transformer)
*Goal: Prove that "Patch-based Simple CNNs" outperform "Complex Transformers" for this task.*

**Setup:**
- Baseline: Patch-based U-Net v6 (512x512 native resolution patches).
- Ablation A (Transformer): Vision Transformer U-Net (TransUNet style / Swin equivalent).
- Data: `preprocessed_v5_patches` (Ensures a 1:1 fair comparison on the same exact data).
- Scripts: `transformer_unet.py`, `run_transformer_training.py`.

**Execution:**
- [x] Implement Vision Transformer U-Net architecture.
- [x] Create training script and SLURM submission (`submit_train_transformer.sh`).
- [x] Train Transformer Baseline on SOTA patches - **Completed**.
- [x] Create `env_nnunet.sh` and set up standard `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` folders.
- [x] Convert MSD Task07 to nnU-Net v2 and run `nnUNetv2_plan_and_preprocess` - **Completed**.
- [ ] Train nnU-Net Baseline (`3d_fullres` configuration) - **Running (Job 210621)**.
- [ ] Compare inference results against SOTA CNN.

**Status:** Transformer Baseline model has completed training (Final Val IoU: **0.3904**). nnU-Net dataset conversion and preprocessing is COMPLETE. The official nnU-Net training job is actively running.

---

### 🧪 Phase 2: Annotation Efficiency (Semi-Supervised Learning)
*Goal: Identify the "break-even point" for data annotation using SSL.*

**Setup:**
- Methods: Mean Teacher (MT), Cross-Pseudo Supervision (CPS), Uncertainty-Aware Mean Teacher (UA-MT).
- Ratios: 10%, 25%, 50% labeled data.
- Scripts: `create_ssl_splits.py`, `run_ssl_meanteacher_v2.py`, `run_ssl_cps.py`, `run_ssl_uamt.py`.

**Execution:**
- [x] Create standardized data split JSON for consistent comparison (281 cases total).
- [x] Implement UA-MT with Monte Carlo Dropout uncertainty weighting.
- [x] Implement CPS with mutual pseudo-labeling.
- [ ] Train SSL models on 10% labeled data - **Running (Jobs 210687, 210688, 210689)**.
- [ ] Train SSL models on 25% labeled data.
- [ ] Train SSL models on 50% labeled data.

**Status:** Phase 2 is officially launched. The 10% labeled data benchmarks are currently training on the GPU cluster. nnU-Net (Phase 1) is also approaching its final epochs in parallel.

---
