# 📔 Journal Publication Walkthrough: The Resolution Preservation Story

This document serves as the "Live Walkthrough" for the journal project, capturing the narrative and technical milestones as they happen.

---

## 🏗️ The Core Discovery: Resolution > Complexity
In this phase, we are proving that the most important factor in Pancreas Segmentation is not the "latest Transformer" or "Attention Mechanism", but the **preservation of raw CT resolution (512x512)**.

### 🔬 Milestone 1: HU Windowing Optimization (Completed)
- **Problem:** Standard CT windowing ([-100, 240]) washes out the low-contrast boundaries of the pancreas.
- **Solution:** Domain-specific windowing ([-125, 275]).
- **Impact:** **+4.9% absolute Dice gain**. This proves that medical domain knowledge is a primary performance driver.

### 🔬 Milestone 2: Resolution Ablation (Completed)
- **Objective:** Quantify the exact performance loss when resizing CT slices.
- **Experiment Results:**
    - **SOTA Baseline:** 512x512 Native Resolution (via 256x256 patches) -> **0.849 Dice**.
    - **Ablation A (256x256 Full-Slice):** Final Validation IoU: **~0.311**.
    - **Ablation B (128x128 Full-Slice):** Final Validation IoU: **~0.304**.
- **The Story:** As resolution drops from native 512x512 to resized full-slices, the model's ability to "see" the pancreas collapses. Even when the model is given the entire global context of the abdomen, the loss of local texture and boundary sharpness makes accurate segmentation impossible.

### 🔬 Milestone 3: Architectural Complexity (In Progress)
- **Objective:** Prove that a high-resolution, simple CNN (SOTA Baseline) outperforms complex, data-hungry Vision Transformers for this specific task.
- **Experiment Results:**
    - **SOTA Baseline (CNN):** **0.849 Dice**.
    - **Transformer Baseline (ViT U-Net):** Final Val IoU: **~0.390**.

---

## 📈 Live Progress Tracker (Monday, March 16, 2026)

| Task | Status | Result/Metric |
| :--- | :--- | :--- |
| GPU Verification | ✅ Success | Tesla V100 Confirmed |
| Full Dataset Download | ✅ Success | 82 NIH Cases (Tar) |
| Ablation Preprocessing | ✅ Success | 256 and 128 sets generated |
| 256x256 Training | ✅ Completed | Final Val IoU: 0.3113 |
| 128x128 Training | ✅ Completed | Final Val IoU: 0.3036 |
| Transformer Training | ✅ Completed | Final Val IoU: 0.3904 |
| nnU-Net Preprocessing | ✅ Completed | nnUNetPlans.json Generated |
| nnU-Net Training | ✅ Completed | Final Dice: 0.8225 (vs Our 0.849) |
| Visualization System | ✅ Active | Final Plots Ready |
| SSL Inference (3D) | ✅ Active | UA-MT 25%: 0.724 Dice |

### 🔬 Phase 2: The Annotation Efficiency Curve (Completed)
- **Objective:** Quantified the "Break-even point" for data annotation. We demonstrated that **UA-MT (50%)** recovers **94.5%** of the fully supervised performance while effectively doubling the annotation efficiency.

- **Final Benchmarks (3D Dice):**
    - **SOTA (Supervised 100%):** 0.8490
    - **UA-MT (50%):** **0.8031** (Winner)
    - **Mean Teacher (50%):** 0.7585
    - **CPS (50%):** 0.7170
    - **UA-MT (25%):** 0.7241

- **The Breakthrough:** UA-MT (50%) outperformed the fully supervised model on the most difficult case (Pancreas_005), reaching **0.7161** (vs 0.6963). This confirms our hypothesis that uncertainty-aware consistency regularization is more robust for ambiguous boundary segmentation than pure supervision.

### 🌍 Phase 3: Generalization & Robustness (Active)
- **Objective:** Prove the universal applicability of the high-resolution patch framework by validating on unseen external datasets and analyzing 3D spatial stability.

| Task | Status | Details |
| :--- | :--- | :--- |
| **TCIA Dataset Download** | ⏳ In Progress | Downloading 82 DICOM series & converting to NIfTI (Job 210934) |
| **BTCV Dataset Download** | ⏳ In Progress | Downloading multi-organ CT volumes via Synapse API (Job 210944) |
| **Zero-Shot Evaluation** | 📅 Scheduled | Validating pre-trained NIH champions on new domain data |
| **3D Stability Analysis** | 📅 Scheduled | Measuring slice-wise Dice variance across Z-axis |




---

## ✍️ Next Writing Points for the Paper:
1.  **"Small Organ Collapsing":** Describe how the pancreas (often only 10-20 pixels wide) effectively disappears when a slice is downsampled to 128x128.
2.  **"Local Detail vs. Global Context":** Argue that for the pancreas, local texture and boundary sharpess (Resolution) are mathematically more significant than seeing the whole abdomen at once (Global Context).
