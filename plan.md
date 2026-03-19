# Roadmap: From SOTA Baseline to Journal Publication & PhD

This document outlines the strategic steps to transition the current Pancreas Segmentation research into a high-impact journal paper and a successful PhD application.

## 🎯 Primary Objectives
1.  **Journal Publication:** Submit to a Q1/Q2 Medical Imaging journal (e.g., *Medical Image Analysis (MedIA)*, *IEEE TMI*, or *Scientific Reports*).
2.  **PhD Selection:** Build a "Research Portfolio" that proves independent research capability, technical mastery of SSL, and contribution to the state-of-the-art.

---

## 📅 Phase 1: Comparative Benchmarking (Current - 1 Month)
*Goal: Prove that "Resolution Preservation" is a novel and superior strategy.*
- [ ] **SOTA vs. Complexity:** Run a structured comparison of our Patch-based U-Net (v6) against **nnU-Net** (the gold standard) and **Swin-UNet** (Transformer). *(Transformer running)*
- [x] **Resolution Ablation:** Quantify the performance drop as resolution is halved (512 vs 256 vs 128). This provides the "Physical Evidence" for our journal's core claim.
- [x] **HU Windowing Study:** Compare the [-125, 275] window against standard [-100, 240] to show domain-specific optimization.

## 🧪 Phase 2: SSL Deep-Dive & Consistency (1-2 Months)
*Goal: Establish our Mean Teacher implementation as a highly efficient SSL protocol.*
- [ ] **MSE vs. Pseudo-labels:** Conduct a deep analysis of *why* Mean Teacher's "soft" consistency beats FixMatch's "hard" thresholding for soft tissue.
- [ ] **Annotation Efficiency Curve:** Plot Dice vs. % Labeled data (10%, 25%, 50%, 100%) to find the "Break-even" point where SSL matches Full Supervision.
- [ ] **Consistency Weighting:** Optimize the sigmoidal ramp-up of $\lambda$ to further stabilize training.

## 🌍 Phase 3: Generalization & Robustness (Completed)
*Goal: Show the framework is universal, not just tuned for one dataset.*
- [x] **TCIA Dataset Download:** Downloading and converting 82 DICOM series from TCIA into NIfTI format. 
- [ ] **BTCV Dataset Download:** Requires Synapse account credentials. Once obtained, we will use `synapse get -r syn3193805`.
- [x] **Cross-Dataset Validation:** Run the champion SOTA model and UA-MT (50%) on these external datasets.
- [x] **Stability Analysis:** Measure the variance of Dice scores across the 3D volume to prove 3D spatial consistency (avoiding "slice anomalies").

## ✍️ Phase 4: Manuscript & Portfolio (Near Complete)
- [x] **High-Resolution Figures:** Multi-slice comparison heatmaps for easy/medium/hard cases (001, 004, 005) + TCIA
- [x] **LaTeX Journal Template:** ~470 lines, 15 figures, 7 tables, 29 citations. Loss ablation section added.
- [x] **PhD Statement of Purpose (SoP):** ~900 word draft at `PhD_Statement_of_Purpose.md`
- [x] **Loss Function Ablation:** Dice+BCE tested and shown inferior (0.824 vs 0.849 BCE-only)
- [ ] **Multi-seed experiments** — needed for error bars (reviewer risk)
- [ ] **Choose target journal** and finalize formatting

---

## 🛡️ Operational Safety: HPC Guidelines
*To ensure a smooth journey on the VU MIF HPC:*
1.  **Computation Strategy:** ALWAYS use `sbatch` for training/inference. Never run calculations on the login node.
2.  **Resource Efficiency:** Stick to the 32GB/1-GPU allocation unless truly needed. 
3.  **CLI Activity:** The "Gemini CLI" is a standard terminal process (similar to `git` or `vim`). Admins monitor for **resource abuse** (CPU/GPU hogging on login nodes) and **security violations**. As long as we follow the rules and only run lightweight CLI commands, it is standard research activity.
4.  **Storage:** Keep large datasets and models on the **Lustre Scratch** (`/scratch/lustre/...`) as you are doing now. This is the HPC's preferred high-performance path.

---

## 🚀 The Vision
By the end of this journey, you won't just have a PhD application; you will have a **published SOTA framework**, a **clean GitHub repository**, and a **proven technical methodology** that makes you a top-tier candidate for any world-class research group.
