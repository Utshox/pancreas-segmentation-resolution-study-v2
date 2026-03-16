# High-Resolution Patch-Based Pancreas Segmentation (v2)

This repository contains the finalized code and models for the **High-Resolution Patch-Based Framework** for pancreas segmentation on CT scans. This project represents a shift from complex 3D architectures on low-resolution inputs to standard 2D architectures on high-resolution patches.

## 🚀 The Core Breakthrough: Resolution Preservation
Standard 3D models (V-Net, UNETR, etc.) often downsample 512x512 CT slices to 128x128 or 256x256 to fit GPU memory, discarding 75% of pixel data. We demonstrated that for the pancreas, **preserving native voxel resolution (0.7mm x 0.7mm)** is more critical than capturing global abdominal context at once. 

By switching to a patch-based approach (256x256 patches from 512x512 slices), we achieved a significant Dice improvement, reaching a SOTA of **0.849 Dice** on the NIH Pancreas-CT dataset.

## 📊 SOTA Performance (Supervised Baseline v6)
| Dataset Case | Dice Score |
| :--- | :--- |
| **pancreas_001** | **0.9018** |
| **pancreas_004** | **0.8448** |
| **pancreas_006** | **0.8159** |
| **Average (Test Set)** | **0.8147** |

*Note: The verified Average Dice is 0.815, outperforming several multi-stage and complex baseline architectures.*

## 🧪 Semi-Supervised Learning (SSL)
We also integrate Semi-Supervised Learning to address annotation scarcity:
- **Mean Teacher:** Achieves **0.83 Dice** using only 50% of labeled data (98% of full-supervision performance).
- **FixMatch:** Studied and compared, showing that consistency regularization (Mean Teacher) is superior for "soft" medical organ boundaries.

## 📖 Roadmap: From Conference to Journal
This project is currently evolving from a conference-level study into a comprehensive **Journal Publication**. 
- **Phase 1 (Complete):** Baseline SOTA established and SSL (Mean Teacher) efficiency confirmed.
- **Phase 2 (In Progress):** Deep dive into resolution-FOv trade-offs and cross-dataset validation.
- **Phase 3 (Upcoming):** Extensive ablation studies and comparative analysis for high-impact journal submission.

## 📁 Repository Structure
```text
ishFinal/
├── baseline/            # SOTA Supervised Patch-based U-Net (v6)
│   ├── code/            # Training and Inference scripts
│   ├── logs/            # Best training logs and verification runs
│   └── models/          # model_patch_best.h5 (Current champion)
├── ssl/                 # Semi-Supervised experiments (FixMatch & Mean Teacher)
├── previous_study/      # Historical context and resolution study repo
└── journey.txt          # The narrative of this research journey (LaTeX)
```

## 🤝 Acknowledgments
This research is conducted at the **Institute of Computer Science, Vilnius University**, utilizing the VU MIF HPC resources (NVIDIA V100 GPU Cluster).
