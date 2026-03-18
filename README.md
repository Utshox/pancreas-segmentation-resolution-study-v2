# High-Resolution Patch-Based Pancreas Segmentation (v2)

This repository contains the finalized code and models for the **High-Resolution Patch-Based Framework** for pancreas segmentation on CT scans. This project represents a shift from complex 3D architectures on low-resolution inputs to standard 2D architectures on high-resolution patches.

## 🚀 The Core Breakthrough: Resolution Preservation
Standard 3D models (V-Net, UNETR, etc.) often downsample 512x512 CT slices to 128x128 or 256x256 to fit GPU memory, discarding 75% of pixel data. We demonstrated that for the pancreas, **preserving native voxel resolution (0.7mm x 0.7mm)** is more critical than capturing global abdominal context at once. 

By switching to a patch-based approach (256x256 patches from 512x512 slices), we achieved a significant Dice improvement, reaching a SOTA of **0.849 Dice** on the Medical Segmentation Decathlon (MSD) Task07_Pancreas dataset.

## 📊 SOTA Performance (Phase 1 & Phase 2 Benchmarks)
| Model | Avg 3D Dice | Case 001 | Case 004 | Case 005 | Case 006 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SOTA (Supervised 100%)** | **0.8490** | 0.9018 | 0.8448 | 0.6963 | 0.8159 |
| **UA-MT (SSL 50%)** | **0.8031** | 0.8915 | 0.8805 | 0.7161 | 0.7243 |
| **Mean Teacher (SSL 50%)** | **0.7585** | 0.9002 | 0.8499 | 0.6478 | 0.6361 |
| **UA-MT (SSL 25%)** | **0.7241** | 0.7657 | 0.8645 | 0.5449 | 0.7211 |

*Note: UA-MT at 50% data actually outperformed the fully supervised model on the most difficult case (Pancreas_005), proving the superior regularizing effect of uncertainty-aware consistency.*

## 📖 Roadmap: From Conference to Q1 Journal
This project is currently evolving from a conference-level study into a comprehensive **Q1 Journal Publication** (Targeting IEEE T-MI or MedIA). 
- **Phase 1 (Complete):** Resolution ablation, windowing optimization, and SOTA baseline established.
- **Phase 2 (Complete):** Annotation efficiency curve established. UA-MT proven to double efficiency compared to standard methods.
- **Phase 3 (Upcoming):** Cross-dataset generalization (validation on external data) and 3D stability analysis.
- **Phase 4 (Drafting):** Formal manuscript preparation with multi-slice qualitative heatmaps.

## 📝 Q1 Journal Submission Guidelines
To ensure we don't prematurely finalize sections (like the Conclusion) before the research is actually complete, and to adhere to strict Q1 journal standards, we will follow these guidelines:

### **1. IEEE Transactions on Medical Imaging (T-MI)**
*   **Length:** Strictly limited to **10 pages** for the initial submission (double-column format). Overlength charges apply after 8 pages.
*   **Format:** IEEE template (Double-column, single-spaced).
*   **Abstract:** Unstructured, typically ~250 words.
*   **Key Focus:** Extremely dense, technically rigorous. Emphasizes mathematical justification and comprehensive ablation (which Phase 1 & 2 provide).

### **2. Medical Image Analysis (MedIA - Elsevier)**
*   **Length:** Flexible (typically ~3,500+ words), allowing for more extensive discussion and qualitative visual analysis.
*   **Format:** Single-column, double-spaced (for review).
*   **Abstract:** **Structured** (Background, Methods, Results, Conclusions) and under 350 words.
*   **Highlights:** Requires 3 to 5 mandatory bullet points summarizing the core novelties.

**Drafting Rule:** The `Conclusion` section in the `.tex` file must remain a `[Placeholder]` until the absolute final Phase (Phase 3) is completed and cross-dataset results are fully analyzed.

## 📁 Repository Structure
```text
ishFinal/
├── baseline/            # Main research directory
│   ├── code/            # Training, Inference, and Plotting scripts
│   ├── logs/            # Best training logs, verification runs, and plots
│   ├── models/          # Saved model weights (SOTA and SSL Champions)
│   └── paper/           # LaTeX source files for the journal manuscript
├── overleaf_export/     # Ready-to-upload LaTeX package for Overleaf preview
├── RESEARCH_LOG.md      # Detailed logs of experiments and metrics
└── JOURNAL_WALKTHROUGH.md # The narrative roadmap for the publication
```

## 🤝 Acknowledgments
This research is conducted at the **Institute of Computer Science, Vilnius University**, utilizing the VU MIF HPC resources (NVIDIA V100 GPU Cluster).
