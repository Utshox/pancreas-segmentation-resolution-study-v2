# High-Resolution Patch-Based Pancreas Segmentation

This repository contains the code, models, and manuscript for a **Q1 journal paper** on pancreas segmentation from CT scans. The core thesis: **preserving native CT resolution (512x512) via patch-based processing matters more than architectural complexity** for small-organ segmentation.

## Key Results

| Model | MSD 3D Dice | TCIA Dice (Zero-Shot) |
|:---|:---|:---|
| **Supervised SOTA (100% labels, BCE)** | **0.849** | 0.544 |
| **UA-MT SSL (50% labels, BCE)** | **0.803** | **0.603** |
| nnU-Net 3d_fullres (our reprod.) | 0.823 | — |
| Vision Transformer (TransUNet-style) | 0.390 IoU | — |
| Ablation 256x256 (downsampled) | 0.311 IoU | — |
| Ablation 128x128 (downsampled) | 0.304 IoU | — |

**Note:** MSD Task07 has parenchyma + tumour labels. We merge both into binary pancreas-vs-background. Our Dice is for the combined binary class.

## Current Status (March 19, 2026)

### Completed
- **Phase 1:** Resolution ablation, HU windowing (+4.94% Dice), architecture comparison (CNN vs ViT vs nnU-Net)
- **Phase 2:** SSL annotation efficiency (MT, UA-MT, CPS at 10/25/50% labels)
- **Phase 3:** Cross-dataset generalization (TCIA, 80 volumes), 3D stability analysis
- **Phase 4:** Full manuscript (~440 lines, 13 figures, 6 tables, 29 citations)
- **Fourier analysis:** Frequency-domain proof of why resolution matters (boundary edge strength drops 38% at 128x128)
- **PhD Statement of Purpose:** Draft complete (`PhD_Statement_of_Purpose.md`)

### Recently Completed
- **SAM/MedSAM Foundation Model Comparison:** SAM ViT-B (bbox)=0.705, MedSAM (bbox)=0.439, SAM (auto)=0.097 — all worse than our 7.8M CNN (0.849). Even with GT bounding box prompts, foundation models fail on small organs.
- **Dice+BCE Loss Ablation:** Negative result — 0.824 vs 0.849 BCE-only. BCE alone is better for extreme class imbalance.
- **Additional Mega Plots:** Pancreas_001 (easy) and Pancreas_004 (medium) qualitative comparisons.

### In Progress
- **Multi-seed training (Jobs 211102/103/104):** Seeds 42, 123, 456 for supervised + UA-MT 50%.
  - After completion: `sbatch baseline/code/submit_multiseed_inference.sh`
  - Then update manuscript Table V with mean±std error bars.

### TODO
1. **Multi-seed inference** and manuscript update with error bars.
2. **Switch to Elsevier/CBM format** — structured abstract, highlights, single-column review format.
3. **Final formatting pass** and submission.

## Repository Structure

```
ishFinal/
├── baseline/
│   ├── code/                    # All Python scripts and SLURM submission scripts
│   │   ├── run_patch_training_v2.py        # Supervised baseline (BCE)
│   │   ├── run_patch_training_dicebce.py   # Supervised baseline (Dice+BCE) [NEW]
│   │   ├── run_ssl_uamt.py                # UA-MT SSL (BCE)
│   │   ├── run_ssl_uamt_dicebce.py        # UA-MT SSL (Dice+BCE) [NEW]
│   │   ├── run_ssl_meanteacher_v2.py      # Mean Teacher SSL
│   │   ├── run_ssl_cps.py                 # Cross-Pseudo Supervision SSL
│   │   ├── sliding_window_inference.py     # 3D volumetric inference
│   │   ├── fourier_analysis.py            # Frequency-domain analysis [NEW]
│   │   ├── generate_diagrams.py           # Architecture/pipeline diagrams [NEW]
│   │   ├── plot_mega_inference.py         # Multi-model qualitative comparison
│   │   ├── plot_mega_additional.py        # Additional mega plots [NEW]
│   │   ├── ssl_splits.json               # Fixed SSL data splits
│   │   └── submit_*.sh                   # SLURM submission scripts
│   ├── logs/verification/               # Inference results and plots
│   │   ├── dice_results_*.txt           # Per-case Dice scores
│   │   └── plots/                       # All generated figures
│   ├── models/                          # Trained model weights
│   │   ├── model_patch_best.h5          # SOTA supervised (0.849 Dice)
│   │   ├── ssl_uamt_50/                # UA-MT 50% champion
│   │   ├── ssl_meanteacher_50/         # Mean Teacher 50%
│   │   ├── ssl_cps_50/                 # CPS 50%
│   │   ├── supervised_dicebce/         # [TRAINING] Dice+BCE supervised
│   │   └── ssl_uamt_50_dicebce/        # [TRAINING] Dice+BCE UA-MT
│   └── paper/journal_manuscript.tex     # Synced copy of manuscript
├── overleaf_export/                     # Upload to Overleaf
│   ├── journal_manuscript.tex           # PRIMARY manuscript file
│   ├── references.bib                   # 29 references
│   └── images/                          # All 13+ figures
├── PhD_Statement_of_Purpose.md          # PhD SoP draft
├── plan.md                              # Strategic roadmap
├── RESEARCH_LOG.md                      # Experiment log
├── JOURNAL_WALKTHROUGH.md               # Publication narrative
└── sample_1.pdf, sample_2.pdf           # Reference papers (style guides)
```

## Technical Details

- **HPC:** VU MIF cluster, NVIDIA V100 32GB, SLURM scheduler
- **Environment:** `~/ish/venv_pancreas/` (Python 3.10, TensorFlow 2.x, numpy 1.26)
- **Data:** `/scratch/lustre/home/kayi9958/ish/data/Task07_Pancreas/` (raw NIfTI), preprocessed patches at `/scratch/lustre/home/kayi9958/ish/preprocessed_v5_patches/`
- **TCIA data:** `/scratch/lustre/home/kayi9958/ish/data_tcia_ras/` (reoriented to RAS)

## Hyperparameters (from source code)

| Config | Optimizer | LR | Batch | Epochs | Loss | EMA alpha |
|--------|-----------|------|-------|--------|------|-----------|
| Supervised | Adam | 1e-4 | 32 | 100 | BCE | — |
| Supervised (new) | Adam | 1e-4 | 32 | 100 | Dice+BCE | — |
| UA-MT | Adam | 1e-4 | 8 | 100 | BCE+MSE | 0.999 |
| UA-MT (new) | Adam | 1e-4 | 8 | 100 | Dice+BCE+MSE | 0.999 |
| Mean Teacher | Adam | 1e-4 | 16 | 100 | BCE+MSE | 0.999 |
| CPS | Adam | 1e-4 | 16 | 100 | BCE+BCE | — |
| nnU-Net | Self-config | — | — | 1000 | DC+CE | — |

## Known Issues / Reviewer Risks

1. **Single-run results** — no error bars across random seeds
2. **Small test set** — only 4 MSD test cases (N=4)
3. **TransUNet comparison** — trained from scratch (unfair, needs pre-trained weights)
4. **MSD label handling** — we merge parenchyma+tumour into binary; published MSD results often report per-class averages
5. **TCIA = NIH-82** — same patients, so "external" validation is less independent than it sounds

## Acknowledgments

Institute of Computer Science, Vilnius University. VU MIF HPC resources (NVIDIA V100 GPU Cluster).
