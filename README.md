# High-Resolution Patch-Based Pancreas Segmentation (v2)

This repository contains the finalized code and models for the **High-Resolution Patch-Based Framework** for pancreas segmentation on CT scans. This project represents a shift from complex 3D architectures on low-resolution inputs to standard 2D architectures on high-resolution patches.

## 🚀 The Core Breakthrough: Resolution Preservation
Standard 3D models (V-Net, UNETR, etc.) often downsample 512x512 CT slices to 128x128 or 256x256 to fit GPU memory, discarding 75% of pixel data. We demonstrated that for the pancreas, **preserving native voxel resolution (0.7mm x 0.7mm)** is more critical than capturing global abdominal context at once. 

By switching to a patch-based approach (256x256 patches from 512x512 slices), we achieved a significant Dice improvement, reaching a SOTA of **0.849 Dice** on the NIH Pancreas-CT dataset.

## 📊 SOTA Performance (Supervised Baseline v6)
| Dataset Case | Dice Score |
| :--- | :--- |
| **pancreas_004** | **0.9169** |
| **pancreas_006** | **0.8604** |
| **Average (Test Set)** | **0.8489** |

*Note: The results outperform methods like RSTN (0.845) and TotalSegmentator (0.801).*

## 🧪 Semi-Supervised Learning (SSL)
We also integrate Semi-Supervised Learning to address annotation scarcity:
- **Mean Teacher:** Achieves **0.83 Dice** using only 50% of labeled data (98% of full-supervision performance).
- **FixMatch:** Studied and compared, showing that consistency regularization (Mean Teacher) is superior for "soft" medical organ boundaries compared to hard-threshold pseudo-labeling.

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

## 🛠️ Installation & Usage
### Prerequisites
- Python 3.10
- TensorFlow 2.x
- Nibabel (for NIfTI files)

### Inference
To verify the results on your own data:
```bash
python baseline/code/sliding_window_inference.py \
    --image_dir /path/to/images/ \
    --label_dir /path/to/labels/ \
    --model_path baseline/models/model_patch_best.h5 \
    --output_dir results/ \
    --exp_name verification_run
```

## 📜 Publication
This work was documented for a conference submission. The full LaTeX draft is available in `journey.txt`.

## 🤝 Acknowledgments
This research was conducted at the **Institute of Computer Science, Vilnius University**, utilizing the VU MIF HPC resources (NVIDIA V100 GPU Cluster).
