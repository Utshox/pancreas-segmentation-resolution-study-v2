# Paper 2: Frequency-Domain Diffusion Models for Pancreatic Cancer Detection

## Core Idea
Train a diffusion model to learn the distribution of "healthy" pancreas CT images. At inference, input a potentially cancerous image → the diffusion process "denoises" (removes) the anomaly → compare before/after → high difference = cancer detected. Do this in both spatial domain and FFT (Fourier) domain. If FFT domain performs better, that's a novel finding.

## Research Questions
1. Can diffusion models trained on healthy pancreas CT reliably detect anomalies (tumours) via reconstruction error?
2. Does operating in the Fourier frequency domain improve anomaly detection compared to spatial domain?
3. Can the spectral difference map localize the tumour region?

## Why This Is Novel
- AnoDDPM (Wyatt et al., 2022) proved diffusion-based anomaly detection works in spatial domain (brain MRI)
- **Nobody has applied diffusion denoising in the Fourier/FFT domain for medical anomaly detection**
- Builds directly on Paper 1's Fourier analysis (we proved high-frequency boundary info is critical)
- The comparison spatial-vs-frequency diffusion is itself a publishable contribution

## Proposed Method
1. **Data preparation:** Separate MSD Task07 into healthy slices (label=0, parenchyma only) vs cancerous slices (label=2, tumour present)
2. **Spatial-domain diffusion:** Train DDPM on healthy pancreas patches (256x256). At inference, add noise → denoise → compute pixel-wise difference map
3. **Frequency-domain diffusion:** Apply 2D FFT to patches → train diffusion in frequency space (magnitude + phase) → inverse FFT after denoising → compare
4. **Anomaly score:** L1/L2 difference between input and reconstruction, thresholded for detection
5. **Localization:** Difference map highlights tumour regions

## Target Journals
- IEEE Transactions on Medical Imaging (IF ~10.6)
- Medical Image Analysis (IF ~10.9)
- MICCAI 2027 conference paper → then extended journal version

## Technical Requirements
- PyTorch (diffusion models)
- DDPM / improved DDPM implementation
- 2D FFT (torch.fft)
- GPU: V100 32GB should be sufficient for 2D patches
- Estimated training: ~24-48 GPU hours per model

## Dataset Options
- MSD Task07_Pancreas (already have — tumour labels available)
- TCIA Pancreas-CT (healthy only — good for training)
- Potential: PDAC datasets from TCIA for more cancer cases

## Timeline Estimate
- Month 1: Literature review, data preparation, baseline DDPM in spatial domain
- Month 2: FFT-domain diffusion implementation, experiments
- Month 3: Comparison study, ablations, manuscript writing

## Connection to Paper 1
- Reuses Fourier analysis framework
- Builds on the resolution-preservation insight
- Same dataset, complementary research questions
- Together they form a strong PhD portfolio: "Resolution-aware medical image analysis"
