# Statement of Purpose

**Istiaque Ahmed**
PhD Applicant, Computer Science / Medical Artificial Intelligence

---

A pancreas occupies roughly 0.5% of an abdominal CT scan. That single fact -- that one of the most lethal cancers in the world hides inside a sliver of voxels that most segmentation models treat as background noise -- is what turned my interest in deep learning into a research commitment. I am applying to PhD programs in Computer Science with a focus on medical image analysis because I believe the hardest unsolved problems in clinical AI are not about building bigger models. They are about building smarter ones: methods that respect the structure of medical data, that learn from scarce annotations, and that generalize beyond the institution where they were trained.

## Research Background

I am currently a researcher at the Institute of Computer Science, Vilnius University, Lithuania, working under the supervision of Kursat Komurcu on high-resolution patch-based pancreas segmentation from CT images. This project began with a simple observation that became my central hypothesis: the standard practice of downsampling CT volumes to 96x96 or 128x128 before feeding them to a segmentation network may be destroying exactly the high-frequency boundary information that small-organ segmentation depends on. Rather than accepting this as a necessary computational trade-off, I asked whether preserving native CT resolution through patch-based processing could recover what downsampling loses -- even with a simpler architecture.

The answer was unambiguous. My resolution-preserving framework, built on a straightforward encoder-decoder CNN operating on 512x512 patches, achieves a Dice score of 0.849 on the Medical Segmentation Decathlon Task07 Pancreas benchmark. This outperforms both nnU-Net (0.823 Dice), the dominant automated segmentation pipeline, and Vision Transformer architectures that I implemented and compared against. The margin is not trivial: it represents a meaningful improvement on one of the most challenging segmentation tasks in the field, and it comes from a model with fewer parameters and no architecture search.

I did not stop at the result. To understand *why* resolution matters, I conducted a Fourier frequency analysis of CT patches at multiple resolutions, demonstrating quantitatively that downsampling destroys high-frequency components corresponding to organ boundaries -- precisely the features a segmentation model needs most for small, irregularly shaped structures like the pancreas. I also performed systematic ablation studies, including HU windowing optimization that yielded a 4.94% Dice improvement by restricting the Hounsfield Unit range to soft-tissue contrast, and a full architectural comparison across CNNs, Vision Transformers, and nnU-Net under controlled conditions. Each experiment was hypothesis-driven, designed to isolate a specific variable, and run on an HPC cluster using NVIDIA V100 GPUs under SLURM job scheduling.

## Data Efficiency and Generalization

The clinical reality of medical imaging is that labeled data is expensive and institutionally siloed. This motivated the second phase of my research: implementing and evaluating three semi-supervised learning methods -- Mean Teacher, Uncertainty-Aware Mean Teacher (UA-MT), and Cross Pseudo Supervision (CPS) -- to determine how much annotation effort can be eliminated without sacrificing segmentation quality. My experiments showed that UA-MT recovers 94.5% of fully supervised performance using only 50% of the labeled training data, a finding with direct implications for clinical deployment where expert annotation time is the primary bottleneck.

More importantly, I tested generalization. When I evaluated my models on 80 CT volumes from the external TCIA Pancreas-CT dataset -- data from a different institution, scanner protocol, and patient population -- the semi-supervised model actually outperformed the fully supervised baseline (0.603 vs. 0.544 Dice). This result suggests that semi-supervised learning acts as an implicit regularizer, and it reinforced my conviction that data-efficient methods are not just practically convenient but may be fundamentally better at learning transferable representations. I am currently preparing these findings for submission to a Q1 medical imaging journal.

## Future Research Directions

My PhD research will extend the principles I have established -- resolution preservation, data efficiency, and rigorous generalization testing -- into two directions. First, I want to investigate self-supervised pretraining strategies for volumetric medical data, particularly methods that learn anatomical priors from unlabeled CT and MRI scans before fine-tuning on small labeled datasets. The gap between natural-image foundation models and their medical counterparts remains wide, and I believe the path forward requires pretraining objectives designed specifically for the spatial and contrast properties of clinical imaging. Second, I am interested in computational pathology, where gigapixel whole-slide images present resolution challenges analogous to those I have addressed in CT -- and where patch-based processing is not a workaround but a necessity.

## Fit and Goals

I am looking for a PhD program with an active medical image analysis group that values methodological rigor and clinical relevance in equal measure. I want an advisor and lab environment where ideas are tested through controlled experiments, where negative results are analyzed rather than discarded, and where the goal is not simply state-of-the-art performance on a leaderboard but understanding *why* a method works and *when* it will fail. My experience designing and running large-scale experiments independently, debugging training pipelines on HPC infrastructure, and writing results for peer review has prepared me to contribute to such a group from the first year.

The question that drives me has not changed since I first looked at a pancreas CT scan: how do we build AI systems that a radiologist would actually trust? I believe the answer lies not in architectural novelty for its own sake, but in principled, data-efficient methods validated on data the model was never designed to see. That is the research I intend to pursue.
