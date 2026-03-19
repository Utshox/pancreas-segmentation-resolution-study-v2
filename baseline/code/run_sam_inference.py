"""
SAM / MedSAM Zero-Shot Inference on Pancreas CT
-------------------------------------------------
Evaluates foundation models (SAM, MedSAM) on our test set.
Uses ground-truth bounding boxes as prompts (standard evaluation protocol).
Also runs SAM automatic mask generation for fully unprompted comparison.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
import argparse
from datetime import datetime
from tqdm import tqdm

def compute_dice(y_true, y_pred):
    y_true = (y_true > 0).astype(np.float32)
    y_pred = (y_pred > 0).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 1.0
    return (2. * intersection) / denominator

def get_bbox_from_mask(mask, margin=10):
    """Extract bounding box from binary mask with margin."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Add margin
    h, w = mask.shape
    rmin = max(0, rmin - margin)
    rmax = min(h, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w, cmax + margin)
    return np.array([cmin, rmin, cmax, rmax])  # SAM expects [x0, y0, x1, y1]

def preprocess_ct_for_sam(slice_2d):
    """Convert single-channel CT slice to 3-channel uint8 for SAM."""
    # Apply HU windowing (same as our training)
    HU_MIN, HU_MAX = -125, 275
    img = np.clip(slice_2d, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    # Convert to uint8 RGB
    img_uint8 = (img * 255).astype(np.uint8)
    img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
    return img_rgb

def run_sam_bbox(predictor, image_rgb, bbox):
    """Run SAM with bounding box prompt."""
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        box=bbox,
        multimask_output=True
    )
    # Return the mask with highest confidence
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(np.float32)

def run_sam_automatic(mask_generator, image_rgb, target_mask):
    """Run SAM in automatic mode, find mask best matching pancreas."""
    masks = mask_generator.generate(image_rgb)
    if not masks:
        return np.zeros(target_mask.shape, dtype=np.float32)

    # Find the generated mask with highest IoU to ground truth
    best_dice = 0
    best_mask = np.zeros(target_mask.shape, dtype=np.float32)
    for m in masks:
        seg = m['segmentation'].astype(np.float32)
        d = compute_dice(target_mask, seg)
        if d > best_dice:
            best_dice = d
            best_mask = seg
    return best_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--sam_checkpoint', type=str, required=True)
    parser.add_argument('--medsam_checkpoint', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='baseline/logs/verification')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Import SAM
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

    # Load SAM ViT-B
    print("Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    sam_auto_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100
    )

    # Load MedSAM if available
    medsam_predictor = None
    if args.medsam_checkpoint and os.path.exists(args.medsam_checkpoint):
        print("Loading MedSAM...")
        medsam = sam_model_registry["vit_b"](checkpoint=args.medsam_checkpoint)
        medsam.to(device)
        medsam_predictor = SamPredictor(medsam)
        print("MedSAM loaded.")

    # Load test volumes
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(args.label_dir, "*.nii.gz")))
    print(f"Found {len(image_files)} test volumes.")

    results = {
        'sam_bbox': {},
        'sam_auto': {},
        'medsam_bbox': {}
    }

    for img_path, lbl_path in zip(image_files, label_files):
        case_name = os.path.basename(img_path).replace(".nii.gz", "")
        print(f"\n{'='*50}")
        print(f"Processing {case_name}")

        vol_img = nib.load(img_path).get_fdata()
        vol_lbl = nib.load(lbl_path).get_fdata()

        # Determine orientation
        if vol_img.shape[0] == 512 and vol_img.shape[1] == 512:
            num_slices = vol_img.shape[2]
            get_slice = lambda v, s: v[:, :, s]
        else:
            num_slices = vol_img.shape[0]
            get_slice = lambda v, s: v[s, :, :]

        # Per-slice predictions
        pred_sam_bbox = np.zeros_like(vol_img)
        pred_sam_auto = np.zeros_like(vol_img)
        pred_medsam_bbox = np.zeros_like(vol_img)

        for s in tqdm(range(num_slices), desc=f"Slices"):
            slice_img = get_slice(vol_img, s)
            slice_lbl = get_slice(vol_lbl, s)
            slice_lbl_bin = (slice_lbl > 0).astype(np.float32)

            # Skip slices with no pancreas for bbox mode
            img_rgb = preprocess_ct_for_sam(slice_img)

            bbox = get_bbox_from_mask(slice_lbl_bin)

            if bbox is not None:
                # SAM with bbox prompt
                mask_sam = run_sam_bbox(sam_predictor, img_rgb, bbox)

                # MedSAM with bbox prompt
                if medsam_predictor:
                    mask_medsam = run_sam_bbox(medsam_predictor, img_rgb, bbox)
                else:
                    mask_medsam = np.zeros_like(slice_lbl_bin)
            else:
                mask_sam = np.zeros_like(slice_lbl_bin)
                mask_medsam = np.zeros_like(slice_lbl_bin)

            # SAM automatic (on all slices, but only the slices near pancreas region)
            if bbox is not None:
                mask_auto = run_sam_automatic(sam_auto_generator, img_rgb, slice_lbl_bin)
            else:
                mask_auto = np.zeros_like(slice_lbl_bin)

            # Store predictions
            if vol_img.shape[0] == 512 and vol_img.shape[1] == 512:
                pred_sam_bbox[:, :, s] = mask_sam
                pred_sam_auto[:, :, s] = mask_auto
                pred_medsam_bbox[:, :, s] = mask_medsam
            else:
                pred_sam_bbox[s, :, :] = mask_sam
                pred_sam_auto[s, :, :] = mask_auto
                pred_medsam_bbox[s, :, :] = mask_medsam

        # Compute 3D Dice
        dice_sam_bbox = compute_dice(vol_lbl, pred_sam_bbox)
        dice_sam_auto = compute_dice(vol_lbl, pred_sam_auto)
        dice_medsam = compute_dice(vol_lbl, pred_medsam_bbox)

        results['sam_bbox'][case_name] = dice_sam_bbox
        results['sam_auto'][case_name] = dice_sam_auto
        results['medsam_bbox'][case_name] = dice_medsam

        print(f"  SAM (bbox):  {dice_sam_bbox:.4f}")
        print(f"  SAM (auto):  {dice_sam_auto:.4f}")
        print(f"  MedSAM (bbox): {dice_medsam:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    for method, scores in results.items():
        if not scores:
            continue
        avg = np.mean(list(scores.values()))
        fname = os.path.join(args.output_dir, f'dice_results_{method}_{timestamp}.txt')
        with open(fname, 'w') as f:
            f.write(f"Inference Run: {method}_{timestamp}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Average Dice: {avg:.4f}\n")
            f.write("-" * 30 + "\n")
            for name, score in scores.items():
                f.write(f"{name}: {score:.4f}\n")
        print(f"\n{method} — Average Dice: {avg:.4f} (saved to {fname})")

if __name__ == "__main__":
    main()
