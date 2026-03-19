import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    img_path = '/scratch/lustre/home/kayi9958/ish/data_external_btcv/averaged-training-images/DET0000101_avg.nii.gz'
    lbl_path = '/scratch/lustre/home/kayi9958/ish/data_external_btcv/averaged-training-labels/DET0000101_avg_seg.nii.gz'

    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)
    
    print(f"Image Affine Codes: {nib.aff2axcodes(img_nii.affine)}")
    print(f"Label Affine Codes: {nib.aff2axcodes(lbl_nii.affine)}")
    
    img_vol = img_nii.get_fdata()
    lbl_vol = lbl_nii.get_fdata()
    
    # Find slice with label
    slice_idx = np.argmax([np.sum(lbl_vol[:, :, s] > 0) for s in range(lbl_vol.shape[2])])
    print(f"Best Slice: {slice_idx}")
    
    slice_img = img_vol[:, :, slice_idx]
    slice_lbl = lbl_vol[:, :, slice_idx]
    
    HU_MIN, HU_MAX = -125, 275
    slice_img = np.clip(slice_img, HU_MIN, HU_MAX)
    slice_img = (slice_img - HU_MIN) / (HU_MAX - HU_MIN)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(slice_img, cmap='gray')
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(slice_img, cmap='gray')
    plt.imshow(slice_lbl, cmap='Reds', alpha=0.5 * (slice_lbl > 0))
    plt.title("Label (Red)")
    
    out_path = 'baseline/logs/verification/plots/btcv_debug_001.png'
    plt.savefig(out_path)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
