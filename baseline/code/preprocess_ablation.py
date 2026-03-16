"""
Resolution Ablation Preprocessing
---------------------------------
Generates datasets at 256x256 and 128x128 (resized full slices) 
to compare against our 512x512 Patch-based SOTA.
"""
import os
import shutil
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
from tqdm import tqdm
import skimage.transform

def process_case(case_name, output_base, img_path, lab_path, resolutions=[256, 128]):
    try:
        # Load NIfTI
        img_obj = nib.load(str(img_path))
        lab_obj = nib.load(str(lab_path))
        
        img_data = img_obj.get_fdata() # (H, W, D)
        lab_data = lab_obj.get_fdata()

        # HU Windowing [-125, 275] (Match V6 Champion)
        HU_MIN, HU_MAX = -125, 275
        img_data = np.clip(img_data, HU_MIN, HU_MAX)
        img_data = (img_data - HU_MIN) / (HU_MAX - HU_MIN)
        img_data = np.clip(img_data, 0.0, 1.0)
        
        num_slices = img_data.shape[2]

        for res in resolutions:
            res_dir = output_base / f"res_{res}"
            res_dir.mkdir(parents=True, exist_ok=True)
            
            target_shape = (res, res)
            
            # We only save slices that contain pancreas + some background for efficiency
            # To match our patch-based v5, we want a balanced set but full slices.
            for i in range(num_slices):
                sl_img = img_data[:, :, i]
                sl_lab = lab_data[:, :, i]
                
                # Only save if there is pancreas (foreground) or 10% chance if background
                if np.sum(sl_lab) > 0 or np.random.rand() < 0.1:
                    # Resize
                    sl_img_r = skimage.transform.resize(sl_img, target_shape, order=1, preserve_range=True, anti_aliasing=True)
                    sl_lab_r = skimage.transform.resize(sl_lab, target_shape, order=0, preserve_range=True, anti_aliasing=False)
                    
                    # Save as individual slice to prevent RAM bloat
                    # Naming: pancreas_XXX_slice_YYY_x.npy
                    slice_id = f"{case_name}_sl_{i:03d}"
                    np.save(res_dir / f"{slice_id}_x.npy", sl_img_r.astype(np.float32))
                    np.save(res_dir / f"{slice_id}_y.npy", sl_lab_r.astype(np.uint8))
        
    except Exception as e:
        print(f"Error processing {case_name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True, help="Path to imagesTr and labelsTr")
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)
    
    img_dir = raw_path / 'imagesTr'
    label_dir = raw_path / 'labelsTr'
    
    cases = sorted(list(img_dir.glob('*.nii.gz')))
    print(f"Found {len(cases)} cases. Generating 256 and 128 ablation sets...")
    
    for case_file in tqdm(cases):
        case_name = case_file.name.replace('.nii.gz', '')
        lab_file = label_dir / case_file.name
        if lab_file.exists():
            process_case(case_name, out_path, case_file, lab_file)

if __name__ == "__main__":
    main()
