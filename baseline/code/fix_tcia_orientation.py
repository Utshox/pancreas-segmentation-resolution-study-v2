import os
import glob
import nibabel as nib
import numpy as np

def fix_orientation(image_dir, label_dir):
    print("--- Fixing Z-Axis Orientation Mismatch in TCIA Dataset ---")
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))
    
    fixed_count = 0
    for lbl_path in label_files:
        case_name = os.path.basename(lbl_path)
        img_path = os.path.join(image_dir, case_name)
        
        if not os.path.exists(img_path):
            continue
            
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        
        img_z_dir = np.sign(img_nii.affine[2, 2])
        lbl_z_dir = np.sign(lbl_nii.affine[2, 2])
        
        # If the Z-axis orientation is flipped (e.g. 1.0 vs -1.0)
        if img_z_dir != lbl_z_dir:
            print(f"Fixing mismatch in {case_name} (Image Z: {img_z_dir}, Label Z: {lbl_z_dir})")
            
            # Get raw label data
            lbl_data = lbl_nii.get_fdata()
            
            # Flip the label data along the Z axis (axis=2)
            lbl_data_flipped = np.flip(lbl_data, axis=2)
            
            # Save the flipped label using the IMAGE's affine matrix so they perfectly align
            fixed_nii = nib.Nifti1Image(lbl_data_flipped, img_nii.affine)
            nib.save(fixed_nii, lbl_path)
            fixed_count += 1
            
    print(f"--- Fixed {fixed_count} label files. ---")

if __name__ == '__main__':
    IMAGE_DIR = "/scratch/lustre/home/kayi9958/ish/data_external/imagesTs"
    LABEL_DIR = "/scratch/lustre/home/kayi9958/ish/data_external/labelsTs"
    fix_orientation(IMAGE_DIR, LABEL_DIR)