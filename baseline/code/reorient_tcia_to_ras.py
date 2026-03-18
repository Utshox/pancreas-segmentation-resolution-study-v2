import os
import glob
import nibabel as nib
import numpy as np

def process_directory(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.nii.gz")))
    print(f"--- Reorienting {len(files)} files in {directory} to RAS ---")
    
    fixed = 0
    for file_path in files:
        nii = nib.load(file_path)
        axcodes = nib.aff2axcodes(nii.affine)
        
        if axcodes != ('R', 'A', 'S'):
            print(f"{os.path.basename(file_path)}: {axcodes} -> ('R', 'A', 'S')")
            # nibabel built-in for reorienting to standard (RAS) coordinate system
            new_nii = nib.as_closest_canonical(nii)
            
            # Save it back
            nib.save(new_nii, file_path)
            fixed += 1
            
    print(f"--- Fixed {fixed} files. ---")

if __name__ == '__main__':
    IMAGE_DIR = "/scratch/lustre/home/kayi9958/ish/data_external/imagesTs"
    LABEL_DIR = "/scratch/lustre/home/kayi9958/ish/data_external/labelsTs"
    
    print("Processing Images...")
    process_directory(IMAGE_DIR)
    
    print("Processing Labels...")
    process_directory(LABEL_DIR)
