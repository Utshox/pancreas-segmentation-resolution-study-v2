import os
import zipfile
import requests
import io
import shutil
from pathlib import Path
import dicom2nifti

def download_tcia_dataset():
    base_dir = Path('/scratch/lustre/home/kayi9958/ish/data_external')
    img_dir = base_dir / 'imagesTs'
    lbl_dir = base_dir / 'labelsTs'
    tmp_dir = base_dir / 'tmp'
    
    for d in [img_dir, lbl_dir, tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 1. Download and Extract Labels
    print("--- Step 1: Downloading TCIA Labels ---")
    labels_url = 'https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_pancreas_labels-02-05-2017-1.zip'
    lbl_zip_path = tmp_dir / 'labels.zip'
    
    if not any(lbl_dir.iterdir()):
        print(f"Downloading labels from {labels_url}...")
        r = requests.get(labels_url, stream=True)
        with open(lbl_zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print("Extracting labels...")
        with zipfile.ZipFile(lbl_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir / 'labels_raw')
            
        # Move to labelsTs
        extracted_dir = tmp_dir / 'labels_raw'
        # The zip might contain a nested folder or just files
        nii_files = list(extracted_dir.rglob('*.nii.gz'))
        for f in nii_files:
            # Labels are usually named label0001.nii.gz
            # We will standardise to pancreas_ext_001.nii.gz
            num_str = ''.join(filter(str.isdigit, f.name))
            new_name = f'pancreas_ext_{int(num_str):03d}.nii.gz'
            shutil.move(str(f), lbl_dir / new_name)
        print(f"Moved {len(nii_files)} label files to {lbl_dir}")

    # 2. Download Images via TCIA API
    print("\n--- Step 2: Downloading TCIA Images (DICOM -> NIfTI) ---")
    manifest_url = 'https://www.cancerimagingarchive.net/wp-content/uploads/Pancreas-CT-20200910.tcia'
    r = requests.get(manifest_url)
    manifest_lines = r.text.strip().split('\n')
    
    # Extract UIDs (skipping headers)
    uids = [line.strip() for line in manifest_lines if line.strip() and not line.startswith('downloadServerUrl') and not line.startswith('includeAnnotation') and not line.startswith('noOfrRetry') and not line.startswith('databasketId') and not line.startswith('manifestVersion') and not line.startswith('ListOfSeriesToDownload')]
    
    print(f"Found {len(uids)} Series UIDs in manifest.")
    
    api_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
    
    for i, uid in enumerate(uids, 1):
        # We need to map UID to the patient number. 
        # The easiest way is to just assign them sequentially 001-082.
        # But wait! We need the image to match the label!
        # This is tricky because the order in the .tcia file might NOT match label0001-label0082.
        # However, TCIA usually provides an API endpoint to get metadata.
        print(f"Downloading series {i}/{len(uids)}: {uid}")
        series_zip = tmp_dir / f"series_{i}.zip"
        
        # Download DICOM zip
        if not series_zip.exists():
            r_img = requests.get(api_url, params={"SeriesInstanceUID": uid}, stream=True)
            if r_img.status_code == 200:
                with open(series_zip, 'wb') as f:
                    for chunk in r_img.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                print(f"Failed to download {uid}")
                continue
                
        # Extract DICOMs
        dicom_dir = tmp_dir / f"dicom_{i}"
        dicom_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(series_zip, 'r') as z:
                z.extractall(dicom_dir)
        except Exception as e:
            print(f"Error unzipping {series_zip}: {e}")
            continue
            
        # Convert to NIfTI
        try:
            # Output directly to the tmp dir, dicom2nifti names the file based on series info
            out_nifti_dir = tmp_dir / f"nifti_{i}"
            out_nifti_dir.mkdir(exist_ok=True)
            dicom2nifti.convert_directory(str(dicom_dir), str(out_nifti_dir), compression=True, reorient=True)
            
            # Find the converted file
            nifti_files = list(out_nifti_dir.glob('*.nii.gz'))
            if nifti_files:
                # We need to extract the patient ID from the DICOM headers to match with the label.
                # Let's read the first dicom file
                import pydicom
                dcm_files = list(dicom_dir.rglob('*.dcm'))
                if dcm_files:
                    dcm = pydicom.dcmread(str(dcm_files[0]))
                    patient_id = dcm.PatientID # e.g., "PANCREAS_0001"
                    num_str = ''.join(filter(str.isdigit, patient_id))
                    new_name = f'pancreas_ext_{int(num_str):03d}.nii.gz'
                    
                    shutil.move(str(nifti_files[0]), img_dir / new_name)
                    print(f"Successfully converted and saved as {new_name}")
        except Exception as e:
            print(f"Error converting {uid}: {e}")
            
    print("--- Download and Conversion Complete! ---")

if __name__ == '__main__':
    download_tcia_dataset()