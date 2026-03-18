import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

# --- Model Definition ---
def get_unet(img_size=256, num_classes=1):
    inputs = layers.Input(shape=(img_size, img_size, 1))
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    return Model(inputs=[inputs], outputs=[outputs])

def predict_sliding_window(model, image, patch_size=256, stride=128):
    h, w = image.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
    h_pad, w_pad = image_padded.shape
    prediction_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    count_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    patches = []
    coords = []
    for y in range(0, h_pad - patch_size + 1, stride):
        for x in range(0, w_pad - patch_size + 1, stride):
            patch = image_padded[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
    if not patches: return prediction_map[:h, :w]
    patches = np.array(patches)
    patches = np.expand_dims(patches, axis=-1)
    preds = model.predict(patches, batch_size=32, verbose=0)
    for i, (y, x) in enumerate(coords):
        pred_patch = preds[i, :, :, 0]
        prediction_map[y:y+patch_size, x:x+patch_size] += pred_patch
        count_map[y:y+patch_size, x:x+patch_size] += 1.0
    avg_pred = prediction_map / np.maximum(count_map, 1.0)
    return avg_pred[:h, :w]

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def analyze_stability(img_path, lbl_path, model_path, case_name):
    print(f"--- Analyzing Stability for {case_name} ---")
    img_vol = nib.load(img_path).get_fdata()
    lbl_vol = nib.load(lbl_path).get_fdata()
    
    model = get_unet()
    model.load_weights(model_path)
    
    slice_dices = []
    slice_indices = []
    
    HU_MIN, HU_MAX = -125, 275
    
    # Analyze all slices that have foreground labels
    for s in range(img_vol.shape[2]):
        if np.sum(lbl_vol[:, :, s] > 0) > 10: # Minimum 10 pixels to be a valid pancreas slice
            slice_img = img_vol[:, :, s]
            slice_lbl = lbl_vol[:, :, s]
            
            # Preprocess
            slice_clipped = np.clip(slice_img, HU_MIN, HU_MAX)
            slice_norm = (slice_clipped - HU_MIN) / (HU_MAX - HU_MIN)
            slice_norm = np.clip(slice_norm, 0, 1)
            
            # Predict
            pred = predict_sliding_window(model, slice_norm)
            pred_bin = (pred > 0.5).astype(np.float32)
            
            # Calculate Dice
            d = dice_coefficient(slice_lbl, pred_bin)
            slice_dices.append(d)
            slice_indices.append(s)
            
            if s % 10 == 0:
                print(f"Slice {s}: Dice {d:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(slice_indices, slice_dices, marker='o', linestyle='-', color='darkblue', linewidth=2, markersize=4)
    plt.fill_between(slice_indices, slice_dices, alpha=0.2, color='blue')
    
    mean_dice = np.mean(slice_dices)
    std_dice = np.std(slice_dices)
    
    plt.axhline(mean_dice, color='red', linestyle='--', label=f'Mean Dice: {mean_dice:.4f}')
    plt.title(f'3D Spatial Stability Analysis: {case_name}\n(Dice Variance: {std_dice:.4f})', fontsize=16, fontweight='bold')
    plt.xlabel('Axial Slice Index', fontsize=14)
    plt.ylabel('2D Dice Similarity Coefficient', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    out_dir = 'baseline/logs/verification/plots'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'stability_analysis_{case_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved stability plot to {out_path}")
    print(f"Mean: {mean_dice:.4f}, Std: {std_dice:.4f}")

if __name__ == '__main__':
    # Analyze MSD case 001 first for baseline stability
    analyze_stability(
        '/scratch/lustre/home/kayi9958/ish/data_val/imagesTr/pancreas_001.nii.gz',
        '/scratch/lustre/home/kayi9958/ish/data_val/labelsTr/pancreas_001.nii.gz',
        'baseline/models/model_patch_best.h5',
        'msd_001_stability'
    )
    
    # Analyze high-performing TCIA case 052 for generalization stability
    analyze_stability(
        '/scratch/lustre/home/kayi9958/ish/data_external/imagesTs/pancreas_ext_052.nii.gz',
        '/scratch/lustre/home/kayi9958/ish/data_external/labelsTs/pancreas_ext_052.nii.gz',
        'baseline/models/model_patch_best.h5',
        'tcia_052_stability'
    )
