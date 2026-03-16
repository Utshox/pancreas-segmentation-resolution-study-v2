
"""
Phase 6: Sliding Window Inference
---------------------------------
Reconstructs 512x512 predictions from the trained Patch U-Net.
"""
import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers
import argparse
from tqdm import tqdm
from datetime import datetime

# --- Model Definition (Must match training) ---
def get_unet(img_size=256, num_classes=1):
    inputs = layers.Input(shape=(img_size, img_size, 1))
    # ... (same as before)
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
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

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
    patches = np.array(patches)
    patches = np.expand_dims(patches, axis=-1)
    preds = model.predict(patches, batch_size=16, verbose=0)
    for i, (y, x) in enumerate(coords):
        pred_patch = preds[i, :, :, 0]
        prediction_map[y:y+patch_size, x:x+patch_size] += pred_patch
        count_map[y:y+patch_size, x:x+patch_size] += 1.0
    avg_pred = prediction_map / np.maximum(count_map, 1.0)
    return avg_pred[:h, :w]

def compute_dice(y_true, y_pred):
    # Ensure binary
    y_true = (y_true > 0).astype(np.float32)
    y_pred = (y_pred > 0).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 1.0  # Perfect match for empty-empty
    return (2. * intersection) / denominator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--exp_name', type=str, default='baseline_v6')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.exp_name}_{timestamp}"
    
    print(f"Starting Inference Run: {run_id}")
    print("Loading Model...")
    model = get_unet()
    model.load_weights(args.model_path)
    
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(args.label_dir, "*.nii.gz")))
    
    print(f"Found {len(image_files)} test images.")
    dice_scores = {}
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for img_path, lbl_path in zip(image_files, label_files):
        case_name = os.path.basename(img_path).replace(".nii.gz", "")
        print(f"Processing {case_name}...")
        
        nii_img = nib.load(img_path)
        nii_lbl = nib.load(lbl_path)
        vol_img = nii_img.get_fdata()
        vol_lbl = nii_lbl.get_fdata()
        
        # STANDARD: Use standard HU windowing [-100, 240] for comparison
        HU_MIN, HU_MAX = -100, 240
        vol_img = np.clip(vol_img, HU_MIN, HU_MAX)
        vol_img = (vol_img - HU_MIN) / (HU_MAX - HU_MIN)
        vol_img = np.clip(vol_img, 0, 1)
        
        # Determine orientation
        if vol_img.shape[0] == 512 and vol_img.shape[1] == 512:
            num_slices = vol_img.shape[2]
            pred_vol = np.zeros_like(vol_img)
            for s in tqdm(range(num_slices), desc=f"Slices for {case_name}"):
                slice_img = vol_img[:, :, s]
                pred_slice = predict_sliding_window(model, slice_img)
                pred_vol[:, :, s] = pred_slice
        else:
            num_slices = vol_img.shape[0]
            pred_vol = np.zeros_like(vol_img)
            for s in tqdm(range(num_slices), desc=f"Slices for {case_name}"):
                slice_img = vol_img[s, :, :]
                pred_slice = predict_sliding_window(model, slice_img)
                pred_vol[s, :, :] = pred_slice

        pred_bin = (pred_vol > 0.5).astype(np.float32)
        dice = compute_dice(vol_lbl, pred_bin)
        dice_scores[case_name] = dice
        print(f"Case {case_name} Dice: {dice:.4f}")
        
    avg_dice = np.mean(list(dice_scores.values()))
    print(f"\nAverage Dice: {avg_dice:.4f}")
    
    # Save results with unique identifier
    result_file = os.path.join(args.output_dir, f'dice_results_{run_id}.txt')
    with open(result_file, 'w') as f:
        f.write(f"Inference Run: {run_id}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Average Dice: {avg_dice:.4f}\n")
        f.write("-" * 30 + "\n")
        for name, score in dice_scores.items():
            f.write(f"{name}: {score:.4f}\n")
    
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
