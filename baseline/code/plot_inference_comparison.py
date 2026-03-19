import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Model Definition ---
def get_unet(img_size=256, num_classes=1):
    inputs = layers.Input(shape=(img_size, img_size, 1))
    # Encoder
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
    # Decoder
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
    patches = np.array(patches)
    patches = np.expand_dims(patches, axis=-1)
    preds = model.predict(patches, batch_size=16, verbose=0)
    for i, (y, x) in enumerate(coords):
        pred_patch = preds[i, :, :, 0]
        prediction_map[y:y+patch_size, x:x+patch_size] += pred_patch
        count_map[y:y+patch_size, x:x+patch_size] += 1.0
    avg_pred = prediction_map / np.maximum(count_map, 1.0)
    return avg_pred[:h, :w]

def main():
    print("Loading Models...")
    model_sota = get_unet()
    model_sota.load_weights("baseline/models/model_patch_best.h5")
    
    model_uamt = get_unet()
    model_uamt.load_weights("baseline/models/ssl_uamt_50/standalone_best.h5")
    
    # Load Pancreas_005
    print("Loading Volume...")
    img_path = "/scratch/lustre/home/kayi9958/ish/data_val/imagesTr/pancreas_005.nii.gz"
    lbl_path = "/scratch/lustre/home/kayi9958/ish/data_val/labelsTr/pancreas_005.nii.gz"
    
    vol_img = nib.load(img_path).get_fdata()
    vol_lbl = nib.load(lbl_path).get_fdata()
    
    # Find slice with largest pancreas
    areas = []
    if vol_lbl.shape[0] == 512 and vol_lbl.shape[1] == 512:
        for s in range(vol_lbl.shape[2]):
            areas.append(np.sum(vol_lbl[:, :, s] > 0))
        best_slice_idx = np.argmax(areas)
        slice_img = vol_img[:, :, best_slice_idx]
        slice_lbl = vol_lbl[:, :, best_slice_idx]
    else:
        for s in range(vol_lbl.shape[0]):
            areas.append(np.sum(vol_lbl[s, :, :] > 0))
        best_slice_idx = np.argmax(areas)
        slice_img = vol_img[best_slice_idx, :, :]
        slice_lbl = vol_lbl[best_slice_idx, :, :]
        
    print(f"Best slice index: {best_slice_idx}")
    
    # Preprocess
    HU_MIN, HU_MAX = -125, 275
    slice_img_clipped = np.clip(slice_img, HU_MIN, HU_MAX)
    slice_img_norm = (slice_img_clipped - HU_MIN) / (HU_MAX - HU_MIN)
    slice_img_norm = np.clip(slice_img_norm, 0, 1)
    
    # Inference
    print("Running Inference SOTA...")
    pred_sota = predict_sliding_window(model_sota, slice_img_norm)
    pred_sota_bin = (pred_sota > 0.5).astype(np.float32)
    
    print("Running Inference UA-MT...")
    pred_uamt = predict_sliding_window(model_uamt, slice_img_norm)
    pred_uamt_bin = (pred_uamt > 0.5).astype(np.float32)
    
    # Plotting
    print("Plotting...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. CT Image
    axes[0].imshow(slice_img_norm, cmap='gray')
    axes[0].set_title("CT Slice", fontsize=14)
    axes[0].axis('off')
    
    # 2. Ground Truth
    axes[1].imshow(slice_img_norm, cmap='gray')
    axes[1].imshow(slice_lbl, cmap='Reds', alpha=0.5 * (slice_lbl > 0))
    axes[1].set_title("Ground Truth", fontsize=14)
    axes[1].axis('off')
    
    # 3. Supervised SOTA
    axes[2].imshow(slice_img_norm, cmap='gray')
    axes[2].imshow(pred_sota_bin, cmap='Blues', alpha=0.5 * (pred_sota_bin > 0))
    axes[2].set_title("Fully Supervised (100%)", fontsize=14)
    axes[2].axis('off')
    
    # 4. UA-MT 50%
    axes[3].imshow(slice_img_norm, cmap='gray')
    axes[3].imshow(pred_uamt_bin, cmap='Greens', alpha=0.5 * (pred_uamt_bin > 0))
    axes[3].set_title("UA-MT (50% Labels)", fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    os.makedirs("baseline/logs/verification/plots", exist_ok=True)
    plt.savefig("baseline/logs/verification/plots/inference_comparison_005.png", dpi=300)
    print("Saved plot to baseline/logs/verification/plots/inference_comparison_005.png")

if __name__ == "__main__":
    main()
