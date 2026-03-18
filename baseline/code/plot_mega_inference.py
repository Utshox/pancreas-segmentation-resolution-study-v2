import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2

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
    preds = model.predict(patches, batch_size=16, verbose=0)
    for i, (y, x) in enumerate(coords):
        pred_patch = preds[i, :, :, 0]
        prediction_map[y:y+patch_size, x:x+patch_size] += pred_patch
        count_map[y:y+patch_size, x:x+patch_size] += 1.0
    avg_pred = prediction_map / np.maximum(count_map, 1.0)
    return avg_pred[:h, :w]

def predict_ablation_256(model, image):
    # Resize to 256x256
    img_256 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    img_256 = np.expand_dims(img_256, axis=(0, -1))
    pred_256 = model.predict(img_256, verbose=0)[0, :, :, 0]
    # Resize back to original
    pred_full = cv2.resize(pred_256, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return pred_full

def main():
    print("Loading Models...")
    model_sota = get_unet()
    model_sota.load_weights("baseline/models/model_patch_best.h5")
    
    model_ablation_256 = get_unet(img_size=256)
    model_ablation_256.load_weights("baseline/models/ablation_256/model_ablation_best.h5")
    
    model_uamt_25 = get_unet()
    model_uamt_25.load_weights("baseline/models/ssl_uamt_25/standalone_best.h5")
    
    model_mt_50 = get_unet()
    model_mt_50.load_weights("baseline/models/ssl_meanteacher_50/standalone_best.h5")
    
    model_uamt_50 = get_unet()
    model_uamt_50.load_weights("baseline/models/ssl_uamt_50/standalone_best.h5")

    print("Loading Volume...")
    img_path = "/scratch/lustre/home/kayi9958/ish/data_val/imagesTr/pancreas_005.nii.gz"
    lbl_path = "/scratch/lustre/home/kayi9958/ish/data_val/labelsTr/pancreas_005.nii.gz"
    
    vol_img = nib.load(img_path).get_fdata()
    vol_lbl = nib.load(lbl_path).get_fdata()
    
    areas = []
    if vol_lbl.shape[0] == 512 and vol_lbl.shape[1] == 512:
        for s in range(vol_lbl.shape[2]):
            areas.append(np.sum(vol_lbl[:, :, s] > 0))
    else:
        for s in range(vol_lbl.shape[0]):
            areas.append(np.sum(vol_lbl[s, :, :] > 0))
            
    best_slice_idx = np.argmax(areas)
    slice_indices = [best_slice_idx - 3, best_slice_idx, best_slice_idx + 3]
    
    columns = [
        "CT Image", 
        "Ground Truth", 
        "Ablation (256x256)", 
        "SOTA (100%)", 
        "UA-MT (25%)", 
        "Mean Teacher (50%)", 
        "UA-MT (50%)"
    ]
    
    fig, axes = plt.subplots(len(slice_indices), len(columns), figsize=(24, 10))
    fig.suptitle("Qualitative Inference Across Architectures and Data Regimes (Pancreas_005)", fontsize=22, fontweight='bold', y=1.02)
    
    for row, s_idx in enumerate(slice_indices):
        if vol_lbl.shape[0] == 512 and vol_lbl.shape[1] == 512:
            slice_img = vol_img[:, :, s_idx]
            slice_lbl = vol_lbl[:, :, s_idx]
        else:
            slice_img = vol_img[s_idx, :, :]
            slice_lbl = vol_lbl[s_idx, :, :]
            
        # Preprocess
        HU_MIN, HU_MAX = -125, 275
        slice_img_clipped = np.clip(slice_img, HU_MIN, HU_MAX)
        slice_img_norm = (slice_img_clipped - HU_MIN) / (HU_MAX - HU_MIN)
        slice_img_norm = np.clip(slice_img_norm, 0, 1)
        
        # Inference
        pred_ablation_bin = (predict_ablation_256(model_ablation_256, slice_img_norm) > 0.5).astype(np.float32)
        pred_sota_bin = (predict_sliding_window(model_sota, slice_img_norm) > 0.5).astype(np.float32)
        pred_uamt_25_bin = (predict_sliding_window(model_uamt_25, slice_img_norm) > 0.5).astype(np.float32)
        pred_mt_50_bin = (predict_sliding_window(model_mt_50, slice_img_norm) > 0.5).astype(np.float32)
        pred_uamt_50_bin = (predict_sliding_window(model_uamt_50, slice_img_norm) > 0.5).astype(np.float32)
        
        preds = [None, slice_lbl, pred_ablation_bin, pred_sota_bin, pred_uamt_25_bin, pred_mt_50_bin, pred_uamt_50_bin]
        
        for col, (ax, pred) in enumerate(zip(axes[row], preds)):
            ax.imshow(slice_img_norm, cmap='gray')
            if pred is not None:
                cmap = 'Greens' if col == 1 else 'Reds'
                ax.imshow(pred, cmap=cmap, alpha=0.5 * (pred > 0))
            
            if row == 0:
                ax.set_title(columns[col], fontsize=16, fontweight='bold', pad=15)
            
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Give space for the suptitle
    
    os.makedirs("baseline/logs/verification/plots", exist_ok=True)
    out_path = "baseline/logs/verification/plots/mega_inference_comparison_005.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved mega plot to {out_path}")

if __name__ == "__main__":
    main()
