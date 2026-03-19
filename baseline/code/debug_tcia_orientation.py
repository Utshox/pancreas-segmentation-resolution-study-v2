import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
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

def main():
    img_path = '/scratch/lustre/home/kayi9958/ish/data_external/imagesTs/pancreas_ext_008.nii.gz'
    lbl_path = '/scratch/lustre/home/kayi9958/ish/data_external/labelsTs/pancreas_ext_008.nii.gz'

    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)
    img_vol = img_nii.get_fdata()
    lbl_vol = lbl_nii.get_fdata()

    print(f"Image shape: {img_vol.shape}")
    print(f"Label shape: {lbl_vol.shape}")

    # Find the slice with the maximum label area
    areas = [np.sum(lbl_vol[:, :, s] > 0) for s in range(lbl_vol.shape[2])]
    best_slice_idx = np.argmax(areas)
    print(f"Best slice index (label): {best_slice_idx}")

    slice_img = img_vol[:, :, best_slice_idx]
    slice_lbl = lbl_vol[:, :, best_slice_idx]

    # Model Inference
    model = get_unet()
    model.load_weights("baseline/models/model_patch_best.h5")
    
    HU_MIN, HU_MAX = -125, 275
    slice_img_clipped = np.clip(slice_img, HU_MIN, HU_MAX)
    slice_img_norm = (slice_img_clipped - HU_MIN) / (HU_MAX - HU_MIN)
    slice_img_norm = np.clip(slice_img_norm, 0, 1)

    pred = predict_sliding_window(model, slice_img_norm)
    pred_bin = (pred > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(slice_img_norm, cmap='gray')
    axes[0].set_title('Raw Image Slice')
    
    axes[1].imshow(slice_img_norm, cmap='gray')
    axes[1].imshow(slice_lbl, cmap='Reds', alpha=0.5 * (slice_lbl > 0))
    axes[1].set_title('Ground Truth Label (Red)')

    axes[2].imshow(slice_img_norm, cmap='gray')
    axes[2].imshow(pred_bin, cmap='Blues', alpha=0.5 * (pred_bin > 0))
    axes[2].set_title('Model Prediction (Blue)')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    out_path = 'baseline/logs/verification/plots/tcia_debug_008.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved debug plot to {out_path}")

if __name__ == '__main__':
    main()
