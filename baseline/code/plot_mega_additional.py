"""
Generate additional mega qualitative comparison plots for:
1. Pancreas_001 (easy case, Dice ~0.90) — shows typical good performance
2. Pancreas_004 (medium case) — diversity of examples
"""

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
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

def generate_mega_plot(case_id, models_dict, img_path, lbl_path, out_path, title):
    """Generate a mega comparison plot for a given case."""
    print(f"Processing {case_id}...")

    vol_img = nib.load(img_path).get_fdata()
    vol_lbl = nib.load(lbl_path).get_fdata()

    # Find best slices
    areas = []
    if vol_lbl.shape[0] == 512 and vol_lbl.shape[1] == 512:
        axis = 'last'
        for s in range(vol_lbl.shape[2]):
            areas.append(np.sum(vol_lbl[:, :, s] > 0))
    else:
        axis = 'first'
        for s in range(vol_lbl.shape[0]):
            areas.append(np.sum(vol_lbl[s, :, :] > 0))

    best_slice_idx = np.argmax(areas)
    slice_indices = [best_slice_idx - 3, best_slice_idx, best_slice_idx + 3]

    columns = list(models_dict.keys())
    columns = ["CT Image", "Ground Truth"] + columns

    fig, axes = plt.subplots(len(slice_indices), len(columns), figsize=(18, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

    for row, s_idx in enumerate(slice_indices):
        if axis == 'last':
            slice_img = vol_img[:, :, s_idx]
            slice_lbl = vol_lbl[:, :, s_idx]
        else:
            slice_img = vol_img[s_idx, :, :]
            slice_lbl = vol_lbl[s_idx, :, :]

        # Preprocess
        HU_MIN, HU_MAX = -125, 275
        slice_img_clipped = np.clip(slice_img, HU_MIN, HU_MAX)
        slice_img_norm = (slice_img_clipped - HU_MIN) / (HU_MAX - HU_MIN)

        # Run all models
        predictions = {}
        for name, model in models_dict.items():
            pred = predict_sliding_window(model, slice_img_norm)
            predictions[name] = (pred > 0.5).astype(np.float32)

        # Plot
        all_preds = [None, slice_lbl] + [predictions[k] for k in models_dict.keys()]

        for col, (ax, pred) in enumerate(zip(axes[row], all_preds)):
            ax.imshow(slice_img_norm, cmap='gray')
            if pred is not None:
                cmap = 'Greens' if col == 1 else 'Reds'
                ax.imshow(pred, cmap=cmap, alpha=0.5 * (pred > 0))
            if row == 0:
                ax.set_title(columns[col], fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Loading Models...")

    model_sota = get_unet()
    model_sota.load_weights("baseline/models/model_patch_best.h5")

    model_uamt_50 = get_unet()
    model_uamt_50.load_weights("baseline/models/ssl_uamt_50/standalone_best.h5")

    model_mt_50 = get_unet()
    model_mt_50.load_weights("baseline/models/ssl_meanteacher_50/standalone_best.h5")

    model_uamt_25 = get_unet()
    model_uamt_25.load_weights("baseline/models/ssl_uamt_25/standalone_best.h5")

    models_dict = {
        "SOTA (100%)": model_sota,
        "UA-MT (25%)": model_uamt_25,
        "Mean Teacher (50%)": model_mt_50,
        "UA-MT (50%)": model_uamt_50,
    }

    DATA_DIR = "/scratch/lustre/home/kayi9958/ish/data_val"
    OUT_DIR = "baseline/logs/verification/plots"

    # Case 001 (Easy, Dice ~0.90)
    generate_mega_plot(
        "pancreas_001", models_dict,
        f"{DATA_DIR}/imagesTr/pancreas_001.nii.gz",
        f"{DATA_DIR}/labelsTr/pancreas_001.nii.gz",
        f"{OUT_DIR}/mega_inference_comparison_001.png",
        "Qualitative Comparison on Easy Case (Pancreas_001, Dice~0.90)"
    )

    # Case 004 (Medium, Dice ~0.84-0.88)
    generate_mega_plot(
        "pancreas_004", models_dict,
        f"{DATA_DIR}/imagesTr/pancreas_004.nii.gz",
        f"{DATA_DIR}/labelsTr/pancreas_004.nii.gz",
        f"{OUT_DIR}/mega_inference_comparison_004.png",
        "Qualitative Comparison on Medium Case (Pancreas_004, Dice~0.85)"
    )

    print("\nAll additional mega plots generated!")

if __name__ == "__main__":
    main()
