
"""
Phase 7: SSL Inference (FixMatch Support)
-----------------------------------------
Handles loading weights from the custom FixMatchTrainer wrapper.
"""
import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, models
import argparse
from tqdm import tqdm

# --- Model Definition (Must match training) ---
def get_unet(img_size=256, num_classes=1):
    inputs = layers.Input(shape=(img_size, img_size, 1))

    # Encoding
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

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoding
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

# --- FixMatch Wrapper Definition (Essential for loading weights) ---
class FixMatchTrainer(models.Model):
    def __init__(self, model, lambda_u=1.0, threshold=0.95):
        super(FixMatchTrainer, self).__init__()
        self.model = model
        self.lambda_u = lambda_u
        self.threshold = threshold
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, inputs):
        # We need call() for build() to work potentially, though training script didn't use it.
        # But load_weights might need structure verification.
        # Pass through to model
        return self.model(inputs)

    # We don't need train_step here for inference

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
    patches = np.expand_dims(patches, axis=-1) # (N, 256, 256, 1)
    
    # Predict in batch
    preds = model.predict(patches, batch_size=16, verbose=0)
    
    for i, (y, x) in enumerate(coords):
        pred_patch = preds[i, :, :, 0]
        prediction_map[y:y+patch_size, x:x+patch_size] += pred_patch
        count_map[y:y+patch_size, x:x+patch_size] += 1.0
        
    avg_pred = prediction_map / np.maximum(count_map, 1.0)
    return avg_pred[:h, :w]

def compute_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='inference_results')
    args = parser.parse_args()
    
    print("Loading FixMatch Model...")
    
    # 1. Instantiate Inner UNet
    unet = get_unet()
    
    # 2. Instantiate Wrapper
    trainer = FixMatchTrainer(unet)
    
    # 3. Build it to initialize variables (Critical for subclassed models)
    # Build by running dummy data or calling .build()
    # We implemented .call(), so we can build.
    # Input shape: (Batch, 256, 256, 1)
    trainer.build(input_shape=(None, 256, 256, 1))
    
    # 4. Load Weights
    print(f"Loading weights from {args.model_path}")
    trainer.load_weights(args.model_path)
    
    # 5. Extract inner model for prediction
    model = trainer.model
    
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(args.label_dir, "*.nii.gz")))
    
    print(f"Found {len(image_files)} test images.")
    
    dice_scores = []
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for img_path, lbl_path in zip(image_files, label_files):
        print(f"Processing {os.path.basename(img_path)}...")
        
        nii_img = nib.load(img_path)
        nii_lbl = nib.load(lbl_path)
        
        vol_img = nii_img.get_fdata()
        vol_lbl = nii_lbl.get_fdata()
        
        # Preprocess
        vol_img = np.clip(vol_img, -100, 240)
        vol_img = (vol_img - (-100)) / (240 - (-100))
        vol_img = np.clip(vol_img, 0, 1)
        
        if vol_img.shape[0] == 512 and vol_img.shape[1] == 512:
            num_slices = vol_img.shape[2]
            pred_vol = np.zeros_like(vol_img)
            for s in tqdm(range(num_slices)):
                slice_img = vol_img[:, :, s]
                pred_slice = predict_sliding_window(model, slice_img)
                pred_vol[:, :, s] = pred_slice
        else:
            num_slices = vol_img.shape[0]
            pred_vol = np.zeros_like(vol_img)
            for s in tqdm(range(num_slices)):
                slice_img = vol_img[s, :, :]
                pred_slice = predict_sliding_window(model, slice_img)
                pred_vol[s, :, :] = pred_slice

        pred_bin = (pred_vol > 0.5).astype(np.float32)
        dice = compute_dice(vol_lbl, pred_bin)
        dice_scores.append(dice)
        print(f"Dice: {dice:.4f}")
        
    print(f"Average Dice: {np.mean(dice_scores):.4f}")
    
    with open(os.path.join(args.output_dir, 'final_dice.txt'), 'w') as f:
        f.write(f"Average Dice: {np.mean(dice_scores):.4f}\n")
        for d in dice_scores:
            f.write(f"{d}\n")

if __name__ == "__main__":
    main()
