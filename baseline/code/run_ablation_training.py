"""
Resolution Ablation Training
----------------------------------------
This script implements full-slice training for the ablation study.
It takes a variable `--img_size` argument (e.g. 256 or 128).
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import argparse
from pathlib import Path

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4

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

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def load_dataset_into_ram(data_dir):
    print(f"Loading Dataset into RAM from {data_dir}...")
    x_files = sorted(list(Path(data_dir).glob('*_x.npy')))
    y_files = sorted(list(Path(data_dir).glob('*_y.npy')))
    
    all_x = []
    all_y = []
    
    for xf, yf in zip(x_files, y_files):
        try:
            bx = np.load(xf)
            by = np.load(yf)
            
            # The preprocessing script saves individual slices as (H, W) or (N, H, W)
            # In ablation prep, we saved them as individual slices, so bx might be (H, W).
            # If so, expand dims to (1, H, W).
            if bx.ndim == 2:
                bx = np.expand_dims(bx, axis=0)
                by = np.expand_dims(by, axis=0)
            
            if len(bx) > 0:
                all_x.append(bx)
                all_y.append(by)
        except:
            pass
            
    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    
    # Add channel dim
    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)
        Y = np.expand_dims(Y, axis=-1)
    
    Y = np.clip(Y, 0, 1) # Force binary labels (0, 1) to prevent IoU crash
    
    print(f"Dataset Loaded: X={X.shape}, Y={Y.shape}")
    return X, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, required=True, help="256 or 128")
    args = parser.parse_args()
    
    # Load Data
    X, Y = load_dataset_into_ram(args.data_dir)
    
    # Split
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    # Model
    model = get_unet(img_size=args.img_size)
    model.compile(optimizer=optimizers.Adam(learning_rate=LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='io_u')])
                  
    # Callbacks
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'model_ablation_best.h5'), 
                                 save_best_only=True, monitor='val_io_u', mode='max')
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, CSVLogger(os.path.join(args.output_dir, 'log.csv'))]
    )

if __name__ == "__main__":
    main()
