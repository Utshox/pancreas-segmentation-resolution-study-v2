import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import argparse
from pathlib import Path

from transformer_unet import create_transformer_unet

# Lower batch size required due to heavy memory footprint of Self-Attention
BATCH_SIZE = 16 
EPOCHS = 100
LR = 1e-4

def load_dataset_into_ram(data_dir):
    print(f"Loading 512x512 SOTA Patches from {data_dir}...")
    x_files = sorted(list(Path(data_dir).glob('*_x.npy')))
    y_files = sorted(list(Path(data_dir).glob('*_y.npy')))
    
    all_x = []
    all_y = []
    
    for xf, yf in zip(x_files, y_files):
        try:
            bx = np.load(xf)
            by = np.load(yf)
            
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
    
    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)
        Y = np.expand_dims(Y, axis=-1)
    
    Y = np.clip(Y, 0, 1) # Force binary labels
    
    print(f"Dataset Loaded: X={X.shape}, Y={Y.shape}")
    return X, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load Data
    X, Y = load_dataset_into_ram(args.data_dir)
    
    # Split
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    # Build Transformer Baseline
    model = create_transformer_unet(input_shape=(256, 256, 1))
    
    # AdamW or standard Adam works well. Using Adam to match SOTA setup.
    model.compile(optimizer=optimizers.Adam(learning_rate=LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='io_u')])
                  
    # Callbacks
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'model_transformer_best.h5'), 
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
