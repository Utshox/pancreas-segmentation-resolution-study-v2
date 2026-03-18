#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(input_shape)
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

    # Bottleneck
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

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return models.Model(inputs=[inputs], outputs=[outputs])

def extract_weights_h5(model_type, checkpoint_path, out_path):
    print(f"Direct HDF5 Extraction for {model_type}...")
    f = h5py.File(checkpoint_path, 'r')
    
    # In MT and UA-MT, model_1 is typically the teacher (second model instantiated)
    # In CPS, we just want one of the two models. model_1 is fine.
    target_group = f['model_1']
    
    target_model = build_unet()
    
    # Keras models save layer weights using their internal names (e.g. conv2d_19)
    # We must sort them naturally to match the model we just built
    saved_layers = [name for name in target_group.keys() if 'conv2d' in name]
    
    def layer_sort_key(name):
        parts = name.split('_')
        if len(parts) == 1 or parts[-1] == 'transpose':
            return 0
        try:
            return int(parts[-1])
        except ValueError:
            return 0
            
    saved_convs = sorted([n for n in saved_layers if not 'transpose' in n], key=layer_sort_key)
    saved_trans = sorted([n for n in saved_layers if 'transpose' in n], key=layer_sort_key)
    saved_layers = saved_convs + saved_trans
    
    keras_convs = [l for l in target_model.layers if isinstance(l, layers.Conv2D) and not isinstance(l, layers.Conv2DTranspose)]
    keras_trans = [l for l in target_model.layers if isinstance(l, layers.Conv2DTranspose)]
    target_keras_layers = keras_convs + keras_trans
    
    if len(saved_layers) != len(target_keras_layers):
        print(f"WARNING: Saved layer count {len(saved_layers)} != Model layer count {len(target_keras_layers)}")
        
    for i, keras_layer in enumerate(target_keras_layers):
        saved_name = saved_layers[i]
        weights_group = target_group[saved_name]
        
        actual_weights = []
        if saved_name in weights_group:
            inner_group = weights_group[saved_name]
        else:
            inner_group = weights_group
            
        # Try finding kernel and bias manually
        weight_names = [n for n in inner_group.keys()]
        kernel_name = [n for n in weight_names if 'kernel' in n]
        bias_name = [n for n in weight_names if 'bias' in n]
        
        if kernel_name:
            actual_weights.append(np.array(inner_group[kernel_name[0]]))
        if bias_name:
            actual_weights.append(np.array(inner_group[bias_name[0]]))
            
        if actual_weights:
            keras_layer.set_weights(actual_weights)
            
    f.close()
    
    print(f"Saving standalone model to {out_path}...")
    target_model.save_weights(out_path)
    print("✓ Saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['mt', 'uamt', 'cps'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    extract_weights_h5(args.type, args.checkpoint, args.out)
