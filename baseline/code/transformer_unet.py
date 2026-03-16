import tensorflow as tf
from tensorflow.keras import layers, models

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_transformer_unet(input_shape=(256, 256, 1), patch_size=16, projection_dim=256, num_heads=8, transformer_layers=8, num_classes=1):
    """
    Vision Transformer U-Net (TransUNet style)
    Serves as the complex Transformer Baseline (Keras equivalent to Swin-UNet/TransUNet)
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- 1. ViT Encoder ---
    # Extract patches directly from the image
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    
    # Calculate grid size and num patches
    grid_size = input_shape[0] // patch_size
    num_patches = grid_size * grid_size
    
    # Reshape patches for the transformer
    _, h, w, c = patches.shape
    patches_reshaped = layers.Reshape((num_patches, c))(patches)
    
    # Linear Projection
    encoded_patches = layers.Dense(projection_dim)(patches_reshaped)
    
    # Position Embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches = encoded_patches + position_embedding

    # Transformer Blocks
    x = encoded_patches
    for _ in range(transformer_layers):
        # Layer Normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip Connection 1
        x2 = layers.Add()([attention_output, x])
        # Layer Normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)
        # Skip Connection 2
        x = layers.Add()([x3, x2])
        
    # --- 2. CNN Decoder (U-Net Style) ---
    # Reshape the 1D sequence back into a 2D spatial grid (16x16x256)
    encoded_image = layers.Reshape((grid_size, grid_size, projection_dim))(x)
    
    # Up-block 1: 16x16 -> 32x32
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(encoded_image)
    c1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c1 = layers.BatchNormalization()(c1)
    
    # Up-block 2: 32x32 -> 64x64
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c2 = layers.BatchNormalization()(c2)
    
    # Up-block 3: 64x64 -> 128x128
    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u3)
    c3 = layers.BatchNormalization()(c3)
    
    # Up-block 4: 128x128 -> 256x256
    u4 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c3)
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.BatchNormalization()(c4)
    
    # Output Head
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c4)
    
    return models.Model(inputs=inputs, outputs=outputs, name="Transformer_Baseline")
