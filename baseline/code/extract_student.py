"""Extract standalone student model from UA-MT Dice+BCE checkpoint."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

ALPHA = 0.999
T = 8

def get_unet(img_size=256, num_classes=1, dropout_rate=0.1):
    inputs = layers.Input(shape=(img_size, img_size, 1))
    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Dropout(dropout_rate)(c1, training=True)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Dropout(dropout_rate)(c2, training=True)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Dropout(dropout_rate)(c3, training=True)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Dropout(dropout_rate)(c4, training=True)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D()(c4)
    # Bottleneck
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = layers.Dropout(dropout_rate)(c5, training=True)
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
    # Decoder
    u6 = layers.UpSampling2D()(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
    u7 = layers.UpSampling2D()(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
    u8 = layers.UpSampling2D()(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
    u9 = layers.UpSampling2D()(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(c9)
    return models.Model(inputs, outputs)

class UAMTTrainer(tf.keras.Model):
    def __init__(self, student, teacher, T=8, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.T = T
        self.epoch_tracker = tf.Variable(0.0, trainable=False)
        self.alpha = ALPHA

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

# Build and load
student = get_unet(dropout_rate=0.1)
teacher = get_unet(dropout_rate=0.1)
teacher.set_weights(student.get_weights())

model = UAMTTrainer(student, teacher, T=T)
model.compile(optimizer=optimizers.Adam(1e-4),
              metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='io_u')])

# Build the model by passing dummy data
dummy = tf.zeros((1, 256, 256, 1))
_ = model(dummy)

# Load checkpoint weights
model.load_weights("baseline/models/ssl_uamt_50_dicebce/model_best.h5")

# Extract and save student
output_path = "baseline/models/ssl_uamt_50_dicebce/standalone_best.h5"
student.save(output_path)
print(f"Student model extracted and saved to {output_path}")
print(f"Student layers: {len(student.layers)}")
