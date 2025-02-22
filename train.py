import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# Enable GPU memory growth to prevent out-of-memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth")
    except RuntimeError as e:
        print(e)

# Enable mixed precision training to reduce memory usage
set_global_policy('mixed_float16')

# Define dataset path (Update this if needed)
DATASET_PATH = "image_data_2"

# Get class names and ensure correct mapping
class_names = sorted(os.listdir(DATASET_PATH))  # Ensure classes are sorted
NUM_CLASSES = len(class_names)
class_to_index = {name: idx for idx, name in enumerate(class_names)}

# Save class mapping for use in test.py
with open("class_mapping.json", "w") as f:
    json.dump(class_to_index, f)
print("Saved class mapping to class_mapping.json")

# Data preprocessing with ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalize and split data

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(96, 96),
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(96, 96),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)

# Model definition
def get_model():
    base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights="imagenet")

    # Unfreeze last 20 layers for better learning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')  # Ensure correct output shape
    ])
    return model


model = get_model()

# Compile the model before training
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=30)

# Save the trained model
model.save("malayalam_sign_model.h5")
print("Model saved as malayalam_sign_model.h5")
