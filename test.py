
from keras.models import load_model
import cv2
import numpy as np
import os

# Path to the test images
IMG_TEST_PATH = "test_data"

# List of Malayalam sign labels (same as in train.py)
label_lst = ['A', 'Aa', 'Ah', 'Ai', 'Am', 'Au', 'Ba', 'Bha', 'Ca', 'Cha', 'D_a', 'D_ha', 'Da', 'Dha', 'E', 'E_', 'Ee',
             'Ee_', 'Ga', 'Gha', 'Ha', 'I', 'Ii', 'Ilh', 'Ill', 'In', 'Irr', 'Ja', 'Ka', 'Kha', 'La', 'Lha', 'Ma',
             'N_a', 'Na', 'Nga', 'Nha', 'Nothing', 'O', 'Oo', 'Pa', 'Pha', 'R', 'Ra', 'Rha', 'Sa', 'Sha', 'Shha',
             'Space', 'T_a', 'T_ha', 'Ta', 'Tha', 'U', 'U_', 'Uu', 'Uu_', 'Va', 'Ya', 'Zha']
NUM_CLASSES = len(label_lst)
REV_CLASS_MAP = {i: label_lst[i] for i in range(NUM_CLASSES)}

def mapper(val):
    return REV_CLASS_MAP[val]

# Load the trained model
model = load_model("malayalam_sign_model.h5")

# Ensure the test folder exists
if not os.path.exists(IMG_TEST_PATH):
    print(f"Error: Test folder '{IMG_TEST_PATH}' not found!")
    exit()

print("\n=== Testing Started ===\n")

for directory in os.listdir(IMG_TEST_PATH):
    path = os.path.join(IMG_TEST_PATH, directory)
    if not os.path.isdir(path):
        continue

    for file_name in os.listdir(path):
        # Skip hidden files and non-image files
        if file_name.startswith(".") or not (file_name.endswith(".jpg") or file_name.endswith(".png")):
            continue

        image_path = os.path.join(path, file_name)

        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Skipping {file_name}: Unable to read image")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96, 96))  # Match training size
            img = img / 255.0  # Normalize pixel values
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Predict sign
            pred = model.predict(img)
            sign_code = np.argmax(pred[0])
            sign_name = mapper(sign_code)

            print(f" Predicted: {sign_name} for image {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

print("\n=== Testing Completed ===")
