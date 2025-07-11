import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Load trained model
MODEL_PATH = "CNN.h5"  # Change to "ANN.h5" or "VGG16.h5" if needed
model = load_model(MODEL_PATH)

# Function to predict a single image
def predict_image(image_path):
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "Healthy"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Test images directory
test_images_dir = "dataset/test/CT/Tumor"  # Folder containing test images

test_results = {}
for image_name in os.listdir(test_images_dir):
    image_path = os.path.join(test_images_dir, image_name)
    label, confidence = predict_image(image_path)
    test_results[image_name] = (label, confidence)
    print(f"{image_name}: {label} (Confidence: {confidence:.2f})")
