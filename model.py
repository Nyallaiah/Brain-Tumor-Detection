import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Image dimensions (must match the training setup)
IMG_HEIGHT = 150
IMG_WIDTH = 150


def predict_image(model_path, image_path):
    """
    Load a trained model and predict whether the given image is Healthy or Tumor.

    :param model_path: Path to the trained model (.h5 file)
    :param image_path: Path to the image for classification
    :return: Classification result (Healthy or Tumor)
    """
    # Load trained model
    model = load_model(model_path)

    # Load and preprocess image
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "Healthy"

    # Display the image with prediction
    plt.imshow(image)
    plt.title(f"Prediction: {label} (Confidence: {prediction:.2f})")
    plt.axis("off")
    plt.show()

    return label


if __name__ == "__main__":
    # Example usage
    model_path = "CNN.h5"  # Change this to your trained model file
    image_path = "dataset/test/Tumor/ct_tumor (4).jpg"  # Change this to an actual test image path

    result = predict_image(model_path, image_path)
    print(f"Model Prediction: {result}")
