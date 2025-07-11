import os
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def count_images(directory):
    categories = {}
    for root, _, files in os.walk(directory):
        label = os.path.basename(root)
        if label not in categories:
            categories[label] = 0
        categories[label] += len(files)
    return categories


def plot_data_distribution(train_dir, test_dir):
    train_counts = count_images(train_dir)
    test_counts = count_images(test_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(train_counts.keys(), train_counts.values(), color='blue')
    axes[0].set_title("Training Data Distribution")
    axes[1].bar(test_counts.keys(), test_counts.values(), color='red')
    axes[1].set_title("Testing Data Distribution")
    plt.show()


if __name__ == "__main__":
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    plot_data_distribution(train_dir, test_dir)
