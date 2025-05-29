import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from collections import Counter
import random # Import random for selecting one image

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    """
    Load image data, build and train a neural network, and evaluate performance.
    """
    # Check command-line arguments
    if len(sys.argv) != 2: # Modified to expect only one argument (data_directory)
        sys.exit("Usage: python traffic.py data_directory")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels_np = np.array(labels)
    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        np.array(images), labels_np, test_size=TEST_SIZE, random_state=42
    )

    # Identify top 5 most common sign types from training data
    top_5_categories = get_top_n_categories(y_train_raw, 5)
    print(f"Top 5 most common sign categories in training data: {top_5_categories}")

    # Convert and save a single random image for each of the top 5 categories
    save_one_random_image_per_category(x_train, y_train_raw, top_5_categories)

    # Convert labels to categorical for model training
    y_train = tf.keras.utils.to_categorical(y_train_raw)
    y_test = tf.keras.utils.to_categorical(y_test_raw)

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Removed the model saving part


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    print(f"Loading data from {data_dir}...")
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            continue
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(resized_img)
                    labels.append(category)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    print("Data loading complete.")
    return images, labels


def get_top_n_categories(labels, n):
    """
    Identifies the top N most common categories from a list of labels.
    Returns a list of category IDs.
    """
    label_counts = Counter(labels)
    top_n = [label for label, count in label_counts.most_common(n)]
    return top_n


def save_one_random_image_per_category(images, labels, top_categories):
    """
    Selects one random image from each of the specified top categories
    and saves them as .png files.
    """
    output_dir = "random_top_5_sign_samples_png"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving one random image per top {len(top_categories)} categories to '{output_dir}'...")

    # Group images by category
    category_images = {category: [] for category in top_categories}
    for i, category in enumerate(labels):
        if category in top_categories:
            category_images[category].append(images[i])

    # Select and save one random image for each category
    for category_id in top_categories:
        if category_images[category_id]: # Ensure there are images for this category
            random_image = random.choice(category_images[category_id])
            filename = f"random_sample_category_{category_id}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, random_image)
            print(f"Saved: {filepath}")
        else:
            print(f"No images found for category {category_id} in the training set.")
    print("Random image saving complete.")


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()