import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
IMAGE_SIZE = (64, 64)  # Resize images to this size
DATA_PATH = 'data/'    # Path to the image data folder

def load_images(data_path, image_size):
    """
    Load images from a given path and resize them to a specified size.

    :param data_path: Path to the directory containing image subfolders.
    :param image_size: Tuple specifying the size to resize images to (width, height).
    :return: A tuple containing the images as a numpy array and the labels as a list.
    """
    images = []
    labels = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                img = imread(img_path)
                img_resized = resize(img, image_size)
                images.append(img_resized)
                labels.append(folder)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(images), labels

# Load images and labels
X, y = load_images(DATA_PATH, IMAGE_SIZE)

# Flatten images for Random Forest input
X_flat = X.reshape(len(X), -1)

# Encode labels to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
