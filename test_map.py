import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd

# Function to preprocess images
def preprocess_image(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize mask to fit image size
    minimap = cv2.bitwise_and(image, image, mask=mask_resized)
    minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    minimap_resized = cv2.resize(minimap_gray, (64, 64))  # Resize to 64x64
    minimap_normalized = minimap_resized / 255.0  # Normalize pixel values
    return minimap_normalized

# Load the labels.csv file
labels_df = pd.read_csv('labels.csv')

# Convert filenames to full paths if necessary
image_dir = ''  # Directory where images are stored
labels_df['filename'] = image_dir + labels_df['filename']

# Extract filenames and winrates
image_paths = labels_df['filename'].values
winrates = labels_df['win'].values  # Or labels_df['win'].values for binary labels

mask_path = 'images/mask.png'  # Path to the mask image

# Preprocess images
images = np.array([preprocess_image(image_path, mask_path) for image_path in image_paths])
images = np.expand_dims(images, axis=-1)  # Add channel dimension

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(images, winrates, test_size=0.2, random_state=42)

# Ensure labels are in the correct shape
y_train = np.array(y_train)
y_val = np.array(y_val)

# Build CNN model with increased convolutional layers
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Additional convolutional layer
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),  # Additional convolutional layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary crossentropy for binary classification
              metrics=['accuracy'])  # Optional: Add metrics like accuracy for evaluation

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Example of predicting winrate for a new minimap
new_minimap = preprocess_image('images/004.png', mask_path)
new_minimap = np.expand_dims(new_minimap, axis=0)  # Add batch dimension
new_minimap = np.expand_dims(new_minimap, axis=-1)  # Add channel dimension (assuming grayscale input)
predicted_winrate = model.predict(new_minimap)
print(f'Predicted Winrate: {predicted_winrate[0][0]}')