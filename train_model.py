import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

def preprocess_image(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize mask to fit image size
    minimap = cv2.bitwise_and(image, image, mask=mask_resized)
    rows, cols, _ = minimap.shape
    minimap = minimap[0:int(rows//2.3), 0:int(cols//4)]
    minimap_resized = cv2.resize(minimap, (64, 64))  # Resize to 64x64
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

# Find the starting index of '1718788568.5275614.jpg'
start_index = next(i for i, image_path in enumerate(image_paths) if '1718788568.5275614.jpg' in image_path.split('/')[-1])
print("PACIFIC INDEX:", start_index)

mask_path = 'images/mask.png'  # Path to the mask image
pacific_mask_path = 'images/mask_pacific.jpg'  # Path to the Pacific mask image

# Select all images from the starting index onwards
normal_images = image_paths[:start_index]
pacific_images = image_paths[start_index:]

# Preprocess images
normal_preprocessed = [preprocess_image(image_path, mask_path) for image_path in normal_images]
pacific_preprocessed = [preprocess_image(image_path, pacific_mask_path) for image_path in pacific_images]

# Combine preprocessed images
combined_images = np.array(normal_preprocessed + pacific_preprocessed)

# Ensure combined_images has the correct shape for model training
if combined_images.shape[-1] == 1:
    combined_images = np.repeat(combined_images, 3, axis=-1)  # Repeat grayscale channel to make it 3 channels

# Combine winrates correspondingly
combined_winrates = np.array(list(winrates[:start_index]) + list(winrates[start_index:]))

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(combined_images, combined_winrates, test_size=0.2, random_state=42)

# Ensure labels are in the correct shape
y_train = np.array(y_train)
y_val = np.array(y_val)

model = Sequential([
    tf.keras.Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(512, (3, 3), activation='relu', padding='same'),  # Adjusted padding to 'same'
    MaxPooling2D((2, 2)),  # Ensure the output dimensions are suitable for the next layers
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary crossentropy for binary classification
              metrics=['accuracy'])  # Optional: Add metrics like accuracy for evaluation

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Save the model to HDF5 file
model.save('model.h5')

# Predict class labels
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)

# Print classification report
print(classification_report(y_val, y_pred))

# Print confusion matrix
print(confusion_matrix(y_val, y_pred))

# Calculate AUC-ROC
roc_auc = roc_auc_score(y_val, y_pred_prob)
print(f'ROC-AUC: {roc_auc}')

# Example of predicting winrate for a new minimap
new_minimap = preprocess_image('images/004.png', mask_path)
new_minimap = np.expand_dims(new_minimap, axis=0)  # Add batch dimension
new_minimap = np.repeat(new_minimap, 3, axis=-1)  # Repeat grayscale channel to make it 3 channels if necessary
predicted_winrate = model.predict(new_minimap)
print(f'Predicted Winrate: {predicted_winrate[0][0]}')
