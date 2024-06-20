import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize mask to fit image size
    minimap = cv2.bitwise_and(image, image, mask=mask_resized)
    rows, cols, _ = minimap.shape
    minimap = minimap[0:int(rows//2.3), 0:int(cols//4)]
    minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    minimap_resized = cv2.resize(minimap_gray, (64, 64))  # Resize to 64x64
    minimap_normalized = minimap_resized / 255.0  # Normalize pixel values
    return minimap_normalized

def predict_winrate(image_path, model):
    mask_path = 'images/mask_pacific.jpg'  # Path to the mask image used during preprocessing
    processed_image = preprocess_image(image_path, mask_path)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=-1)  # Add channel dimension (assuming grayscale input)
    prediction = model.predict(processed_image)
    return prediction[0, 0]  # Assuming binary classification; adjust if needed

def main():
    # Load the trained model
    model = tf.keras.models.load_model('model.h5')

    # Example of predicting winrate for a new minimap image
    image_path = 'output/screenshots/1718766691.0107048.png'
    predicted_winrate = predict_winrate(image_path, model)
    print(f'Predicted Winrate: {predicted_winrate}')
    cv2.imshow('Minimap', cv2.imread(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
