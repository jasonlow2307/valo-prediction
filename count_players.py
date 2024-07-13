import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the screenshot
#image = cv2.imread('output/screenshots/1718764797.863994.png')
image = cv2.imread('images/002.png')

def count_players(img):
    # Display the cropped regions for verification
    rows, cols, _ = img.shape

    top_left_y_left = int(rows * 0.486)
    bottom_right_y_left = int(rows * 1.019)
    top_left_x_left = int(cols * 0)
    bottom_right_x_left = int(cols * 0.182)

    top_left_y_right = int(rows * 0.486)
    bottom_right_y_right = int(rows * 1.019)
    top_left_x_right = int(cols * 0.818)
    bottom_right_x_right = cols

    left_region = img[top_left_y_left:bottom_right_y_left, top_left_x_left:bottom_right_x_left]
    right_region = img[top_left_y_right:bottom_right_y_right, top_left_x_right:bottom_right_x_right]

    images = [left_region, right_region]

    for idx, image in enumerate(images):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Define the RGB values to search for with a threshold
        if idx == 0:
            target_color = (30, 255, 197)
        else:
            target_color = (255, 81, 95)
        
        threshold = 30  # Adjust this based on your tolerance for color variation

        # Create a mask for pixels with RGB values within the threshold range
        mask = np.all(np.abs(image_rgb - target_color) <= threshold, axis=-1)

        # Perform closing operation on the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Find contours in the closed mask
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of contours (players)
        num_players = len(contours)

        # Visualization
        plt.subplot(2, 2, idx * 2 + 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, idx * 2 + 2)
        plt.imshow(mask_closed, cmap='gray')
        plt.title('Health Bars ({} players)'.format(num_players))  # Calculate number of players
        plt.axis('off')

    plt.show()

count_players(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
