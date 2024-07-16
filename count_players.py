import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the screenshot
#image = cv2.imread('output/screenshots/1718764797.863994.png')
#image = cv2.imread('images/002.png')
image = cv2.imread('output/screenshots/1719488223.220506.jpg')

def color(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract the color value at (17, 800)
    color_value = image_rgb[17, 800, :]

    # Print the color value
    print("Color value at (17, 800):", color_value)

    # Define target colors
    red_color = np.array([105, 44, 49])
    green_color = np.array([40, 133, 114])

    # Calculate the Euclidean distances to the target colors
    distance_to_red = np.linalg.norm(color_value - red_color)
    distance_to_green = np.linalg.norm(color_value - green_color)

    # Create a small 1x1 image with the extracted color value for visualization
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[:, :] = color_value

    '''
    # Display the color
    plt.imshow(color_image)
    plt.title("Color at (17, 800)")
    plt.axis('off')
    plt.show()
    '''

    # Determine the closest color
    if distance_to_red < distance_to_green:
        print("The color is closer to red.")
        return "Red"
    else:
        print("The color is closer to green.")
        return "Green"

def count_players(img):
    c = color(img)
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
      
        if (c == "Red"):
            # Define the RGB values to search for with a threshold
            if idx == 0:
                target_color = (255, 81, 95)
            else:
                target_color = (30, 255, 197)
        else:
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

        # Sort contours by y value
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        # Filter contours based on the remainder condition and minimum area
        filtered_contours = []
        players_health = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (y % 105) >= 80 and (y % 105) <= 85 and area >= 10:
                players_health.append(w)
                filtered_contours.append(contour)

        # Count the number of filtered contours (players)
        num_players = len(filtered_contours)

        # Visualization
        plt.subplot(2, 2, idx * 2 + 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, idx * 2 + 2)
        plt.imshow(mask_closed, cmap='gray')
        plt.title(f'Health Bars ({num_players} players) - {players_health}')
        plt.axis('off')

    plt.show()

count_players(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
