import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the screenshot
image = cv2.imread('output/screenshots/1719488223.220506.jpg')

def count_minimap(img):
    # Display the cropped regions for verification
    rows, cols, _ = img.shape

    minimap = img[0:428, 0:597]

    image_rgb = cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB)

    # Perform Canny edge detection on the image
    edges = cv2.Canny(image_rgb, 100, 200)

    # Apply a Gaussian blur to reduce noise
    blurred_edges = cv2.GaussianBlur(edges, (3, 3), 0)

    # Find contours in the blurred edges
    contours, _ = cv2.findContours(blurred_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.zeros_like(blurred_edges)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    # Apply closing on the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the original image
    result = cv2.bitwise_and(minimap, minimap, mask=closed_mask)

    

    # Display the edges
    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    plt.show()

# Call the function to process the image
count_minimap(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
