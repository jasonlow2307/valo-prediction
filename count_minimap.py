import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the screenshot
image = cv2.imread('output/screenshots/1719488223.220506.jpg')

def count_minimap(img):
    # Display the cropped regions for verification
    rows, cols, _ = img.shape

    minimap = img[0:int(rows*0.396), 0:int(cols*0.311)]

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

    minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

    feature_vector = minimap_gray.flatten()

    print(feature_vector)

    '''
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Define the target color and threshold range
    target_color = np.array([233, 194, 193]) #([63, 107, 74], 
    threshold = 15
    lower_bound = target_color - threshold
    upper_bound = target_color + threshold

    # Create a mask for the target color range
    mask_color = cv2.inRange(result, lower_bound, upper_bound)

    # Find contours in the color mask
    contours_color, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the result image
    cv2.drawContours(result, contours_color, -1, (0, 255, 0), 2)
    '''

    # Display the result image with contours
    plt.figure()
    plt.imshow(result)
    plt.title('Contours for Color (63, 107, 74)')
    plt.axis('off')
    plt.show()

# Call the function to process the image
count_minimap(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
