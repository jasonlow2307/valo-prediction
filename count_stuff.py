import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the screenshot
#image = cv2.imread('images/001.png')
image = cv2.imread('output/screenshots/1718764832.027849.png')

# Function to process the image and return contours of the target color
def process_image(image, target_color, threshold=50):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.all(np.abs(image_rgb - target_color) <= threshold, axis=-1).astype(np.uint8) * 255

    # Define the kernel size
    kernel = np.ones((3, 3), np.uint8)

    # Perform opening (erosion followed by dilation) to remove small white regions
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Perform closing (dilation followed by erosion) to grow larger white regions
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_closed

def filter_points(points, part, type):
    if (type == "Ability"):
        if (part == "Left"):
            points_sorted = sorted(points, key=lambda x: cv2.boundingRect(x)[0])  # Sort by x-coordinate
        else:
            points_sorted = sorted(points, key=lambda x: cv2.boundingRect(x)[0], reverse=True)
        valid_points = []
        y_values = [cv2.boundingRect(point)[1] for point in points_sorted]  # Extract all y values

        for point in points_sorted:
            x, y = cv2.boundingRect(point)[:2]
            remainder_107 = y % 107

            # Check if the remainder is around 64 (valid points)
            if 60 <= remainder_107 <= 70:
                if (part=="Left") and (x > 90):
                    # Ensure no other valid point has a similar y value but x difference exceeds 140
                    if all(abs(x - cv2.boundingRect(vp)[0]) <= 140 for vp in valid_points if abs(y - cv2.boundingRect(vp)[1]) <= 2):
                        valid_points.append(point)
                    # Check if it has a unique y value
                    elif y_values.count(y) == 1:
                        if all(abs(x - cv2.boundingRect(vp)[0]) <= 140 for vp in valid_points if abs(y - cv2.boundingRect(vp)[1]) <= 2):
                            valid_points.append(point)
                elif (part=="Right") and (x < 260):
                    # Ensure no other valid point has a similar y value but x difference exceeds 140
                    if all(abs(x - cv2.boundingRect(vp)[0]) <= 140 for vp in valid_points if abs(y - cv2.boundingRect(vp)[1]) <= 2):
                        valid_points.append(point)
                    # Check if it has a unique y value
                    elif y_values.count(y) == 1:
                        if all(abs(x - cv2.boundingRect(vp)[0]) <= 140 for vp in valid_points if abs(y - cv2.boundingRect(vp)[1]) <= 2):
                            valid_points.append(point)
            # Check if the remainder is around 92 (noise points)
            elif 90 <= remainder_107 <= 94:
                continue  # Skip noise points
    else:
        if (part == "Left"):
            points_sorted = sorted(points, key=lambda x: cv2.boundingRect(x)[0])  # Sort by x-coordinate
        else:
            points_sorted = sorted(points, key=lambda x: cv2.boundingRect(x)[0], reverse=True)
        valid_points = []
        y_values = [cv2.boundingRect(point)[1] for point in points_sorted]  # Extract all y values

        for point in points_sorted:
            x, y = cv2.boundingRect(point)[:2]
            remainder_107 = y % 107

            # Check if the remainder is around 64 (valid points)
            if 90 <= remainder_107 <= 100:
                if (part=="Left"):
                    if x < 255 and x > 120:
                        valid_points.append(point)
                        print("LEFT", x, y)
                        continue
                else:
                    if x > 90 and x < 210:
                        valid_points.append(point)
                        print("RIGHT", x, y)
                        continue
                # Check if it has a unique y value
                if y_values.count(y) == 1:
                    if (part=="Left"):
                        if x < 255 and x > 120:
                            valid_points.append(point)
                            print("LEFT", x, y)
                            continue
                    else:
                        if x > 90 and x < 210:
                            valid_points.append(point)
                            print("RIGHT", x, y)
                            continue
    return valid_points


def count_shapes(img):
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
        target_color = (255, 255, 255)  # Adjust if necessary
        
        ult_contours, ult_mask = process_image(image, target_color)
        ability_contours, ability_mask = process_image(image, target_color)

        # Define area ranges for ult points and ability points
        min_area_ult_points = 3
        max_area_ult_points = 15
        min_area_ult = 1000
        max_area_ult = 1200
        min_area_ability = 20
        max_area_ability = 55

        ult_points = [contour for contour in ult_contours if min_area_ult_points <= cv2.contourArea(contour) <= max_area_ult_points]
        ability_points = [contour for contour in ability_contours if min_area_ability <= cv2.contourArea(contour) <= max_area_ability]
        ults = [contour for contour in ability_contours if min_area_ult <= cv2.contourArea(contour) <= max_area_ult]

        # Filter points based on y-distance
        if image is left_region:
            direction = "Left"
        else:
            direction = "Right"

        ult_points = filter_points(ult_points, direction, "Ult")
        ability_points = filter_points(ability_points, direction, "Ability")

        num_ult_points = len(ult_points)
        num_ability_points = len(ability_points)
        num_ults = len(ults)

        # Draw contours on the image for visualization
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, ult_points, -1, (0, 255, 0), 2)  # Green for ult points
        cv2.drawContours(image_with_contours, ability_points, -1, (0, 0, 255), 2)  # Red for ability points
        cv2.drawContours(image_with_contours, ults, -1, (255, 0, 0), 2)  # Red for ability points

        # Visualization
        plt.subplot(2, 2, idx * 2 + 1)
        plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
        plt.title('Original Image with Contours')
        plt.axis('off')

        plt.subplot(2, 2, idx * 2 + 2)
        combined_mask = ult_mask | ability_mask
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Ult Points: {} | Ability Points: {} | Ults {}'.format(num_ult_points, num_ability_points, num_ults))
        plt.axis('off')

    plt.show()

count_shapes(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
