import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the screenshot
#image = cv2.imread('output/screenshots/1718764832.027849.png')
#image = cv2.imread('images/001.png')
image = cv2.imread('output/screenshots/1719488208.457024.jpg')

def color(image):
    rows, cols, _ = image.shape

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract the color value at (17, 800)
    color_value = image_rgb[int(rows*0.0157), int(cols*0.417), :]

    # Print the color value
    print("Color value at (17, 800):", color_value)

    # Define target colors
    red_color = np.array([105, 44, 49])
    green_color = np.array([40, 133, 114])

    # Calculate the Euclidean distances to the target colors
    distance_to_red = np.linalg.norm(color_value - red_color)
    distance_to_green = np.linalg.norm(color_value - green_color)

    # Determine the closest color
    if distance_to_red < distance_to_green:
        print("The color is closer to red.")
        return "Red"
    else:
        print("The color is closer to green.")
        return "Green"
    


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

def filter_points(points, part, type, cols, rows):
    if type == "Ability":
        points_sorted = sorted(points, key=lambda x: cv2.boundingRect(x)[0], reverse=(part == "Right"))
        valid_points = []
        y_values = [cv2.boundingRect(point)[1] for point in points_sorted]

        for point in points_sorted:
            x, y = cv2.boundingRect(point)[:2]
            remainder_107 = y % int(rows * 0.099)

            if 0.056 <= remainder_107 / rows <= 0.065:
                if part == "Left" and x / cols < 0.17:
                    if all(abs(x - cv2.boundingRect(vp)[0]) <= int(cols * 0.073) for vp in valid_points if abs(y - cv2.boundingRect(vp)[1]) <= int(rows * 0.002)):
                        valid_points.append(point)
                    elif y_values.count(y) == 1 and all(abs(x - cv2.boundingRect(vp)[0]) <= int(cols * 0.073) for vp in valid_points if abs(y - cv2.boundingRect(vp)[1]) <= int(rows * 0.002)):
                        valid_points.append(point)
                elif part == "Right" and x / cols > 0.059:
                    if all(abs(x - cv2.boundingRect(vp)[0]) <= int(cols * 0.073) for vp in valid_points if abs(y - cv2.boundingRect(vp)[1]) <= int(rows * 0.002)):
                        valid_points.append(point)
                    elif y_values.count(y) == 1 and all(abs(x - cv2.boundingRect(vp)[0]) <= int(cols * 0.073) for vp in valid_points if abs(y - cv2.boundingRect(vp)[1]) <= int(rows * 0.002)):
                        valid_points.append(point)
            elif 0.083 <= remainder_107 / rows <= 0.087:
                continue
    else:
        points_sorted = sorted(points, key=lambda x: cv2.boundingRect(x)[0], reverse=(part == "Right"))
        valid_points = []
        y_values = [cv2.boundingRect(point)[1] for point in points_sorted]

        for point in points_sorted:
            x, y = cv2.boundingRect(point)[:2]
            remainder_107 = y % int(rows * 0.099)

            if 0.083 <= remainder_107 / rows <= 0.093:
                if part == "Left" and 0.063 < x / cols < 0.132:
                    valid_points.append(point)
                    print("LEFT", x, y)
                    continue
                elif part == "Right" and 0.047 < x / cols < 0.109:
                    valid_points.append(point)
                    print("RIGHT", x, y)
                    continue
                if y_values.count(y) == 1:
                    if part == "Left" and 0.063 < x / cols < 0.132:
                        valid_points.append(point)
                        print("LEFT", x, y)
                        continue
                    elif part == "Right" and 0.047 < x / cols < 0.109:
                        valid_points.append(point)
                        print("RIGHT", x, y)
                        continue
    return valid_points

def count_players(img, direction):
    rows, cols, _ = img.shape
    print("ROWS", rows)
    print("COLS", cols)

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

    c = color(img)
    

    num_players = [0, 0]

    for idx, image in enumerate(images):
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

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        threshold = 30  # Adjust this based on your tolerance for color variation

        # Create a mask for pixels with RGB values within the threshold range
        mask = np.all(np.abs(image_rgb - target_color) <= threshold, axis=-1)

        # Perform closing operation on the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Find contours in the closed mask
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of contours (players)
        num_players[idx] = len(contours)

    return num_players

def count_shapes(img):
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
        min_area_ult_points = int(cols * rows * 0.0000014)
        max_area_ult_points = int(cols * rows * 0.0000072)
        min_area_ult = int(cols * rows * 0.00048)
        max_area_ult = int(cols * rows * 0.00058)
        min_area_ability = int(cols * rows * 0.00000965)
        max_area_ability = int(cols * rows * 0.0000265)

        ult_points = [contour for contour in ult_contours if min_area_ult_points <= cv2.contourArea(contour) <= max_area_ult_points]
        ability_points = [contour for contour in ability_contours if min_area_ability <= cv2.contourArea(contour) <= max_area_ability]
        ults = [contour for contour in ability_contours if min_area_ult <= cv2.contourArea(contour) <= max_area_ult]

        direction = "Left" if image is left_region else "Right"

        ult_points = filter_points(ult_points, direction, "Ult", cols, rows)
        ability_points = filter_points(ability_points, direction, "Ability", cols, rows)

        num_ult_points = len(ult_points)
        num_ability_points = len(ability_points)
        num_ults = len(ults)
        num_alive_players = count_players(img, direction)

        print(f"{direction} side:")
        print(f"Number of players alive: {num_alive_players[idx]}")
        print(f"Number of ult points: {num_ult_points}")
        print(f"Number of ability points: {num_ability_points}")
        print(f"Number of ults available: {num_ults}")

        # Draw contours on the image for visualization
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, ult_points, -1, (0, 255, 0), 2)
        cv2.drawContours(image_with_contours, ability_points, -1, (255, 0, 0), 2)
        cv2.drawContours(image_with_contours, ults, -1, (0, 0, 255), 2)

        plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
        plt.title(f'{direction} side with contours')
        plt.show()

count_shapes(image)
cv2.waitKey()
cv2.destroyAllWindows()
