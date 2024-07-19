import pandas as pd
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model, features = joblib.load('best_rf_model_with_features.pkl')

# Determine the color of the left team
def color(image):
    rows, cols, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_value = image_rgb[int(rows*0.0157), int(cols*0.417), :]
    #print("Color value at (17, 800):", color_value)
    red_color = np.array([105, 44, 49])
    green_color = np.array([40, 133, 114])
    distance_to_red = np.linalg.norm(color_value - red_color)
    distance_to_green = np.linalg.norm(color_value - green_color)
    return "Red" if distance_to_red < distance_to_green else "Green"

# Function to process the image and return contours of the target color
def process_image(image, target_color, threshold=50):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.all(np.abs(image_rgb - target_color) <= threshold, axis=-1).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_closed

# Function to filter points based on their position and size
def filter_points(points, part, type, cols, rows):
    valid_points = []
    points_sorted = sorted(points, key=lambda x: cv2.boundingRect(x)[0], reverse=(part == "Right"))
    y_values = [cv2.boundingRect(point)[1] for point in points_sorted]
    
    if type == "Ability":
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
    else:
        for point in points_sorted:
            x, y = cv2.boundingRect(point)[:2]
            remainder_107 = y % int(rows * 0.099)
            if 0.083 <= remainder_107 / rows <= 0.093:
                if part == "Left" and 0.063 < x / cols < 0.132:
                    valid_points.append(point)
                    continue
                elif part == "Right" and 0.047 < x / cols < 0.109:
                    valid_points.append(point)
                    continue
                if y_values.count(y) == 1:
                    if part == "Left" and 0.063 < x / cols < 0.132:
                        valid_points.append(point)
                        continue
                    elif part == "Right" and 0.047 < x / cols < 0.109:
                        valid_points.append(point)
                        continue
    return valid_points

# Function to count the number of players on each side
def count_players(img):
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

    c = color(img)

    num_players = [0, 0]
    players_health = [0, 0]

    for idx, image in enumerate(images):
        if c == "Red":
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

        # Sort contours by y value
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        # Filter contours based on the remainder condition and minimum area
        filtered_contours = []
        health = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (y % 105) >= 80 and (y % 105) <= 85 and area >= 10:
                filtered_contours.append(contour)
                health.append(w)

        # Count the number of filtered contours (players)
        num_players[idx] = len(filtered_contours)
        if (len(health) < 5):
            health.extend([0] * (5 - len(health)))
        players_health[idx] = health

    return num_players, players_health

# Function to determine number of ability points and ults
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

    num_ability_points = []
    num_ults = []
    num_ult_points = []

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

        num_ult_points.append(len(ult_points))
        num_ability_points.append(len(ability_points))
        num_ults.append(len(ults))
        num_alive_players, players_health = count_players(img)

        player_health_1 = []
        player_health_2 = []
        player_health_3 = []
        player_health_4 = []
        player_health_5 = []

        for health in players_health:
            player_health_1.append(health[0])
            player_health_2.append(health[1])
            player_health_3.append(health[2])
            player_health_4.append(health[3])
            player_health_5.append(health[4])

        print(f"{direction} side:")
        print(f"Number of players alive: {num_alive_players[idx]}")
        print(f"Player 1 health: {player_health_1[idx]}")
        print(f"Player 2 health: {player_health_2[idx]}")
        print(f"Player 3 health: {player_health_3[idx]}")
        print(f"Player 4 health: {player_health_4[idx]}")
        print(f"Player 5 health: {player_health_5[idx]}")
        print(f"Number of ult points: {num_ult_points}")
        print(f"Number of ability points: {num_ability_points}")
        print(f"Number of ults available: {num_ults}")

    return num_alive_players, num_ability_points, num_ult_points, num_ults, player_health_1, player_health_2, player_health_3, player_health_4, player_health_5 

# Function to determine if spike is planted and return the countdown
def count_spike(img):
    rows, cols, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    spike = img [int(rows*0.0102):int(rows*0.0602), int(cols*0.469):int(cols*0.523)]

    # Check if spike has the target color within a range
    lower_range = np.array([245, 73, 98])
    upper_range = np.array([265, 93, 118])


    # Create a mask based on the lower and upper color ranges
    mask = cv2.inRange(spike, lower_range, upper_range)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    cv2.drawContours(spike, contours, -1, (0, 255, 0), 2)

    if len(contours) > 0:
        print("Spike detected!")
        return True
    else:
        print("No spike detected.")
        return False

spike_countdown = 0

# Function to extract features from the image using functions above
def extract_features(image_path):
    global spike_countdown
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    rows, cols, _ = img.shape
    print(f"Extracting features from image of size: {rows}x{cols}")

    num_alive_players, num_ability_points, num_ult_points, num_ults, player_health_1, player_health_2, player_health_3, player_health_4, player_health_5 = count_shapes(img)

    if color(img) == "Green":
        green = 0
        red = 1
    else:
        green = 1
        red = 0

    print(num_ults[green])

    spike_planted = count_spike(img)
    if spike_planted:
        print(f"Spike planted in {image_path}")
        spike_countdown += 0.5
        print("COUNTDWN: ", spike_countdown)
        if (spike_countdown == 45):
            print(f"Spike exploded! in {image_path}")
    else:
        spike_countdown = 0


    features = {
        'green_players_alive' : num_alive_players[green],
        'green_ability_count' : num_ability_points[green],
        'green_health_1' : player_health_1[green],
        'green_health_2' : player_health_2[green],
        'green_health_3' : player_health_3[green],
        'green_health_4' : player_health_4[green],
        'green_health_5' : player_health_5[green],
        'green_ults' : num_ults[green],
        
        'red_players_alive' : num_alive_players[red],
        'red_ability_count' : num_ability_points[red],
        'red_health_1' : player_health_1[red],
        'red_health_2' : player_health_2[red],
        'red_health_3' : player_health_3[red],
        'red_health_4' : player_health_4[red],
        'red_health_5' : player_health_5[red],
        'red_ults' : num_ults[red],

        'spike_countdown' : spike_countdown

    }

    df = pd.DataFrame([features])
    print(features)
    return df

def predict(df):
    # Drop 'image path' column if it exists
    if 'image path' in df.columns:
        df = df.drop(columns=['image path'])
    
    # Predict using the loaded model
    prediction = model.predict(df)
    print(f"Predicted value: {prediction[0]}")
    return prediction[0]

def main(image_path):
    try:
        df = extract_features(image_path)
        predicted_value = predict(df)
        
        img = cv2.imread(image_path)
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
        if predicted_value == 0:
            plt.title("Predicted value: Red Win")
        else:
            plt.title("Predicted value: Green Win")
        plt.show()


        print(f"Predicted value for image: {predicted_value}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Select your image to predict here
    image_path = 'images/004.png' 
    main(image_path)
