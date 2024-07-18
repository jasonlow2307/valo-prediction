from datetime import datetime
from io import BytesIO
import pandas as pd
import joblib
import cv2
import numpy as np
import cv2
import numpy as np
import time
from tkinter import Tk, simpledialog
from PIL import Image
import pyautogui
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import pygetwindow as gw
import threading
from queue import Queue

########## CAPTURING WINDOW ###########
selected_window = None
screenshot_interval = 0.2  # Interval in seconds
win_rate_history = deque(maxlen=60)  # Store last 60 seconds of win rates
extracted_features_history = []  # Global list to store extracted features

green_player_alive = []
red_player_alive = []

green_ability_count = []
red_ability_count = []

def capture_and_preprocess(selected_window):
    screenshot = pyautogui.screenshot(region=(selected_window.left, selected_window.top, selected_window.width, selected_window.height))
    
    # Convert the screenshot to an in-memory file-like object
    buffer = BytesIO()
    screenshot.save(buffer, format="PNG")
    buffer.seek(0)

    # Load the image from the in-memory file-like object
    pil_image = Image.open(buffer)
    frame = np.array(pil_image)

    # Convert RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Process the frame
    features = extract_features(frame)

    # Add a delay to avoid rapid looping (adjust as needed)
    time.sleep(0.5)

    return features

def update_live_plot():
    global win_rate_history

    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10, 4))  # Create a figure for the win rate plot

    # Win Rate Plot
    green_line, = ax1.plot([], [], 'g-', marker='o', markersize=5, label='Win Rate')  # Green line with markers
    ax1.set_xlim(0, 60)
    ax1.set_ylim(-0.1, 1.1)  # Adjusted to show binary values
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Live Win Rate')
    ax1.legend()

    while True:
        win_rates = list(win_rate_history)
        x_data = list(range(len(win_rates)))

        green_line.set_data(x_data, win_rates)
        if len(green_player_alive) > 0 and len(win_rate_history) > 0 and len(green_ability_count) > 0 and len(red_ability_count) > 0:
            ax1.set_title(f'Green {green_player_alive[-1]} Red {red_player_alive[-1]} Win Rate: {win_rate_history[-1]} G Ability: {green_ability_count[-1]} R Ability: {red_ability_count[-1]}')
        ax1.relim()
        ax1.autoscale_view()

        plt.draw()
        plt.pause(0.2)  # Update plot every 0.2 seconds


def select_window():
    global selected_window
    windows = gw.getWindowsWithTitle('')
    window_titles = [w.title for w in windows if w.title]
    if not window_titles:
        print("No windows found.")
        return None

    root = Tk()
    root.withdraw()
    selected_window_title = simpledialog.askstring("Input", f"Select a window from the following list:\n\n{window_titles}")
    if selected_window_title:
        selected_window = next((w for w in windows if w.title == selected_window_title), None)
        if selected_window:
            print(f"Selected window: {selected_window.title}")
        else:
            print("Window not found.")
    root.destroy()


# Load the trained model
model = joblib.load('best_rf_model.pkl')

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

def process_image(image, target_color, threshold=50):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.all(np.abs(image_rgb - target_color) <= threshold, axis=-1).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_closed

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

def count_players(img):
    rows, cols, _ = img.shape

    # Define the regions of interest
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
    players_health = [[], []]

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

        threshold = 50  # Adjust this based on your tolerance for color variation

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
            if area >= 10:
                filtered_contours.append(contour)
                health.append(w)

        # Debugging print statements
        print(f"Image region {idx}:")
        print(f"Target color: {target_color}")
        print(f"Number of contours found: {len(contours)}")
        print(f"Location of contours {[(cv2.boundingRect(c)[1]) for c in contours]}")
        print(f"Number of filtered contours: {len(filtered_contours)}")
        print(f"Health values: {health}")

        # Count the number of filtered contours (players)
        num_players[idx] = len(filtered_contours)
        if len(health) < 5:
            health.extend([0] * (5 - len(health)))
        players_health[idx] = health

    return num_players, players_health

spike_countdown = 0

def count_shapes(img):
    global spike_countdown
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
    
    spike_planted = count_spike(img)
    if spike_planted:
        print("Spike planted")
        spike_countdown += 0.5
        print("COUNTDWN: ", spike_countdown)
        if (spike_countdown == 45):
            print("Spike exploded! in")
    else:
        spike_countdown = 0


    return num_alive_players, num_ability_points, num_ult_points, num_ults, player_health_1, player_health_2, player_health_3, player_health_4, player_health_5, spike_countdown 

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



def extract_features(img):
    rows, cols, _ = img.shape
    print(f"Extracting features from image of size: {rows}x{cols}")

    num_alive_players, num_ability_points, num_ult_points, num_ults, player_health_1, player_health_2, player_health_3, player_health_4, player_health_5, spike_countdown = count_shapes(img)

    if color(img) == "Green":
        green = 0
        red = 1
    else:
        green = 1
        red = 0

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

        'spike_countdown': spike_countdown

    }
    extracted_features_history.append(features)
    green_player_alive.append(num_alive_players[green])
    red_player_alive.append(num_alive_players[red])

    green_ability_count.append(num_ability_points[green])
    red_ability_count.append(num_ability_points[red])

    df = pd.DataFrame([features])
    return df

def predict(df):
    # Drop 'image path' column if it exists
    if 'image path' in df.columns:
        df = df.drop(columns=['image path'])
    
    # Predict using the loaded model
    prediction = model.predict(df)
    print(f"Prediction: {prediction[0]}")
    return prediction[0]

def main():
    global selected_window, win_rate_history, win_rate_queue, extracted_features_history

    # Select the window to capture
    select_window()

    # Start live plot update in a separate thread
    update_live_plot_thread = threading.Thread(target=update_live_plot)
    update_live_plot_thread.start()

    # Main loop to capture and predict
    while True:
        if selected_window:
            try:
                start_time = time.time()
                features = capture_and_preprocess(selected_window)
                prediction = predict(features)
                
                # Add win rate to the win rate history for plotting
                win_rate_history.append(prediction)
                
                # Log prediction for debugging or further analysis
                print(f"Prediction: {prediction}")
                
                elapsed_time = time.time() - start_time
                time.sleep(max(0, screenshot_interval - elapsed_time))
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    main()