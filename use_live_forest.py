import pandas as pd
import joblib
import cv2
import numpy as np
import cv2
import numpy as np
import time
import threading
from tkinter import Tk, simpledialog
import pyautogui
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import pygetwindow as gw

########## CAPTURING WINDOW ###########
selected_window = None
screenshot_interval = 0.2  # Interval in seconds
win_rate_history = deque(maxlen=60)  # Store last 60 seconds of win rates

def capture_and_preprocess(selected_window):
    screenshot = pyautogui.screenshot(region=(selected_window.left, selected_window.top, selected_window.width, selected_window.height))
    frame = np.array(screenshot)
    # Display processed image using OpenCV
    cv2.imshow('Processed Minimap', frame)
    cv2.waitKey(1)  # Required for imshow to work properly
    features = extract_features(frame)
    return features

plt.ion()

def update_live_plot():
    global win_rate_history
    plt.ion()  # Turn on interactive mode for live plotting
    fig, ax = plt.subplots()
    green_line, = ax.plot([], [], 'g-', label='Win Rate')  # Green line for win rate
    red_line, = ax.plot([], [], 'r-', label='Complement Win Rate')  # Red line for complement
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Win Rate')
    plt.title('Live Win Rate')
    plt.legend()

    while True:
        win_rates = list(win_rate_history)
        complement_win_rates = [1 - rate for rate in win_rates]  # Complement win rate
        x_data = list(range(len(win_rates)))
        green_line.set_data(x_data, win_rates)
        red_line.set_data(x_data, complement_win_rates)
        ax.relim()
        ax.autoscale_view()
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

def count_players(img, direction):
    rows, cols, _ = img.shape
    print(f"Processing image with dimensions: {rows}x{cols}")
    
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
        if c == "Red":
            target_color = (255, 81, 95) if idx == 0 else (30, 255, 197)
        else:
            target_color = (30, 255, 197) if idx == 0 else (255, 81, 95)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        threshold = 30
        mask = np.all(np.abs(image_rgb - target_color) <= threshold, axis=-1)
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_players[idx] = len(contours)

    if c == "Green":
        print("GREEN", num_players[0])
        print("RED", num_players[1])
    else:
        print("RED", num_players[0])
        print("GREEN", num_players[1])
    return num_players

def count_shapes(img):
    rows, cols, _ = img.shape
    #print("Counting shapes in the image...")

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
    results = []

    c = color(img)

    for idx, image in enumerate(images):
        target_color = (255, 255, 255)
        ult_contours, ult_mask = process_image(image, target_color)
        ability_contours, ability_mask = process_image(image, target_color)

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

        
        team = "GREEN" if c == "Green" else "RED"
        opponent = "RED" if c == "Green" else "GREEN"
        side = "LEFT" if idx == 0 else "RIGHT"

        print("\n===========================================\n")

        if side == "LEFT":
            print(f"TEAM: {c.upper()} - Ults: {num_ults}, Ability Points: {num_ability_points}, Ult Points: {num_ult_points} - SIDE: {direction}")
        else:
            print(f"TEAM: {opponent.upper()} - Ults: {num_ults}, Ability Points: {num_ability_points}, Ult Points: {num_ult_points} - SIDE: {direction}")           

        results.append((num_ults, num_ability_points, num_ult_points))

        print("\n===========================================")

    left_results = results[0]
    right_results = results[1]

    # Green always on the left
    if color(img) == "Red":
        return right_results + left_results 
    else:
        return left_results + right_results

def extract_features(img):
    rows, cols, _ = img.shape
    print(f"Extracting features from image of size: {rows}x{cols}")

    players_alive = count_players(img, "Left")
    left_ults, left_ability_points, left_ult_points, right_ults, right_ability_points, right_ult_points = count_shapes(img)

    features = {
        'green_players alive': players_alive[0],
        'green_ability_count': left_ability_points,
        'green_ult_points': left_ult_points,
        'green_ults': left_ults,
        'red_players_alive': players_alive[1],
        'red_ability_count': right_ability_points,
        'red_ult_points': right_ult_points,
        'red_ults': right_ults
    }

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
    global selected_window, win_rate_history
    
    # Select the window to capture
    select_window()
    
    # Initialize live plot thread
    live_plot_thread = threading.Thread(target=update_live_plot)
    live_plot_thread.start()
    
    # Main loop to capture and predict
    while True:
        if selected_window:
            try:
                start_time = time.time()
                features = capture_and_preprocess(selected_window)
                print(features)
                prediction = predict(features)
                win_rate_history.append(prediction)
                elapsed_time = time.time() - start_time
                time.sleep(max(0, screenshot_interval - elapsed_time))
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    main()
