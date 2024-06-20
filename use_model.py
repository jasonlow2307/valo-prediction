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

model = tf.keras.models.load_model('model.h5')

selected_window = None
screenshot_interval = 0.25  # Interval in seconds
win_rate_history = deque(maxlen=60)  # Store last 60 seconds of win rates

def capture_and_preprocess(selected_window):
    screenshot = pyautogui.screenshot(region=(selected_window.left, selected_window.top, selected_window.width, selected_window.height))
    frame = np.array(screenshot)
    mask = cv2.imread('images/mask_pacific.jpg', 0)
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize mask to fit image size
    minimap = cv2.bitwise_and(frame, frame, mask=mask_resized)
    rows, cols, _ = minimap.shape
    minimap = minimap[0:int(rows//2.3), 0:int(cols//4)]
    #minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    minimap_resized = cv2.resize(minimap, (64, 64))  # Resize to 64x64
    minimap_normalized = minimap_resized / 255.0  # Normalize pixel values
    
    # Display processed image using OpenCV
    cv2.imshow('Processed Minimap', minimap)
    cv2.waitKey(1)  # Required for imshow to work properly
    
    return minimap_normalized

def predict_win(image, model):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction[0, 0]  # Assuming binary classification; adjust if needed

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
        plt.pause(1)  # Update plot every second

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
                image = capture_and_preprocess(selected_window)
                prediction = predict_win(image, model)
                win_rate_history.append(prediction)
                elapsed_time = time.time() - start_time
                time.sleep(max(0, screenshot_interval - elapsed_time))
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)

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

if __name__ == "__main__":
    main()
