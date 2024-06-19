import cv2
import numpy as np
import pandas as pd
import os
import time
import threading
from playsound import playsound
import pyautogui
import keyboard
import pygetwindow as gw
from tkinter import Tk, simpledialog, messagebox

# Set up paths and parameters
output_dir = 'output/screenshots'
labels_file = 'labels.csv'
screenshot_interval = 0.5  # Time interval in seconds

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load mask (assuming it's a binary mask)
mask = cv2.imread('images/mask.png', 0)

# Initialize variables
capturing = False
round_screenshots = []
selected_window = None

# Function to play sound
def play_sound(sound_file):
    try:
        playsound(sound_file)
    except Exception as e:
        print(f"Error playing sound: {e}")

# Function to extract and save minimap
def extract_minimap(image, mask):
    minimap = cv2.bitwise_and(image, image, mask=mask)
    minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    minimap_resized = cv2.resize(minimap_gray, (64, 64))
    minimap_normalized = minimap_resized / 255.0
    return minimap_normalized

# Function to handle keyboard input
def keyboard_listener():
    global capturing
    while True:
        if keyboard.is_pressed('p'):
            capturing = not capturing
            if capturing:
                print("CAPTURING")
                threading.Thread(target=play_sound, args=('sounds/start_sound.mp3',)).start()
            else:
                print("ENDED")
                threading.Thread(target=play_sound, args=('sounds/stop_sound.mp3',)).start()
                ask_result_and_save()
            time.sleep(1)  # Add a delay to avoid multiple toggles

# Function to ask for the result and save screenshots
def ask_result_and_save():
    global round_screenshots
    root = Tk()
    root.withdraw()
    result = simpledialog.askstring("Input", "Enter the result (1 for win, 0 for loss):")
    if result is not None:
        result = int(result)
        with open(labels_file, 'a') as f:
            for ts, img in round_screenshots:
                f.write(f'{img},{result}\n')
        round_screenshots = []

# Function to select the window to capture
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

# Start the keyboard listener thread
def start_keyboard_listener():
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    while not keyboard_thread.is_alive():
        time.sleep(0.1)  # Wait for the thread to be fully started
    print("Ready")

# Select the window to capture
select_window()

# Ensure the window is selected before starting the keyboard listener
if selected_window:
    start_keyboard_listener()

# Capture screenshots from the selected window
while True:
    if capturing and selected_window:
        timestamp = time.time()
        screenshot = pyautogui.screenshot(region=(selected_window.left, selected_window.top, selected_window.width, selected_window.height))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = os.path.join(output_dir, f'{timestamp}.jpg')  # Change file extension to .jpg
        # Save as JPEG with compression quality set to 95 (adjust as needed)
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        round_screenshots.append((timestamp, filename))
        time.sleep(screenshot_interval)