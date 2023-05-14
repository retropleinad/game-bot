import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import sys
import time

# What is the name of the window
window_name = 'minecraft'

# Define the codec
codec = cv2.VideoWriter_fourcc(*'MJPG')

# Define the fps
fps = 30.

# Get the window
window = gw.getWindowsWithTitle(window_name)[0]

# Get the screen height and width
screen_width, screen_height = window.size
screen_size = (screen_width, screen_height)

# Create the video write object
out_file = '../gameplay.avi'
out = cv2.VideoWriter(out_file, codec, fps, screen_size)

# Set the start timer for the required duration
record_seconds = 10
start_time = time.time()
end_time = start_time + record_seconds
current_time = time.time()

while current_time < end_time:
    image = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
    current_time = time.time()

cv2.destroyAllWindows()
out.release()