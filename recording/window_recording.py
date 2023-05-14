import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import sys


# The window name, e.g 'notepad', 'chrome', etc
window_name = sys.argv[1]

# define the codc
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# frames per second
fps = 12.

# The time you want to record in seconds
record_seconds = 10

# Search for the window, getting the first matched with the title
w = gw.getWindowsWithTitle(window_name)[0]

# Activate the window
w.activate()

# Create the video write object
out = cv2.VideoWriter('output.avi', fourcc, fps, tuple(w.size))

for i in range(int(record_seconds * fps)):
    # Make a screenshot
    img = pyautogui.screenshot(region=(w.left,))