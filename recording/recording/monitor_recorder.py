"""
https://www.thepythoncode.com/article/make-screen-recorder-python
https://towardsdatascience.com/developing-your-own-screen-recording-software-with-python-927ab25fbfc6
"""

import cv2
import numpy as np
import pyautogui
import time

# Obtain the screen height and width
screen_width, screen_height = pyautogui.size()
screen_size = (screen_width, screen_height)

# Creating the 4-character
four_cc = cv2.VideoWriter_fourcc(*'MJPG')

# Declare fps
fps = 30.

# Output directory
file_name = 'screen_test.avi'
result = cv2.VideoWriter(file_name, four_cc, fps, screen_size)

# Setting the start timer for the required duration
start_time = time.time()
duration = 15
end_time = start_time + duration
current_time = time.time()

while current_time < end_time:

    image = pyautogui.screenshot()
    frames = np.array(image)
    frames_RGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    # Save the result
    result.write(frames_RGB)

    # update time
    current_time = time.time()



# Next step:
# 1.) Finish towardsdatascience tutorial
# 2.) Record for a specific screen
# 3.) How are we lining up the screen with the keys/mouse? special video format?