"""
https://www.thepythoncode.com/article/make-screen-recorder-python
"""

import cv2
import numpy as np
import pyautogui

# Display screen resolution
SCREEN_SIZE = tuple(pyautogui.size())

# Define the codc
fourcc = cv2.VideoWriter_fourcc(*"XVID")

# fps
fps = 12.

# Create the video write object
out = cv2.VideoWriter('output.avi', fourcc, fps, SCREEN_SIZE)

# The time you want to record in seconds
record_seconds = 10

for i in range(int(record_seconds * fps)):
    # Make a screenshot
    img = pyautogui.screenshot()
    # Convert these pizels to a proper numpy array to work with Open CV
    frame = np.array(img)
    # Convert colors from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Write the frame
    out.write(frame)
    # Show the frame
    # cv2.imshow('screenshot', frame)


# Make sure everything is closed when exited
cv2.destroyAllWindows()
out.release()