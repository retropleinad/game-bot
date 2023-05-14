import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import time

from pynput import keyboard
from pynput import mouse

class Recorder:

    def __init__(self, out_filename, window_name='minecraft', codec='MJPG', fps=30., record_seconds=10):
        self.out_filename = out_filename
        self.window_name = window_name
        self.codec = codec
        self.fps = fps
        self.record_seconds = record_seconds

        self.codec = cv2.VideoWriter_fourcc(*'{0}'.format(codec))
        self.window = gw.getWindowsWithTitle(window_name)[0]
        self.screen_width, self.screen_height = window.size
        self.screen_size = (screen_width, screen_height)

        self.out = cv2.VideoWriter(out_file, codec, fps, screen_size)

        self.pressed = []
        self.released = []
        self.keyboard_out = []

    def on_press(self, key):
        try:
            k = key.char
        except:
            k = key.name
        self.pressed.append(k)

    def on_release(self, key):
        try:
            k = key.char
        except:
            k = key.name
        self.released.append(k)

    def run(self):
        start_time = time.time()
        end_time = start_time + self.record_seconds
        current_time = time.time()

        while current_time < end_time:
            image = pyautogui.screenshot(region=(self.window.left,
                                                 self.window.top,
                                                 self.window.width,
                                                 self.window.height))
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
            current_time = time.time()

    def quit(self):
        cv2.destroyAllWindows()
        self.out.release()


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