"""
recording.py:
"""

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import time
import os
import pandas as pd

from pynput import keyboard
from pynput import mouse


TEST_FILE = 'D:/Python Projects/gameBot/recording output/gameplay'


class Recorder:

    """
    Recorder:
    Description: This class allows us to record the screen, keyboard, and mouse all at the same time.
                 The screen is output to a video file, while the keyboard/mouse are output to a csv

    Methods:
    1.) on_press(self, key):
    2.) on_release(self, key):
    3.) on_move(self, x, y):
    4.) on_click(self, x, y, button, pressed):
    5.) run(self, record_seconds=10):
    6.) quit(self):
    7.) future: cleaning methods

    __init__:

    Variables Called:
    1.) out_filename: The name of the file for the output file. This will be the same for video and csv
    2.) window_name: What is the name of the window/program we're recording?
    3.) codec: What codec are we using?
    4.) fps: At what fps should we record?

    Variables Initialized:
    1.) self.codec: The codec variable we're using
    2.) self.window: The window we're recording
    3.) self.screen_width: The width of the window we're recording
    4.) self.screen_height: The height of the window we're recording
    5.) self.screen_size: The tuple of the width by height
    6.) self.keys_pressed: Keeps track of the keys pressed in a particular frame
    7.) self.keys_released: Keeps track of the keys released in a particular frame
    8.) self.mouse_moved: Keeps track of where the mouse moved in a particular frame
    9.) self.mouse_clicked: Keeps track of where/how the mouse clicked in a particular frame
    10.) self.keyboard_out: Used to keep track of data for all frames
    11.) self.keyboard_outfile: The name of the file we're exporting to
    12.) self.outfile_headers: The headers for the export file
    13.) self.df_out: The dataframe we use for transforming the output
    """

    def __init__(self, out_filename, window_name='minecraft', codec='MJPG', fps=30.):
        # Initialize the codec we're using
        self.codec = cv2.VideoWriter_fourcc(*'{0}'.format(codec))

        # Grab the window we're recording and its dimensions
        self.window = gw.getWindowsWithTitle(window_name)[0]
        self.screen_width, self.screen_height = self.window.size
        self.screen_size = (self.screen_width, self.screen_height)

        # Initialize video writer to record video
        self.out = cv2.VideoWriter(out_filename + '.avi', self.codec, fps, self.screen_size)

        # Initialize lists used for tracking mouse and keyboard
        self.frame_tracking = dict()
        self.initialize_frame_tracking()

        # Initialize variables needed for creating the outfile
        self.keyboard_out = []
        self.keyboard_outfile = out_filename + '.csv'
        self.df_out = pd.DataFrame()

    def initialize_frame_tracking(self):
        self.frame_tracking = {
            'id': 0,
            'timestamp': 0.,
            'mouse_x': 0.,
            'mouse_y': 0.
        }

        keys = ('1', '2', '3', '4', '5', '6', '7', '8', '9',
                'space', 'w', 'a', 's', 'd', 'shift',
                'lmouse', 'rmouse')

        for key in keys:
            self.frame_tracking[key + '_press'] = 0.
            self.frame_tracking[key + '_release'] = 0.

    """
    pynput key functions
    
    on_press: tells the listener what to do when a key is pressed
    on_release: tells the listener what to do when a key is released
    
    In both instances, we append keys to the appropriate list for their frame
    """

    def on_press(self, key):
        try:
            k = key.char
        except:
            k = key.name
        self.frame_tracking[k + '_press'] = 1.

    def on_release(self, key):
        try:
            k = key.char
        except:
            k = key.name
        self.frame_tracking[k + '_release'] = 1.

    """
    pynput mouse functions
    
    on_move: tells the listener what to do when the mouse is moved
    on_click: tells the listener what to do when the mouse button is clicked
    
    """

    def on_move(self, x, y):
        self.frame_tracking['mouse_x'] = x
        self.frame_tracking['mouse_y'] = y

    def on_click(self, x, y, button, pressed):
        # Determine which button is clicked and save it
        if button == mouse.Button.left:
            if pressed:
                self.frame_tracking['lmouse_press'] = 1.
            else:
                self.frame_tracking['lmouse_release'] = 1.
        elif button == mouse.Button.right:
            if pressed:
                self.frame_tracking['rmouse_press'] = 1.
            else:
                self.frame_tracking['rmouse_release'] = 1.

        # Keep track of if the button was pressed or released and save it
        self.frame_tracking['mouse_x'] = x
        self.frame_tracking['mouse_y'] = y
    
    def remove_empty_frames(self):
        # Remove columns where nothing happened during that frame
        # Should we also remove ones that are just mouse movement?
        i = 0
        code = 'self.df_out = self.df_out['
        for key, value in self.frame_tracking.items():
            if key not in ('timestamp', 'id'):
                code += '(self.df_out[\'{0}\'] != 0.0) '.format(key)
                if i == len(self.frame_tracking) - 1:
                    code += ']'
                else:
                    code += '|'
            i += 1
        exec(code)

    def clean_output(self):
        self.df_out = pd.DataFrame.from_dict(self.keyboard_out)
        # Remove empty frames
        self.remove_empty_frames()
        # For each key, track held vs released
        # Make sure positions are right

    def run(self, record_seconds=10):
        start_time = time.time()
        end_time = start_time + record_seconds
        current_time = time.time()

        keyboard_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        mouse_listener = mouse.Listener(on_move=self.on_move,
                                        on_click=self.on_click)
        keyboard_listener.start()
        mouse_listener.start()

        i = 0

        while current_time < end_time:
            image = pyautogui.screenshot(region=(self.window.left,
                                                 self.window.top,
                                                 self.window.width,
                                                 self.window.height))
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out.write(frame)

            self.frame_tracking['timestamp'] = (current_time - start_time) * 1000
            self.frame_tracking['id'] = i
            self.keyboard_out.append(self.frame_tracking)

            self.initialize_frame_tracking()
            current_time = time.time()
            i += 1

        self.clean_output()

        if os.path.isfile(self.keyboard_outfile):
            os.remove(self.keyboard_outfile)

        self.df_out.to_csv(self.keyboard_outfile, header='column+names', index=False)

    def quit(self):
        cv2.destroyAllWindows()
        self.out.release()


r = Recorder(TEST_FILE)
r.run()
r.quit()

# Next comment everything