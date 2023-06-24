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
        __init__(self, out_filename, window_name='minecraft', codec='MJPG', fps=12.)
        _initialize_frame_tracking(self)
        on_press(self, key):
        on_release(self, key):
        on_move(self, x, y):
        on_click(self, x, y, button, pressed):
        _clean_output(self):
        run(self, record_seconds=10):
        quit(self):
    """

    def __init__(self, out_filename, window_name='minecraft', codec='MJPG', fps=12.):
        """
        Parameters
            out_filename: The name of the file for the output file. This will be the same for video and csv
            window_name: What is the name of the window/program we're recording?
            codec: What codec are we using?
            fps: At what fps should we record?

        Variables Initialized:
            self.codec: The codec variable we're using
            self.window: The window we're recording
            self.screen_width: The width of the window we're recording
            self.screen_height: The height of the window we're recording
            self.screen_size: The tuple of the width by height
            self.out: cv2 VideoWriter object to record and write video
            self.frame_tracking:
            self.keyboard_out: Used to keep track of data for all frames
            self.keyboard_outfile: The name of the file we're exporting to
            self.df_out: The dataframe we use for transforming the output
        """

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
        self._initialize_frame_tracking()

        # Initialize variables needed for creating the outfile
        self.keyboard_out = []
        self.keyboard_outfile = out_filename + '.csv'
        self.df_out = pd.DataFrame()

    def _initialize_frame_tracking(self):
        """
        Description:
            Prepares dict of keys to track if they're pressed in a given frame.
            Due to the nature of pynput, we use a class variable, self.frame_tracking,
            instead of returning a new dict.
        Called By:
            self.__init__()
            self.run()
        """

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
    
    In both instances, we update the key in the self.frame_tracking dict
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
    
    Both methods update the self.frame_tracking dict for mouse location and mouse press/release
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

    def _clean_output(self):
        """
        Description:
            Method for cleaning output before printing to csv.
            Currently only holds a transform from dict to pandas dataframe
        Called By:
            self.run()
        """
        self.df_out = pd.DataFrame.from_dict(self.keyboard_out)

    def run(self, record_seconds=10):
        """
        Parameters:
            record_seconds: How many seconds should we record the screen?
        Description:
            Method to call when we want to record.
            Records the screen to an avi and the keys/mouse to a csv
        """

        # Initialize times for recording timestamps later on
        start_time = time.time()
        end_time = start_time + record_seconds
        current_time = time.time()

        # Create pynput mouse and keyboard listener objects
        keyboard_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        mouse_listener = mouse.Listener(on_move=self.on_move,
                                        on_click=self.on_click)
        keyboard_listener.start()
        mouse_listener.start()

        # Use to keep track of frame id in csv
        i = 0

        # Main loop for recording
        while current_time < end_time:
            # Take a screenshot, process it, and write it to avi
            image = pyautogui.screenshot(region=(self.window.left,
                                                 self.window.top,
                                                 self.window.width,
                                                 self.window.height))
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out.write(frame)

            # Add timestamp and id to dict used for this row then append it to main output dict
            self.frame_tracking['timestamp'] = (current_time - start_time) * 1000
            self.frame_tracking['id'] = i
            self.keyboard_out.append(self.frame_tracking)

            # Reset dict used for recording row, track timestamp, and increase id
            self._initialize_frame_tracking()
            current_time = time.time()
            i += 1

        # Clean csv output
        self._clean_output()

        # Check if path to write to already exists and remove it if it does
        if os.path.isfile(self.keyboard_outfile):
            os.remove(self.keyboard_outfile)

        # Write output dict to csv
        self.df_out.to_csv(self.keyboard_outfile, header='column+names', index=False)

    def quit(self):
        """
        Description:
            Call at the end of using the Recorder object
            Calls cv2 memory cleanup methods
        """
        cv2.destroyAllWindows()
        self.out.release()


def main():
    r = Recorder(TEST_FILE)
    r.run(record_seconds=15)
    r.quit()


# main()