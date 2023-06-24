import cv2
import pyautogui
import pygetwindow as gw
import numpy as np

from pynput import keyboard
from keras.models import load_model


class Predictor:

    """
    Predictor:
    Description:

    Methods:
        __init__(self, model_address, window_name)
        on_press(self, key)
        on_release(self, key)
        run_predictions(self)

    """

    def __init__(self, model_address, window_name='minecraft'):
        """
        Parameters
            model_address: The address of the model save
            window_name: The name of the window we're making predictions on

        Variables Initialized:
            self.saved_key: Saves a key when pressed - class variable due to pynput functionality
            self.window: The window we're making predictions on
            self.screen_width: Width of self.window
            self.screen_height: Height of self.window
            self.screen_size: Size of the window
            self.model: keras model loaded to make predictions
        """

        # Initialize saved key
        self.saved_key = ''

        # Initialize variables related to window and window size
        self.window = gw.getWindowsWithTitle(window_name)[0]
        self.screen_width, self.screen_height = self.window.size
        self.screen_size = (self.screen_width, self.screen_height)

        # Initialize keras model
        self.model = load_model(model_address)

    """
    pynput key functions
    
    on_press: Tells the listener what to do when a key is pressed (here we save it)
    on_release: Tells the listener what to do when a key is released (here we don't need to do anything)
    """

    def on_press(self, key):
        try:
            self.saved_key = key.char
        except AttributeError:
            self.saved_key = key.name

    def on_release(self, key):
        return True

    def run_predictions(self):
        """
        Description:
            Loops through window, feeds window into model, then makes predictions on key/mouse
            Then performs predicted key/mouse presses
            Should be the only method called after initializing this class
        """

        # Initialize keyboard listener and controller
        key_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        key_listener.start()
        key_control = keyboard.Controller()

        # Runs while true
        run = True
        # When true, make predictions
        predict = False

        # Main loop
        while run:
            # Basic keybinds to make Run false and toggle predictions
            if self.saved_key == 'esc':
                run = False
            if self.saved_key == 'f8':
                predict = True
            if self.saved_key == 'f7':
                predict = False

            if predict:
                # Grab the screenshot
                image = pyautogui.screenshot(region=(self.window.left,
                                                     self.window.top,
                                                     self.window.width,
                                                     self.window.height))

                # Do basic processing on frame to prepare it for model
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame).astype('uint8').reshape((1, 876, 1616, 3))
                # Frame should be ndarray (876, 1616, 3)

                # Output predictions from model
                predictions = self.model.predict(frame)

                # Hardcoded for: 'w_press', 'w_release', 'a_press', 'a_release'
                # If the model predicts hit/release, then perform that action
                if predictions[0][0, 1] > .5:
                    print('pressing w')
                    key_control.press('w')
                if predictions[1][0, 1] > .5:
                    pass
                    key_control.release('w')
                if predictions[2][0, 1] > .5:
                    print('pressing a')
                    key_control.press('a')
                if predictions[3][0, 1] > .5:
                    pass
                    key_control.release('a')