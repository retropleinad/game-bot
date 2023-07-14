import cv2
import pyautogui
import pygetwindow as gw
import numpy as np
import json

from pynput import keyboard, mouse
from keras.models import load_model
from preprocessing import resize_np_image

from model import create_weighted_binary_crossentropy


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

    def __init__(self, model_address, json_address, window_name='minecraft'):
        """
        Parameters
            model_address: The address of the model save
            window_name: The name of the window we're making predictions on
            json_address: The address of the json save file

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
        print('Beginning to load model')
        self.model = load_model(model_address,
                                custom_objects={'weighted_binary_crossentropy': create_weighted_binary_crossentropy})
        print('Finished loading model')

        # Initialize json save data
        self.json_save_data = json.load(open(json_address, 'r'))

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

        # Initialize mouse controller
        mouse_control = mouse.Controller()

        # Runs while true
        run = True
        # When true, make predictions
        predict = False

        # Main loop
        while run:
            # Basic keybinds to make Run false and toggle predictions
            if self.saved_key == 'f9':
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
                frame = resize_np_image(frame, .05)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame).astype('float16').reshape((1, 43, 80, 3))
                frame = (frame + 1) / 256

                # Output predictions from model
                predictions = self.model.predict(frame)

                coords = [-1000, -1000]
                # Remember lmouse & rmouse
                col_names = self.json_save_data['model_branch_ordinance']
                for i in range(0, len(col_names)):

                    if col_names[i] == 'mouse_x_normalized':
                        coords[0] = predictions[i][0][0] * 1616
                        if coords[1] != -1000:
                            pass
                            # mouse_control.move(coords[0], coords[1])
                    elif col_names[i] == 'mouse_y_normalized':
                        coords[1] = predictions[i][0][0] * 876
                        if coords[0] != -1000:
                            pass
                            # mouse_control.move(coords[0], coords[1])

                    elif predictions[i][0, 1] >= .5:
                        key = col_names[i].split('_')[0]

                        if i % 2 == 0:
                            print('Pressing ', col_names[i])
                            if key == 'lmouse':
                                mouse_control.press(mouse.Button.left)
                            elif key == 'rmouse':
                                mouse_control.press(mouse.Button.right)
                            else:
                                key_control.press(col_names[i][0])

                        else:
                            print('Releasing ', col_names[i])
                            if key == 'lmouse':
                                mouse_control.release(mouse.Button.left)
                            elif key == 'rmouse':
                                mouse_control.release(mouse.Button.right)
                            else:
                                key_control.release(col_names[i][0])
