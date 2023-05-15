import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import time
import os
import csv

from pynput import keyboard
from pynput import mouse


TEST_FILE = 'D:/Python Projects/gameBot/recording output/gameplay'


class Recorder:

    def __init__(self, out_filename, window_name='minecraft', codec='MJPG', fps=30.):
        self.codec = cv2.VideoWriter_fourcc(*'{0}'.format(codec))
        self.window = gw.getWindowsWithTitle(window_name)[0]
        self.screen_width, self.screen_height = self.window.size
        self.screen_size = (self.screen_width, self.screen_height)

        self.out = cv2.VideoWriter(out_filename + '.avi', self.codec, fps, self.screen_size)

        self.keys_pressed = []
        self.keys_released = []
        self.mouse_moved = []
        self.mouse_clicked = []

        self.keyboard_out = []
        self.keyboard_outfile = out_filename + '.csv'
        self.outfile_headers = ('timestamp', 'keys_pressed', 'keys_released', 'mouse_moved', 'mouse_clicked')

    def on_press(self, key):
        try:
            k = key.char
        except:
            k = key.name
        self.keys_pressed.append(k)

    def on_release(self, key):
        try:
            k = key.char
        except:
            k = key.name
        self.keys_released.append(k)

    def on_move(self, x, y):
        self.mouse_moved.append((x, y))

    def on_click(self, x, y, button, pressed):
        button_clicked = None

        if button == mouse.Button.left:
            button_clicked = 'left'
        elif button == mouse.Button.right:
            button_clicked = 'right'

        if pressed:
            self.mouse_clicked.append((x, y, button_clicked, 'pressed'))
        else:
            self.mouse_clicked.append((x, y, button_clicked, 'released'))

    def clean_mouse_position(self):
        pass

    def clean_csv(self):
        pass

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

        while current_time < end_time:
            image = pyautogui.screenshot(region=(self.window.left,
                                                 self.window.top,
                                                 self.window.width,
                                                 self.window.height))
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out.write(frame)

            self.keyboard_out.append([current_time,
                                      self.keys_pressed, self.keys_released,
                                      self.mouse_moved, self.mouse_clicked])
            self.keys_pressed = []
            self.keys_released = []
            self.mouse_moved = []
            self.mouse_clicked = []

            current_time = time.time()

        if os.path.isfile(self.keyboard_outfile):
            os.remove(self.keyboard_outfile)

        with open(self.keyboard_outfile, newline='', mode='a') as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            writer.writerow(self.outfile_headers)

            for k in self.keyboard_out:
                writer.writerow(k)

    def quit(self):
        cv2.destroyAllWindows()
        self.out.release()


r = Recorder(TEST_FILE)
r.run()
r.quit()

# Next comment everything