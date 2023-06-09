"""
https://stackoverflow.com/questions/11918999/key-listeners-in-python
https://pynput.readthedocs.io/en/latest/
https://pynput.readthedocs.io/en/latest/mouse.html
https://pynput.readthedocs.io/en/latest/keyboard.html#monitoring-the-keyboard
"""


import numpy as np

from pynput import keyboard
from pynput import mouse


def on_press(key):
    if key == keyboard.Key.esc:
        return False
    try:
        k = key.char
    except:
        k = key.name
    print('Key pressed: ' + k)


def on_release(key):
    k = key.char
    print('Key released: ' + k)


def on_move(x, y):
    print('Pointer moved to {0}'.format(x, y))


def on_click(x, y, button, pressed):
    print('{0} at {1}'.format('Pressed' if pressed else 'Released', (x, y)))


def on_scroll(x, y, dx, dy):
    print('Scrolled {0} at {1}'.format('down' if dy < 0 else 'up', (x, y)))


def listen():
    keyboard_listener = keyboard.Listener(on_press=on_press,
                                          on_release=on_release)
    mouse_listener = mouse.Listener(on_move=on_move,
                                    on_click=on_click,
                                    on_scroll=on_scroll)
    keyboard_listener.start()
    mouse_listener.start()
    keyboard_listener.join()

# Next steps: Format basic output file
listen()