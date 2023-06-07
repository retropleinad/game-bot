from pynput import keyboard, mouse

saved_key = ''


def on_press(key):
    global saved_key
    try:
        saved_key = key.char
    except AttributeError:
        saved_key = key.name


def on_release(key):
    return True


key_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
key_listener.start()

mouse_control = mouse.Controller()
key_control = keyboard.Controller()

run = True
while run:
    if saved_key == 'esc':
        run = False
    if saved_key == 'f8':
        key_control.press('w')
    if saved_key == 'f7':
        key_control.release('w')