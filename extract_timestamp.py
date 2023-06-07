# https://stackoverflow.com/questions/47743246/getting-timestamp-of-each-frame-in-a-video
# https://docs.opencv.org/3.3.1/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
import cv2


def stack_tutorial():
    video = cv2.VideoCapture('D:/Python Projects/gameBot/recording output/gameplay.avi')
    fps = video.get(cv2.CAP_PROP_FPS)

    timestamps = [video.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]

    run = True

    while run:
        frame_exists, current_frame = video.read()
        if frame_exists:
            timestamps.append(video.get(cv2.CAP_PROP_POS_MSEC))
            calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
        else:
            run = False

    video.release()

    for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
        print('Frame %d difference:' %i, cts)


stack_tutorial()