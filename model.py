import pandas as pd
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers import BatchNormalization

TEST_FILE = 'D:/Python Projects/gameBot/recording output/gameplay'


def parse_video():
    video = cv2.VideoCapture(TEST_FILE + ".avi")
    fps = video.get(cv2.CAP_PROP_FPS)

    timestamps = [video.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]
    frames = [[]]
    frame_ids = [0]

    run = True
    i = 1

    while run:
        frame_exists, current_frame = video.read()

        if frame_exists:
            timestamps.append(video.get(cv2.CAP_PROP_POS_MSEC))
            calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)
            frames.append(current_frame)
            frame_ids.append(i)
            i += 1
        else:
            run = False

    video.release()

    for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
        print('Frame %d difference:' % i, abs(ts - cts))

    return pd.DataFrame(data={'id': frame_ids,
                              'timestamp': timestamps,
                              'calc_timestamp': calc_timestamps,
                              'frame': frames})


def build_dataset():
    video_df = parse_video()
    keyboard_df = pd.read_csv(TEST_FILE + ".csv")
    data = keyboard_df.merge(video_df, left_on='id', right_on='id')
    data = data.drop(columns=['timestamp_x', 'timestamp_y'])
    return data


def make_default_hidden_layers(data):
    x = Conv2D(16, (3, 3), padding='same')(data)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    return x


def w_press(data):
    x = make_default_hidden_layers(data)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2)(x)
    x = Activation('sigmoid', name='w_press')(x)
    return x


# https://www.tutorialspoint.com/tensorflow/image_recognition_using_tensorflow.htm
# https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
def build_model(data):
    data['mouse_x'] = data['mouse_x'] / data['mouse_x'].max()
    data['mouse_y'] = data['mouse_y'] / data['mouse_y'].max()

    x = data['frame'].values
    y = data.drop(columns=['id', 'frame']).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    y_train = np_utils.to_categorical(y_train)

    num_classes = y_train.shape[1]

    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=(64, 64, 3),
                     padding='same',
                     activation='relu',
                     kernel_constraint=maxnorm(3)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(512,
                    activation='relu',
                    kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(num_classes, activation='softmax')

    epochs = 10
    learn_rate = 0.01
    decay = learn_rate / epochs
    optimizer = SGD(lr=learn_rate, momentum=0.9, decay=decay, nesterov=False)
    model.complie(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=True)
    return True


build_model(build_dataset())

# Next
# 1.) Figure out matching timestamps
