import pandas as pd
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers import BatchNormalization

TEST_FILE = 'D:/Python Projects/gameBot/recording output/gameplay'


# https://stackoverflow.com/questions/7755684/flatten-opencv-numpy-array
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

            current_frame = current_frame.T.flatten()
            frames.append(current_frame)

            frame_ids.append(i)
            i += 1
        else:
            run = False

    video.release()

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
    x = Conv2D(16, (3, 3), padding='same', input_shape=[2, data.shape[1]])(data)
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
# https://www.tensorflow.org/tutorials/images/cnn
def build_model(data):
    data['mouse_x'] = data['mouse_x'] / data['mouse_x'].max()
    data['mouse_y'] = data['mouse_y'] / data['mouse_y'].max()

    x = data['frame'].to_list()
    y = data['w_press'].values

    x = np.asarray(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    y_train = np_utils.to_categorical(y_train)
    # x_train = np.asarray(x_train).astype('float32')

    w_press_branch = w_press(x_train)

    model = Model(inputs=y_train,
                  outputs=[w_press_branch],
                  name='tree_farm')

    initial_learn_rate = .0004
    epochs = 100
    optimizer = Adam(learning_rate=initial_learn_rate, decay=initial_learn_rate / epochs)

    model.compile(optimizer=optimizer,
                  loss={'w_press': 'binary_crossentropy'},
                  loss_weights={'w_press': 0.1},
                  metrics={'w_press': 'accuracy'})
    return model


build_model(build_dataset())

# Next
# 1.) Add training
# 2.) Figure out shape
