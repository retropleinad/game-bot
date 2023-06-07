import pandas as pd
import cv2
import numpy as np

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.layers import Input

TEST_FILE = 'D:/Python Projects/gameBot/recording output/gameplay'
BATCH_SIZE = 12


class VideoParser:

    def __init__(self, file_name, train_test_split=0.7):
        self.train_test_split = train_test_split
        self.video = cv2.VideoCapture(file_name + '.avi')
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.csv_file_name = file_name + '.csv'
        self.keys_df_headers = pd.read_csv(self.csv_file_name, nrows=1).columns
        self.keys_df = None

    def parse_video(self, num_frames, start=0):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        frame_ids = []

        i = start
        while i < start + num_frames:
            frame_exists, current_frame = self.video.read()

            if frame_exists:
                frames.append(current_frame)
                frame_ids.append(i)
            i += 1

        return pd.DataFrame(data={'id': frame_ids, 'frame': frames})

    # Starts at 0 and includes provided start frame
    def parse_csv(self, num_frames, start=1):
        keyboard_df = pd.read_csv(self.csv_file_name,
                                  skiprows=start, nrows=num_frames,
                                  header=None, names=self.keys_df_headers)
        return keyboard_df

    def remove_empty_frames(self, df, df_name):
        i = 0
        code = '{0} = {0}['.format(df_name)
        for col in df.columns:
            if col not in ('timestamp', 'id', 'frame'):
                code += '({0}[\'{1}\'] != 0.0) '.format(df_name, col)
                if i == len(df.columns) - 1:
                    code += ']'
                else:
                    code += '|'
            i += 1
        exec(code)

    # Main issue: Column headers to later rows for df
    def parse_dataset(self, num_frames, start=0):
        self.keys_df = self.parse_csv(num_frames, start+1)
        self.remove_empty_frames(self.keys_df, 'self.keys_df')
        if self.keys_df.shape[0] != 0:
            video_df = self.parse_video(num_frames, start)
            data = self.keys_df.merge(video_df, left_on='id', right_on='id')
            return data
        else:
            return None

    def generate_split_indexes(self):
        total_frames = cv2.CAP_PROP_FRAME_COUNT
        permutation = np.random.permutation(total_frames)
        train_up_to = int(total_frames * self.train_test_split)

        train_idx = permutation[:train_up_to]
        test_idx = permutation[train_up_to:]

        train_up_to = int(train_up_to * self.train_test_split)
        train_idx, validation_idx = train_idx[:train_up_to], train_idx[train_up_to:]

        return train_idx, validation_idx, test_idx

    def get_total_frames(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def quit(self):
        self.video.release()


class KeyModel:

    def __init__(self, input_shape, initial_learn_rate, epochs):
        # [BATCH_SIZE, 876, 1616, 3]
        self.input_shape = input_shape
        self.initial_learn_rate = initial_learn_rate
        self.epochs = epochs
        self.model = None

        self.assemble_model()
        self.compile_model()
        self.fit_model()

    def make_default_hidden_layers(self, data):
        x = Conv2D(16, (3, 3), padding='same', input_shape=self.input_shape)(data)
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

    def w_press(self, data):
        x = self.make_default_hidden_layers(data)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(2)(x)
        x = Activation('sigmoid', name='w_press')(x)
        return x

    def w_release(self, data):
        x = self.make_default_hidden_layers(data)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(2)(x)
        x = Activation('sigmoid', name='w_release')(x)
        return x

    def key_press_release(self, name):
        pass

    def assemble_model(self):
        inputs = Input(shape=self.input_shape)

        w_press_branch = self.w_press(inputs)
        w_release_branch = self.w_release(inputs)

        self.model = Model(inputs=inputs,
                           outputs=[w_press_branch, w_release_branch],
                           name='tree_farm')

    def compile_model(self):
        optimizer = Adam(learning_rate=self.initial_learn_rate,
                         decay=self.initial_learn_rate / self.epochs)
        self.model.compile(optimizer=optimizer,
                           loss={'w_press': 'binary_crossentropy',
                                 'w_release': 'binary_crossentropy'},
                           loss_weights={'w_press': 0.1,
                                         'w_release': 0.1},
                           metrics={'w_press': 'accuracy',
                                    'w_release': 'accuracy'})
        
# https://pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    def fit_model(self):
        batch_size = 12
        train_parser = VideoParser(TEST_FILE)
        train_idx, validation_idx, test_idx = train_parser.generate_split_indexes()
        train_gen = train_parser.parse_dataset(BATCH_SIZE)
        train = self.model.fit(train_idx,
                                         epochs=self.epochs,
                                         validation_data=validation_idx)


# https://www.tutorialspoint.com/tensorflow/image_recognition_using_tensorflow.htm
# https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
# https://www.tensorflow.org/tutorials/images/cnn
# https://stackoverflow.com/questions/47665391/keras-valueerror-input-0-is-incompatible-with-layer-conv2d-1-expected-ndim-4
# https://stackoverflow.com/questions/67345171/valueerror-input-0-of-layer-conv2d-is-incompatible-with-the-layer-expected-m?rq=3
def main():
    batch_size = 12
    shape = [876, 1616, 3]

    parser = VideoParser(TEST_FILE)
    df = parser.parse_dataset(batch_size)

    model = KeyModel(shape, .0004, 100)


main()
# Next
# 1.) Add training
# 2.) Figure out shape

