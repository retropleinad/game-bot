import pandas as pd
import cv2
import numpy as np
import tensorflow as tf

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
from keras.layers import Input

INPUT_CSV = 'D:/Python Projects/gameBot/processed output/gameplay.csv'
INPUT_AVI = 'D:/Python Projects/gameBot/recording output/gameplay.avi'
BATCH_SIZE = 12


class VideoParser(tf.keras.utils.Sequence):

    """
    class VideoParser

    """

    def __init__(self,                  # self
                 csv_file_name,             # Name of the file without extension
                 avi_file_name,
                 processed_avi_file_name,
                 y_labels,              # Columns we want to predict
                 batch_size=12):        # How many frames are in each batch?

        # Use file_name to find the video file and the csv
        self.video = cv2.VideoCapture(avi_file_name)
        self.csv_file_name = csv_file_name

        # Load headers and save column names we care about
        self.keys_df_headers = pd.read_csv(self.csv_file_name, nrows=1).columns
        self.y_labels = y_labels

        # Save fields related to batches and size of video
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.batch_size = batch_size
        processed_video = cv2.VideoCapture(processed_avi_file_name)
        self.num_frames = int(processed_video.get(cv2.CAP_PROP_FRAME_COUNT))

    """
    __getitem__:
    Inputs:
        index: Where in the csv/video file to start the batch
    Description:
        Generate one batch of data
        Required for inheriting tf sequence
    Returns: (x, y)
        x will be a numpy array of shape [batch_size, input_height, input_width, input_channel]
        y will be a tuple with numpy arrays of shape [batch_size, label]
    """
    def __getitem__(self, index):
        # Run parse dataset
        df = self.__parse_dataset(self.batch_size, start=index, labels=self.y_labels)

        # Preprocess X
        x = df['frame'].to_list()
        x = np.asarray(x).astype('uint8')
        # x = x / 255

        # Preprocess Y
        y_values = []
        for y_label in self.y_labels:
            y_values.append(tf.keras.utils.to_categorical(df[y_label], 2))
        y_values = tuple(y_values)

        return x, y_values

    """
    __len__:
    Description:
        Required for inheriting tf sequence
        Must return int
    Returns:
        The number of batches the generator can produce
    """
    def __len__(self):
        num_batches = int(self.num_frames // self.batch_size)
        return num_batches

    def __parse_video(self, num_frames=0, ids=None, start=0):
        if num_frames == 0:
            return self.__parse_video_specific(ids)
        return self.__parse_video_iterative(num_frames, start)

    """
    __parse_video_iterative:
    Inputs:
        num_frames: The number of frames we want to parse
        start: What frame do we want to start parsing on?
    Description:
        Parse frames from the video file
    Returns:
        pd DataFrame with two columns: id and frame
        ID is later used to match with ID in csv
        Frame is then loaded frame in np array format
    Called By:
        __parse_dataset:
    """
    def __parse_video_iterative(self, num_frames, start=0):
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

    def __parse_video_specific(self, ids):
        frames = []
        frame_ids = []

        for i in ids:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            frame_exists, current_frame = self.video.read()

            if frame_exists:
                frames.append(current_frame)
                frame_ids.append(i)

        print('frame ids: ', frame_ids)
        return pd.DataFrame(data={'id': frame_ids, 'frame': frames})


    """
    __parse_csv:
    Inputs:
        num_frames: The number of frames we want to parse
        start: What frame do we want to start parsing on?
        labels: What columns do we want to return? None by default and returns all
    Description:
        Parse frames from the csv file
    Returns:
        pd DataFrame with id and key press/release columns
        ID is later used to match with ID from video
        press/release columns hold data on whether or not a key was pressed/released that frame
    Called By:
        __parse_dataset:
    """
    def __parse_csv(self, num_frames, start=1, labels=None):
        keyboard_df = pd.read_csv(self.csv_file_name,
                                  skiprows=start, nrows=num_frames,
                                  header=None, names=self.keys_df_headers)
        if labels is None:
            return keyboard_df

        cols = ['id']
        for label in labels:
            cols.append(label)

        print('csv ids: ', keyboard_df['id'].to_list())
        return keyboard_df[cols]

    """
    
    """
    def __parse_dataset(self, num_frames, start=0, labels=None):
        keys_df = self.__parse_csv(num_frames,
                                   start * self.batch_size + 1,
                                   labels)
        ids = keys_df['id'].to_list()
        if keys_df.shape[0] != 0:
            video_df = self.__parse_video(ids=ids)
            data = keys_df.merge(video_df, left_on='id', right_on='id')
            return data
        else:
            return None

    def get_total_frames(self):
        return self.num_frames

    def quit(self):
        self.video.release()


class KeyModel:

    def __init__(self, input_shape, initial_learn_rate=.004, epochs=20, batch_size=12, keys=('w_press', 'w_release')):
        # [BATCH_SIZE, 876, 1616, 3]
        self.input_shape = input_shape
        self.initial_learn_rate = initial_learn_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.keys = keys
        self.model = None

        self.assemble_model()
        self.compile_model()
        self.fit_model()

    def make_default_hidden_layers(self, data):
        # 2D Convolutional layer: good for pictures
        # filters: The number of output filters in the convolution
        # kernel_size: The height and width of the 2D convolution window
        # padding: 'valid' means no padding
        # padding: 'same' results in even padding to the left/right
        # padding: 'same' with strides=1 means output has the same size as input
        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=self.input_shape)(data)

        # ReLu: 0 if x < 0, otherwise x=y
        x = Activation('relu')(x)

        # Normalizes inputs
        # Keeps the mean output close to 0 and the output stdev close to 1
        # During training (when using fit()), normalizes using mean/stdev from current batch
        # During inference (when using predict()), normalizes output using moving average of mean/stdev
        # axis: The axis that should be normalized
        x = BatchNormalization(axis=-1)(x)

        # Downsamples the input by taking the maximum value over an input window
        # pool_size: The window size over which to take the maximum
        # Strides (default None): Specifies how far the pooling window moves for each pooling step
        # If strides is None, it will default to pool_size
        x = MaxPooling2D(pool_size=(3, 3))(x)

        # Randomly sets input units to 0
        # This is only true during training
        # rate: float between 0 and 1 that specifies the fraction of the input units to drop
        x = Dropout(0.25)(x)

        # Add another group of layers similar to those explained above
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Add another group of layers similar to those explained above
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        return x

    def add_key_branch(self, data, key_name):
        x = self.make_default_hidden_layers(data)

        # Flattens the input
        x = Flatten()(x)

        # A layer in which every neuron from the previous layer is connected to every neuron of the next layer
        # units: Dimensionality of the output space (number of neurons)
        x = Dense(128)(x)

        # ReLu: 0 if x < 0, otherwise x=y
        x = Activation('relu')(x)

        # Normalizes inputs
        # Keeps the mean output close to 0 and the output stdev close to 1
        # During training (when using fit()), normalizes using mean/stdev from current batch
        # During inference (when using predict()), normalizes output using moving average of mean/stdev
        x = BatchNormalization()(x)

        # Randomly sets input units to 0
        # This is only true during training
        # rate: float between 0 and 1 that specifies the fraction of the input units to drop
        x = Dropout(0.5)(x)

        # A layer in which every neuron from the previous layer is connected to every neuron of the next layer
        # units: Dimensionality of the output space (number of neurons)
        x = Dense(2)(x)

        # sigmoid(x) = 1 / (1 + exp(-x))
        # sigmoid is good for classification
        x = Activation('sigmoid', name=key_name)(x)
        return x

    def assemble_model(self):
        inputs = Input(shape=self.input_shape)

        branches = []
        for key in self.keys:
            branch = self.add_key_branch(inputs, key)
            branches.append(branch)

        self.model = Model(inputs=inputs,
                           outputs=branches,
                           name='tree_farm')

    def compile_model(self):
        optimizer = Adam(learning_rate=self.initial_learn_rate,
                         decay=self.initial_learn_rate / self.epochs)

        loss = {}
        loss_weights = {}
        metrics = {}

        for key in self.keys:
            loss[key] = 'binary_crossentropy'
            loss_weights[key] = 0.1
            metrics[key] = 'accuracy'

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           loss_weights=loss_weights,
                           metrics=metrics)
        
    # https://pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    # Current main problem: expects 1 input but is looking at all 12
    def fit_model(self):
        train_generator = VideoParser(INPUT_CSV, INPUT_AVI,
                                      'D:/Python Projects/gameBot/processed output/gameplay.avi',
                                      self.keys, batch_size=self.batch_size)
        test_generator = VideoParser(INPUT_CSV, INPUT_AVI, 'D:/Python Projects/gameBot/processed output/gameplay.avi',
                                     self.keys, batch_size=self.batch_size)

        train = self.model.fit(train_generator,
                               validation_data=test_generator,
                               epochs=20)


# https://www.tutorialspoint.com/tensorflow/image_recognition_using_tensorflow.htm
# https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
# https://www.tensorflow.org/tutorials/images/cnn
# https://stackoverflow.com/questions/47665391/keras-valueerror-input-0-is-incompatible-with-layer-conv2d-1-expected-ndim-4
# https://stackoverflow.com/questions/67345171/valueerror-input-0-of-layer-conv2d-is-incompatible-with-the-layer-expected-m?rq=3
def main():
    pass