import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import math
import random
import json

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Input


def shuffle_batches(train_test_split, num_batches, simple=False, shuffle=True, json_address=None):
    """
    Params:
        train_test_split: Decimal value of what percent of the dataset should be training
        num_batches: how many batches are we shuffling
        simple: should we shuffle simply or more algorithmically intensively
        shuffle: should lists returned be shuffled
        json_address: Provide an address for a json if you want to save shuffle and train_test_split to it
    Description:
        Function to randomize batches
        Batches are 0-indexed
    Returns:
        dict where ['train'] is a list of indexes for train batches and ['test'] is for test batches
    Called By:
        KeyModel._fit_model()
    """

    # Initialize lists to save batch numbers for train and test batches
    train_batches = []
    test_batches = []

    # Create list with each batch
    # Randomly calculate index and grab batch
    # Add batch to return list and remove it from list with each batch
    if not simple:
        num_train = math.floor(train_test_split * num_batches)
        batches = [i for i in range(0, num_batches)]

        while len(train_batches) < num_train:
            r = random.randint(0, len(batches) - 1)
            train_batches.append(batches[r])
            batches.remove(batches[r])

        test_batches = batches

    # Alternatively, iterate through each batch and use weighted generator to assign output
    # Problem: Pseudo-random
    else:
        for i in range(0, num_batches):
            if random.random() <= train_test_split:
                train_batches.append(i)
            else:
                test_batches.append(i)

    # Shuffle where batch number appears in array
    if shuffle:
        random.shuffle(train_batches)
        random.shuffle(test_batches)

    # Save data to json
    if json_address is not None:
        try:
            json_save_data = json.load(open(json_address, 'r'))
            json_save_data['shuffle'] = shuffle
            json_save_data['train_test_split'] = train_test_split
            json.dump(json_save_data, open(json_address, 'w'))
        except FileNotFoundError:
            print('given json file does not exist')

    return {'train': train_batches, 'test': test_batches}


# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy
# https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99
def calculate_weights():
    """
    Should return a dict of {class_label: class_weight}
    Integrate weights into a customized loss function
    Cannot yet use built in class_weight
    :return:
    """
    pass


def weighted_categorical_crossentropy(class_weight):

    def loss(y_obs, y_pred):
        pass


class VideoParser(tf.keras.utils.Sequence):
    """
    class VideoParser

    Description:
        Data generator object to feed data into the model
        Parses data from both avi and csv files and transforms data to be ready for neural network
        Outputs in batches to prevent murdering RAM

    Methods:
        __init__()
        __getitem__(self, index)
        __len__(self)
        _parse_video(self, num_frames=0, ids=None, start=0)
        _parse_video_iterative(self, num_frames, start=0)
        _parse_video_specific(self, ids)
        _parse_csv(self, num_frames, start=1, labels=None)
        _normalize_mouse_pos(self, df, x_col_name, y_col_name)
        _parse_dataset_specific(self, num_frames, start=0, labels=None)
        get_total_frames(self)
        quit(self)
    """

    def __init__(self,
                 processed_csv_address,
                 processed_avi_address,
                 processed_total_frames,
                 y_labels,
                 mouse_x_max=None,
                 mouse_y_max=None,
                 batch_size=12,
                 batches=None):
        """
        Parameters:
            processed_csv_address: The address of the processed csv file
            initial_avi_address: The address of the unprocessed avi file
            processed_total_frames: The total number of frames in the processed file
            y_labels: Set the columns that we care about for y
            mouse_x_max: What is the maximum mouse x position value?
            mouse_y_max: What is the maximum mouse y position value?
            batch_size: How big of batches should we use?
            batches: Directly feed the model the batches to use

        Variables Initialized:
            self.video: Video file for the initial avi (which here we're working primarily with)
            self.processed_csv_address: The address of the processed csv file
            self.keys_df_headers: dataframe of column headers from csv
            self.y_labels: Save the columns that we care about for y
            self.fps: Save the fps of the video
            self.batch_size: Save the batch size
            self.num_frames: Calculate the number of frames we're working with
            self.batches: Save the specific batches we're pulling in this generator object
            self.mouse_x_max: Save the maximum x position for the mouse
            self.mouse_y_max: Save the maximum y position for the mouse
        """

        # Use file_name to find the video file and the csv
        self.video = cv2.VideoCapture(processed_avi_address)
        self.processed_csv_address = processed_csv_address

        # Load headers and save column names we care about
        self.keys_df_headers = pd.read_csv(self.processed_csv_address, nrows=1).columns
        self.y_labels = y_labels

        # Save fields related to batches and size of video
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.batch_size = batch_size
        self.num_frames = processed_total_frames
        self.batches = batches

        # Save mouse max values
        self.mouse_x_max = mouse_x_max
        self.mouse_y_max = mouse_y_max

    def __getitem__(self, index):
        """
        Inputs:
            index: Where in the csv/video file to start the batch
        Description:
            Generate one batch of data
            Required for inheriting tf sequence
        Returns: (x, y)
            x will be a numpy array of shape [batch_size, input_height, input_width, input_channel]
            y will be a tuple with numpy arrays of shape [batch_size, label]
        """

        # Run parse dataset
        if self.batches is None:
            df = self._parse_dataset_specific(self.batch_size, start=index, labels=self.y_labels)
        else:
            df = self._parse_dataset_specific(self.batch_size,
                                              start=self.batches[index],
                                              labels=self.y_labels)
        # Preprocess X
        x = df['frame'].to_list()
        x = np.asarray(x).astype('float16')
        x = (x + 1) / 256
        # x = x / 255

        # Preprocess Y
        y_values = []
        for y_label in self.y_labels:
            y_values.append(tf.keras.utils.to_categorical(df[y_label], 2))

        if self.mouse_x_max is not None:
            y_values.append(df['mouse_x_normalized'].astype('float16').to_numpy())
        if self.mouse_y_max is not None:
            y_values.append(df['mouse_y_normalized'].astype('float16').to_numpy())

        y_values = tuple(y_values)
        return x, y_values

    def __len__(self):
        """
        Description:
            Required for inheriting tf sequence
            Must return int
        Returns:
            The number of batches the generator can produce
        """

        if self.batches is not None:
            return len(self.batches)
        num_batches = int(self.num_frames // self.batch_size)
        return num_batches

    def _parse_video(self, num_frames=0, ids=None, start=0):
        """
        Parameters:
            num_frames: The number of frames to parse
            ids: List of ids to parse, overrids num_frames
            start: ID of frame to start with
        Description:
            Parse frames from the avi file
        Returns:
            Pandas dataframe of frame id and frame
        Called By:
            self.parse_dataset_specific()
        """

        if num_frames == 0:
            return self._parse_video_specific(ids)
        return self._parse_video_iterative(num_frames, start)

    def _parse_video_iterative(self, num_frames, start=0):
        """
        Parameters:
            num_frames: The number of frames we want to parse
            start: What frame do we want to start parsing on?
        Description:
            Parse frames from the video file.
            Starts at a given frame and parses the next n frames
        Returns:
            pd DataFrame with two columns: id and frame
            ID is later used to match with ID in csv
            Frame is then loaded frame in np array format
        Called By:
            _parse_video():
        """

        # Set video to start frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start)

        # Create lists for saving frames and ids
        frames = []
        frame_ids = []

        # Loop through each frame
        i = start
        while i < start + num_frames:
            # Read the frame
            frame_exists, current_frame = self.video.read()

            # Save the frame and the frame id
            if frame_exists:
                frames.append(current_frame)
                frame_ids.append(i)
            i += 1

        return pd.DataFrame(data={'id': frame_ids, 'frame': frames})

    def _parse_video_specific(self, ids):
        """
        Parameters:
            ids:
        Description:
            Parse frames from the video file.
            Parses frames at given ids
        Returns:
            pd DataFrame with two columns: id and frame
            ID is later used to match with ID in csv
            Frame is then loaded frame in np array format
        Called By:
            _parse_video():
        """

        # Create lists for saved frames
        frames = []

        # Loop through each id
        for i in ids:
            # Read the frame
            self.video.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            frame_exists, current_frame = self.video.read()

            # Save the frame
            if frame_exists:
                frames.append(current_frame)

        # print('frame ids: ', frame_ids)
        return pd.DataFrame(data={'id': ids, 'frame': frames})

    def _parse_csv(self, num_frames, labels, start=1):
        """
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
            self._parse_dataset_specific():
        """

        # Read the csv to a df
        keyboard_df = pd.read_csv(self.processed_csv_address,
                                  skiprows=start, nrows=num_frames,
                                  header=None, names=self.keys_df_headers)

        # Add each label to list
        cols = ['id']
        for label in labels:
            cols.append(label)

        # Normalize mouse positions
        if self.mouse_x_max is not None or self.mouse_y_max is not None:
            keyboard_df = self._normalize_mouse_pos(keyboard_df, 'mouse_x', 'mouse_y')

        # Add normalized mouse columns to list of cols we care about
        if self.mouse_x_max is not None:
            cols.append('mouse_x_normalized')
        if self.mouse_y_max is not None:
            cols.append('mouse_y_normalized')

        # Return only columns we're using
        return keyboard_df[cols]

    def _normalize_mouse_pos(self, df, x_col_name, y_col_name):
        """
        Inputs:
            df: The dataframe that contains mouse positions
            x_col_name: The name of the column with mouse x positions
            y_col_name: The name of the column with mouse y positions
        Description:
            Normalizes mouse positions
            Divides by max mouse x or y value for given axis to get value between 0-1
        Returns:
            dataframe with normalized mouse position columns appended
        Called By:
            self._parse_csv():
        """

        # Divide each mouse axis by max x or y value, respectively
        if self.mouse_x_max is not None:
            df['mouse_x_normalized'] = df[x_col_name] / self.mouse_x_max
        if self.mouse_y_max is not None:
            df['mouse_y_normalized'] = df[y_col_name] / self.mouse_y_max
        return df

    def _parse_dataset_specific(self, num_frames, labels, start=0):
        """
        Parameters:
            num_frames: How many frames should we parse?
            labels: What y-values do we want to parse from the csv?
            start: What frame should we start parsing at?
        Description:
            Calls methods to parse data from the csv and from the video
            Then merges data into one pandas df
        Returns:
            pandas df of merged dataset
        Called By:
            __getitem__()
        """

        # Parse data from the csv
        keys_df = self._parse_csv(num_frames, labels, start=start * self.batch_size + 1)

        # Get the ids of the frames from csv so that we can pull them from avi
        ids = keys_df['id'].to_list()

        if keys_df.shape[0] != 0:
            # Parse data from the avi
            video_df = self._parse_video(ids=ids)

            # Merge and return the data
            data = keys_df.merge(video_df, left_on='id', right_on='id')
            return data
        else:
            return None

    def get_total_frames(self):
        """
        Description
            Getter to retrieve total number of frames
        Returns
            Total number of frames
        """
        return self.num_frames

    def quit(self):
        """
        Description
            Call after parsing video
            Cleans up cv2 data from memory
        Returns
            True to indicate a successful call
        """
        self.video.release()
        return True


class KeyModel:

    """
    class KeyModel:

    Description:
        Our model class
        Builds the model

    Methods:
        __init__()
        _make_default_hidden_layers(self, data)
        _add_key_branch(self, data, key_name)
        _add_mouse_branch(self, data, mouse_axis)
        _assemble_model(self)
        _save_json(self, model_address):
        _compile_model(self)
        _fit_model(self)
        build_model(self, model_address):
    """

    def __init__(self,
                 json_address,
                 input_shape=None,
                 initial_learn_rate=.004,
                 epochs=20,
                 batch_size=12,
                 keys=None,
                 mouse=False):

        """
        Parameters
            json_address: The address of the json save file
            input_shape: ndarray dimensions for the model
            initial_learn_rate: model initial learn rate
            epochs: How many epochs should we train the model?
            batch_size: How big should each training/testing batch be?
            keys: What keys are we trying to predict?
            mouse: Are we also trying to predict mouse position?

        Variables Initialized
            self.json_save_data: Dict loaded from json save
            self.json_address: Saved name of the json file
            self.input_shape: ndarray dimensions for the model
            self.initial_learn_Rate: model initial learn rate
            self.epochs: How many epochs should we train the model?
            self.batch_size: How big should each training/testing batch be?
            self.keys: What key press/release are we trying to predict?
            self.model: The tensorflow model that we're building
            self.mouse: Boolean are we trying to predict mouse position
        """

        # [BATCH_SIZE, 876, 1616, 3]
        # Save json data
        self.json_save_data = json.load(open(json_address, 'r'))
        self.json_address = json_address

        # Initialize variables
        self.initial_learn_rate = initial_learn_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.mouse = mouse

        if input_shape is not None:
            self.input_shape = input_shape
        else:
            self.input_shape = self.json_save_data['input_shape']

        if keys is not None:
            self.keys = keys
        else:
            self.keys = self.json_save_data['keys']

    def _make_default_hidden_layers(self, data):
        """
        Parameters
            data: Input object of shape self.input_shape
        Description:
            Builds neural network initial hidden layers
            These layers are kind and gentle preprocessing type layers before we later add dense layers
        Returns
            Generated layers
        Called By
            self._add_key_branch()
            self.add_mouse_branch
        """

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

    def _add_key_branch(self, data, key_name):
        """
        Parameters
            data: Input object of shape self.input_shape
            key_name: The name of the key column for this y variable (example w_press or w_release)
        Description
            Adds a branch to the neural network to predict pressing/releasing a particular key.
            Note that key press and key release should be separate branches
        Returns
            Neural network branch
        Called By
            self._assemble_model()
        """

        # Build initial preprocessing type layers
        x = self._make_default_hidden_layers(data)

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

    # Screen width:
    # Be sure to make sure everything is positive
    # I think we can assume min value is 0
    # width = 1616, height = 876
    def _add_mouse_branch(self, data, mouse_axis):
        """
        Parameters
            data: Input object of shape self.input_shape
            mouse_axis: The name of the key column for this y variable (example mouse_x or mouse_y)
        Description
            Adds a branch to the neural network to predict mouse position along the x or y axis.
            Note that x and y axes should happen in separate branches.
        Returns
            Neural network branch
        Called By
            self._assemble_model()
        """

        # Build initial preprocessing layers to NN
        x = self._make_default_hidden_layers(data)

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
        # axis: The axis that should be normalized
        x = BatchNormalization()(x)

        # Randomly sets input units to 0
        # This is only true during training
        # rate: float between 0 and 1 that specifies the fraction of the input units to drop
        x = Dropout(0.5)(x)

        # A layer in which every neuron from the previous layer is connected to every neuron of the next layer
        # units: Dimensionality of the output space (number of neurons)
        x = Dense(1)(x)

        # Activation function: returns x = y
        x = Activation('linear', name=mouse_axis)(x)
        return x

    def _assemble_model(self):
        """
        Description:
            Assemble the model.
            This method pieces together all key and mouse branches for the model.
            Defines self.model.
            Does not compile or fit the model.
        Called By:
            self.__init__() (for now)
            self.build_model() (after updates)
        """

        # Create Input object that we feed to methods to build NN
        inputs = Input(shape=self.input_shape)

        # Build branches for each key press/release
        branches = []
        for key in self.keys:
            branch = self._add_key_branch(inputs, key)
            branches.append(branch)

        # Build branches for mouse position if we're predicting those
        if self.mouse:
            branches.append(self._add_mouse_branch(inputs, 'mouse_x_normalized'))
            branches.append(self._add_mouse_branch(inputs, 'mouse_y_normalized'))

        # Define the model object
        self.model = Model(inputs=inputs,
                           outputs=branches,
                           name='tree_farm')

    def _save_json(self, model_address):
        """
        Parameters:
            model_address: The address of where we're saving the model
        Description:
            Update and save our json with class variables
        Returns:
            True to indicate successful run
        Called By:
            self.build_model()
        """

        # Save class variables to json output dict
        self.json_save_data['batch_size'] = self.batch_size
        self.json_save_data['initial_learn_rate'] = self.initial_learn_rate
        self.json_save_data['epochs'] = self.epochs
        self.json_save_data['model_address'] = model_address

        # Save json and return true
        json.dump(self.json_save_data, open(self.json_address, 'w'))
        return True

    def _compile_model(self):
        """
        Description:
            Compile the model.
            This method generates optimizer, loss functions, and applies loss weights and accuracy.
            Then compiles self.model.
        Called By:
            self.__init__() (for now)
            self.build_model() (after updates)
        """

        # Build the optimizer for the NN
        optimizer = Adam(learning_rate=self.initial_learn_rate)

        # Create dicts to assign loss, loss_weights, and metrics
        loss = {}
        loss_weights = {}
        metrics = {}

        # Create list to keep track of the ordinance of each branch
        model_branch_ordinance = []

        # Populate the dicts for key press
        for key in self.keys:
            loss[key] = 'binary_crossentropy'
            loss_weights[key] = 0.1
            metrics[key] = 'accuracy'
            model_branch_ordinance.append(key)

        # Populate the dicts if we're calculating mouse position
        if self.mouse:
            loss['mouse_x_normalized'] = 'mse'
            loss_weights['mouse_x_normalized'] = 4.
            metrics['mouse_x_normalized'] = 'mae'

            loss['mouse_y_normalized'] = 'mse'
            loss_weights['mouse_y_normalized'] = 4.
            metrics['mouse_y_normalized'] = 'mae'

            model_branch_ordinance.append('mouse_x_normalized')
            model_branch_ordinance.append('mouse_y_normalized')

        # Compile the model
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           loss_weights=loss_weights,
                           metrics=metrics)

        # Output ordinance of branches to model
        self.json_save_data['model_branch_ordinance'] = model_branch_ordinance

    # https://pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
    def _fit_model(self):
        """
        Description:
            Still very hard-coded, needs to be changed.
            Calculates variables related to fitting the model.
            Then fits the model.
        Called By:
            self.__init__() (for now)
            self.build_model() (after updates)
        """

        # Grab the number of frames from json save and divide by batch size
        num_batches = int(self.json_save_data['processed_total_frames'] // self.batch_size)

        # Shuffle batch numbers for randomness in order and what we feed into train/test generators
        batches = shuffle_batches(train_test_split=.7,
                                  num_batches=num_batches,
                                  shuffle=True,
                                  json_address=self.json_address)

        # Assign mouse_x_max and mouse_y_max
        if self.mouse:
            mouse_x_max = 1616
            mouse_y_max = 876
        else:
            mouse_x_max = None
            mouse_y_max = None

        # Build train and test generators
        # The key difference between the two is setting batches equal to batches['train] or batches['test]
        train_generator = VideoParser(self.json_save_data['processed_csv_address'],
                                      self.json_save_data['processed_avi_address'],
                                      self.json_save_data['processed_total_frames'],
                                      y_labels=self.keys,
                                      batch_size=self.batch_size,
                                      batches=batches['train'],
                                      mouse_x_max=mouse_x_max,
                                      mouse_y_max=mouse_y_max)

        test_generator = VideoParser(self.json_save_data['processed_csv_address'],
                                     self.json_save_data['processed_avi_address'],
                                     self.json_save_data['processed_total_frames'],
                                     y_labels=self.keys,
                                     batch_size=self.batch_size,
                                     batches=batches['test'],
                                     mouse_x_max=mouse_x_max,
                                     mouse_y_max=mouse_y_max)

        # Fit the model (this likely will take a while)
        self.model.fit(train_generator, validation_data=test_generator, epochs=self.epochs)

    # 'D:/Python Projects/gameBot/models/tree_farm'
    def build_model(self, model_address="D:/Python Projects/gameBot/models/tree_farm"):
        """
        Description:
            Method to call to do everything involved with creating a model
            Calls functions to assemble, compile, and fit the model. As well as save the json and the model.
            This is the function that users should call, instead of calling each individual function
        Parameters:
            model_address: What is the address of where we should save the model?
        Returns:
            True to indicate a successful run
        """

        self._assemble_model()
        self._compile_model()
        self._fit_model()
        self.model.save(model_address)
        self._save_json(model_address)
        return True


# https://www.tutorialspoint.com/tensorflow/image_recognition_using_tensorflow.htm
# https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
# https://www.tensorflow.org/tutorials/images/cnn
# https://stackoverflow.com/questions/47665391/keras-valueerror-input-0-is-incompatible-with-layer-conv2d-1-expected-ndim-4
# https://stackoverflow.com/questions/67345171/valueerror-input-0-of-layer-conv2d-is-incompatible-with-the-layer-expected-m?rq=3
def main():
    pass