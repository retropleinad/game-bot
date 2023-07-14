import cv2
import pandas as pd
import numpy as np
import json


def resize_np_image(image_array, resize_percent):
    """
    Parameters:
        image_array: numpy array of image we're resizing
        resize_percent: The extent to which we're scaling the image. Percents should be input as decimals.
    Description:
        Function to resize a cv2 array converted to a numpy array
    Returns
        Resized array
    Called By
        EmptyFrameRemover._shrink_video_data()
        predict.Predictor.run_predictions()
    """
    width = int(image_array.shape[1] * resize_percent)
    height = int(image_array.shape[0] * resize_percent)
    return cv2.resize(image_array, [width, height], interpolation=cv2.INTER_AREA)


class EmptyFrameRemover:

    """
    Class EmptyFrameRemover:

    Description: Class used to remove empty rows

    Methods:
        __init__(self)
        df_remove_empty_frames(self, df)
        _remove_empty_frames_csv(self, output_csv_address, search_cols=None)
        _shrink_video_data(self, output_avi_address, resize_percent=1.)
        _save_json(self):
        remove_empty_frames(self, output_csv_address, output_avi_address, resize_percent=1., search_cols=None):
    """

    def __init__(self, json_address):
        """
        Parameters:
            json_address: The address of the json save file
        Variables Initialized:
            self.df_cleaned: df used for removing empty frames,
                must be a class variable due to how dynamic python works.
            self.json_save_data: dict object of loaded json
            self.json_address: saves the address of the json
        """
        self.df_cleaned = None
        self.json_save_data = json.load(open(json_address, 'r'))
        self.json_address = json_address

    def df_remove_empty_frames(self, df, search_cols=None):
        """
        Inputs:
            df: The pandas dataframe that you want to remove empty rows from
            search_cols: Columns to search if they're empty. If none, search all
        Description:
            Method to remove empty rows from a pandas dataframe
            Does not affect the input dataframe and instead returns an edited copy
        Returns:
            pandas df: A copy of the input dataframe with empty rows removed
        Called By:
            _remove_empty_frames_csv()
        """

        # Get dataframe and initial part of dynamic python string ready
        self.df_cleaned = df.copy()
        i = 0
        code = '{0} = {0}['.format('self.df_cleaned')

        # If search_cols is None, go with default, otherwise loop through specific cols
        if search_cols is None:
            # Loop through every column and append string to check if each column is empty
            for col in df.columns:
                if col not in ('timestamp', 'id', 'frame', 'mouse_x', 'mouse_y'):
                    code += '({0}[\'{1}\'] != 0.0) '.format('self.df_cleaned', col)
                    if i == len(df.columns) - 1:
                        code += ']'
                    else:
                        code += '|'
                i += 1
        else:
            # Loop through specific columns and append string to see if columns are empty
            for col in search_cols:
                code += '({0}[\'{1}\'] != 0.0) '.format('self.df_cleaned', col)
                if i == len(search_cols) - 1:
                    code += ']'
                else:
                    code += '|'
                i += 1

        # Execute the python string and return altered dataframe
        exec(code)
        return self.df_cleaned

    def _remove_empty_frames_csv(self, output_csv_address, search_cols=None):
        """
        Parameters:
            output_csv_address: the address of the csv we're outputting cleaned data
            search_cols: Columns to search if they're empty. If none, search all.
        Description:
            Takes a csv and removes rows where target columns have a value of 0
        Returns
            True to indicate successful run
        Called By:
            remove_empty_frames()
        """

        # Initialize header related data
        csv_header_df = pd.read_csv(self.json_save_data['recorded_csv_address'], nrows=0)
        csv_header_cols = csv_header_df.columns
        csv_header_df.to_csv(output_csv_address, header='column+names', index=False)

        # Iterator for where to pull rows from csv
        i = 0

        # Initiate int to track processed frames
        processed_total_frames = 0

        # Read the first 1000 rows of the csv
        keyboard_df = pd.read_csv(self.json_save_data['recorded_csv_address'],
                                  skiprows=1,
                                  nrows=1000,
                                  header=None,
                                  names=csv_header_cols)

        # Loop through each row of the df and clean them in batches
        while keyboard_df.shape[0] > 0:
            # Clean and output cleaned data to csv
            keyboard_df = self.df_remove_empty_frames(keyboard_df, search_cols)
            keyboard_df.to_csv(output_csv_address,
                               mode='a',
                               header=False,
                               index=False)

            # Save good ids, increase iterator by 1000, and read more rows
            processed_total_frames = processed_total_frames + keyboard_df.shape[0]
            i += 1000
            keyboard_df = pd.read_csv(self.json_save_data['recorded_csv_address'],
                                      skiprows=i + 1,
                                      nrows=1000,
                                      header=None,
                                      names=csv_header_cols)

        # Save number of remaining frames and processed file name to json
        self.json_save_data['processed_total_frames'] = processed_total_frames
        self.json_save_data['processed_csv_address'] = output_csv_address
        return True

    def _shrink_video_data(self, output_avi_address, resize_percent=1.):
        """
        Parameters:
            output_avi_address: the address of the avi we're outputting cleaned data
            resize_percent: The percentage by which we should resize a frame. For example, .5 would cut it in half
        Description:
            Given a list of frame_ids, only writes those to output file
        Returns:
            True to indicate successful run
        Called By:
            remove_empty_frames()
        """

        # Create video reader object
        video_reader = cv2.VideoCapture(self.json_save_data['recorded_avi_address'])

        # Grab fps from VideoCapture object to use in VideoWriter object
        read_fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        # Grab fourcc from VideoCapture object to use in VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        # Grab frame sizes from VideoCapture object to use in VideoWriter object
        read_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_percent)
        read_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_percent)
        frame_size = (read_width, read_height)

        # Create VideoWriter output object
        video_writer = cv2.VideoWriter(output_avi_address,
                                       fourcc,
                                       read_fps,
                                       frame_size)

        # Loop through the id of each frame we want to keep
        for i in range(0, int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Grab the frame
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, i)
            frame_exists, current_frame = video_reader.read()

            # Resize the frame
            if resize_percent != 1.:
                current_frame = resize_np_image(current_frame, resize_percent)

            # Convert frame to numpy array and output it
            current_frame = np.array(current_frame)
            video_writer.write(current_frame)

        # Save the processed avi address to the json dict
        self.json_save_data['input_shape'] = current_frame.shape
        self.json_save_data['processed_avi_address'] = output_avi_address
        return True

    def _save_json(self):
        """
        Description:
            Call to save the json with total frames and paths for parsed files
        Returns:
            True to indicate a successful call
        """
        json.dump(self.json_save_data, open(self.json_address, 'w'))
        return True

    def remove_empty_frames(self, output_csv_address, output_avi_address, resize_percent=1., search_cols=None):
        """
        Parameters:
            output_csv_address: The address of the csv we're outputting cleaned data
            output_avi_address: The address of the avi we're outputting cleaned data
            resize_percent: The percentage by which we should resize a frame. For example, .5 would cut it in half
            search_cols: Columns to search if they're empty. If none, search all.
        Description:
            Removes empty frames from the avi and csv and creates new files with frames removed
            Updates json with total processed frames and addresses for processed files
        Returns:
            True to indicate successful run
        """

        # Process the csv and save good frame ids
        self._remove_empty_frames_csv(output_csv_address, search_cols)
        # Process the avi
        self._shrink_video_data(output_avi_address, resize_percent=resize_percent)
        # Save everything to the json file
        self._save_json()
        return True