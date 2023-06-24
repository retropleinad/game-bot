import cv2
import pandas as pd
import numpy as np


class EmptyFrameRemover:

    """
    Class EmptyFrameRemover:

    Description: Class used to remove rows that lack outputs for dynamic keys

    Methods:
        __init__(self)
        df_remove_empty_frames(self, df)
    """

    def __init__(self):
        # df used for removing empty frames
        # Must be a class or global variable due to how dynamic python works
        self.df_remove_empty_frames = None

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
        self.df_remove_empty_frames = df.copy()
        i = 0
        code = '{0} = {0}['.format('self.df_remove_empty_frames')

        # If search_cols is None, go with default, otherwise loop through specific cols
        if search_cols is None:
            # Loop through every column and append string to check if each column is empty
            for col in df.columns:
                if col not in ('timestamp', 'id', 'frame', 'mouse_x', 'mouse_y'):
                    code += '({0}[\'{1}\'] != 0.0) '.format('self.df_remove_empty_frames', col)
                    if i == len(df.columns) - 1:
                        code += ']'
                    else:
                        code += '|'
                i += 1
        else:
            # Loop through specific columns and append string to see if columns are empty
            for col in search_cols:
                code += '({0}[\'{1}\'] != 0.0) '.format('self.df_remove_empty_frames', col)
                if i == len(df.columns) - 1:
                    code += ']'
                else:
                    code += '|'
                i += 1

        # Execute the python string and return altered dataframe
        exec(code)
        return self.df_remove_empty_frames


def _remove_empty_frames_csv(input_csv_address, output_csv_address, search_cols=None):
    """
    Parameters:
        input_csv_address: the address of the csv we're cleaning
        output_csv_address: the address of the csv we're outputting cleaned data
        search_cols: Columns to search if they're empty. If none, search all.
    Description:
        Takes a csv and removes rows where target columns have a value of 0
    Returns:
        List of ids of records not removed
    Called By:
        remove_empty_frames()
    """

    # Initialize frame remover
    frame_remover = EmptyFrameRemover()

    # Initialize header related data
    csv_header_df = pd.read_csv(input_csv_address, nrows=0)
    csv_header_cols = csv_header_df.columns
    csv_header_df.to_csv(output_csv_address,
                         header='column+names',
                         index=False)

    # List used to keep track of ids we don't remove
    ids = []
    # Iterator for where to pull rows from csv
    i = 0

    # Read the first 1000 rows of the csv
    keyboard_df = pd.read_csv(input_csv_address,
                              skiprows=1,
                              nrows=1000,
                              header=None,
                              names=csv_header_cols)

    # Loop through each row of the df and clean them in batches
    while keyboard_df.shape[0] > 0:
        # Clean and output cleaned data to csv
        keyboard_df = frame_remover.df_remove_empty_frames(keyboard_df, search_cols)
        keyboard_df.to_csv(output_csv_address,
                           mode='a',
                           header=False,
                           index=False)

        # Save good ids, increase iterator by 1000, and read more rows
        ids = ids + keyboard_df['id'].to_list()
        i += 1000
        keyboard_df = pd.read_csv(input_csv_address,
                                  skiprows=i + 1,
                                  nrows=1000,
                                  header=None,
                                  names=csv_header_cols)
    # Return ids of rows that we kept
    return ids


def _remove_empty_frames_video(input_avi_address, output_avi_address, frame_ids):
    """
    Parameters:
        input_avi_address: the address of the avi we're cleaning
        output_avi_address: the address of the avi we're outputting cleaned data
        frame_ids: ids of frames we want to keep
    Description:
        Given a list of frame_ids, only writes those to output file
    Returns:
        True to indicate successful run
    Called By:
        remove_empty_frames()
    """

    # Create video reader object
    video_reader = cv2.VideoCapture(input_avi_address)

    # Grab fps from VideoCapture object to use in VideoWriter object
    read_fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    # Grab fourcc from VideoCapture object to use in VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Grab frame sizes from VideoCapture object to use in VideoWriter object
    read_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    read_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (read_width, read_height)

    # Create VideoWriter output object
    video_writer = cv2.VideoWriter(output_avi_address,
                                   fourcc,
                                   read_fps,
                                   frame_size)

    # Loop through the id of each frame we want to keep
    for i in frame_ids:
        # Grab the frame
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        frame_exists, current_frame = video_reader.read()

        # Convert frame to numpy array and output it
        current_frame = np.array(current_frame)
        video_writer.write(current_frame)
    return True


def remove_empty_frames(input_csv_address, output_csv_address,
                        input_avi_address, output_avi_address,
                        search_cols=None):
    """
    Parameters:
        input_csv_address: The address of the csv we're cleaning
        output_csv_address: The address of the csv we're outputting cleaned data
        input_avi_address: The address of the avi we're cleaning
        output_avi_address: The address of the avi we're outputting cleaned data
        search_cols: Columns to search if they're empty. If none, search all.
    Description:
        Removes empty frames from the avi and csv and creates new files with frames removed
    Returns:
        True to indicate successful run
    """

    # Process the csv and save good frame ids
    ids = _remove_empty_frames_csv(input_csv_address, output_csv_address, search_cols)
    # Process the avi
    _remove_empty_frames_video(input_avi_address, output_avi_address, ids)
    return True