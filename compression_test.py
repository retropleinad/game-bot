import json

from model import VideoParser
from model import shuffle_batches


def datatype_test():
    batch_size = 3
    json_save_data = json.load(open('D:/Python Projects/gameBot/saves/minecraft.json', 'r'))

    num_batches = int(json_save_data['processed_total_frames'] // batch_size)

    # Shuffle batch numbers for randomness in order and what we feed into train/test generators
    batches = shuffle_batches(train_test_split=.7,
                              num_batches=num_batches,
                              shuffle=True)

    train_generator = VideoParser(json_save_data['processed_csv_address'],
                                  json_save_data['recorded_avi_address'],
                                  json_save_data['processed_total_frames'],
                                  y_labels=['w_press', 'w_release'],
                                  batch_size=batch_size,
                                  batches=batches['train'])

    # Grab and look at data
    x = train_generator[5]
    sample = x[0]
    frame = sample[2]
    # row = frame[200]
    num_bytes = frame.itemsize * frame.size
    print('Default frame setup')
    print('Number of mb consumed by 1 frame: ', num_bytes * 0.000001)
    print('================')

    # I believe we're already using uint8 but try to convert to uint8 anyways
    frame = frame.astype('uint8')
    num_bytes = frame.itemsize * frame.size
    print('Converted to uint8')
    print('Number of mb consumed by 1 frame: ', num_bytes * 0.000001)
    print('================')

    # Convert to float16
    frame = frame.astype('float16')
    num_bytes = frame.itemsize * frame.size
    print('Converted to float16')
    print('Number of mb consumed by 1 frame: ', num_bytes * 0.000001)
    print('================')

    # Divide by 255 and turn 0 into something a little bit more


datatype_test()
# https://github.com/tensorflow/tensorflow/issues/35030