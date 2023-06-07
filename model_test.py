import pandas as pd

from model import VideoParser


TEST_FILE = 'D:/Python Projects/gameBot/recording output/gameplay'
BATCH_SIZE = 12


def video_parser_test():
    # Create a video parser object
    parser = VideoParser(TEST_FILE)

    # Test parsing the csv
    for i in range(0, 2):
        df = parser.parse_csv(BATCH_SIZE, i)
        a = True

    # Test parsing the video
    i = 12
    while i < 36:
        df = parser.parse_video(BATCH_SIZE, i)
        i += BATCH_SIZE

    # Test removing columns
    df = parser.parse_csv(BATCH_SIZE)
    parser.remove_empty_frames(df, 'df')

    # Test parsing dataset
    i = 0
    while i < parser.get_total_frames():
        parser.parse_dataset(BATCH_SIZE, i)
        i += BATCH_SIZE
        print("Iteration {0}: ".format(int(i/12)))


video_parser_test()