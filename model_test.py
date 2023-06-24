from model import VideoParser
from model import KeyModel
from model import shuffle_batches
from predict import Predictor


INPUT_CSV = 'D:/Python Projects/gameBot/processed output/gameplay.csv'
INPUT_AVI = 'D:/Python Projects/gameBot/recording output/gameplay.avi'
BATCH_SIZE = 12


def video_parser_test():
    # Create a video parser object
    parser = VideoParser(INPUT_CSV, INPUT_AVI)

    # Test parsing the csv
    for i in range(0, 2):
        df = parser.__parse_csv(BATCH_SIZE, i)
        a = True

    # Test parsing the video
    i = 12
    while i < 36:
        df = parser.__parse_video(BATCH_SIZE, i)
        i += BATCH_SIZE

    # Test removing columns
    df = parser.__parse_csv(BATCH_SIZE)
    parser.remove_empty_frames(df, 'df')

    # Test parsing dataset
    i = 0
    while i < parser.get_total_frames():
        parser.__parse_dataset(BATCH_SIZE, i)
        i += BATCH_SIZE
        print("Iteration {0}: ".format(int(i/12)))


def data_generator_test():
    # Create the generator
    train_generator = VideoParser(INPUT_CSV, INPUT_AVI,
                                  'D:/Python Projects/gameBot/processed output/gameplay.avi',
                                  y_labels=('w_press', 'w_release'),
                                  mouse_x_max=1616, mouse_y_max=876)

    # Grab the length of the generator
    print('Number of batches: ', train_generator.__len__())

    # Look to see if we can grab the first few outputs of the generator
    for i in range(0, 5):
        x, y = train_generator[i]
        # print('Output set: ', i)
        # print(y)

    # Is each y different
    initial_x, initial_y = train_generator[0]
    for i in range(i, train_generator.__len__()):
        x, y = train_generator[i]
        print(len(y))


def key_model_test():
    km = KeyModel([876, 1616, 3], .004, 20, 3, keys=('w_press', 'w_release', 'a_press', 'a_release'), mouse=True)


def shuffle_batches_test():
    # Test simple batches first
    batches = shuffle_batches(.7, 10)

    # Are the lengths of both right?
    print('Train batch length: ', len(batches['train']))
    print('Test batch length: ', len(batches['test']))

    # Test not simple batches
    batches = shuffle_batches(.7, 10, simple=False)

    # Are the lengths correct?
    print('Train batch length: ', len(batches['train']))
    print('Test batch length: ', len(batches['test']))
    print('Train: ', batches['train'])
    print('Test: ', batches['test'])

    # Stress test
    batches = shuffle_batches(.7, 10000, simple=False)

    # Look at lengths
    print('Train batch length: ', len(batches['train']))
    print('Test batch length: ', len(batches['test']))


def predict_test():
    # Create predictor (be sure to have Minecraft open)
    predictor = Predictor('D:/Python Projects/gameBot/models/tree_farm')
    predictor.run_predictions()


data_generator_test()