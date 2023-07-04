import json

from recording import Recorder
from preprocessing import EmptyFrameRemover

from model import KeyModel

minecraft_json = 'D:/Python Projects/gameBot/saves/minecraft.json'
recorded_csv = 'D:/Python Projects/gameBot/recording output/minecraft_gameplay.csv'
recorded_avi = 'D:/Python Projects/gameBot/recording output/minecraft_gameplay.avi'
processed_csv = 'D:/Python Projects/gameBot/processed output/minecraft_gameplay.csv'
processed_avi = 'D:/Python Projects/gameBot/processed output/minecraft_gameplay.avi'

minecraft_all_keys = ('1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'space', 'w', 'a', 's', 'd', 'shift',
                      'lmouse', 'rmouse')

basic_keys = ('w', 'a', 's', 'd', 'lmouse', 'rmouse')


def generate_column_names(keys):
    cols = []
    for key in keys:
        cols.append(key + '_press')
        cols.append(key + '_release')
    return cols


def build_dataset(record_seconds):
    recorder = Recorder(recorded_csv, recorded_avi, minecraft_all_keys)
    recorder.run(minecraft_json, record_seconds)
    recorder.quit()

    efr = EmptyFrameRemover(minecraft_json)
    # Clean up json read/write in EmptyFrameRemover
    efr.remove_empty_frames(processed_csv, processed_avi)


def build_model():
    save_data = json.load(open(minecraft_json))

    km = KeyModel(input_shape=[876, 1616, 3],
                  initial_learn_rate=.004,
                  epochs=10,
                  batch_size=3,
                  keys=generate_column_names(basic_keys),
                  mouse=True)

    km.build_model()


build_dataset(10)