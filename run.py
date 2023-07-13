import json
import os
import tensorflow as tf

from recording import Recorder
from preprocessing import EmptyFrameRemover
from predict import Predictor

from model import KeyModel

minecraft_json = 'D:/Python Projects/gameBot/saves/minecraft.json'
recorded_csv = 'D:/Python Projects/gameBot/recording output/minecraft_gameplay.csv'
recorded_avi = 'D:/Python Projects/gameBot/recording output/minecraft_gameplay.avi'
processed_csv = 'D:/Python Projects/gameBot/processed output/minecraft_gameplay.csv'
processed_avi = 'D:/Python Projects/gameBot/processed output/minecraft_gameplay.avi'

minecraft_all_keys = ('1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'space', 'w', 'a', 's', 'd', 'e', 'shift',
                      'lmouse', 'rmouse')

tree_farm_keys = ('1', '2', '3',
                  # 'w', 'a', 's', 'd',
                  # 'e', 'shift',
                  'lmouse', 'rmouse')

basic_keys = ('w', 'a', 's', 'd', 'lmouse', 'rmouse')
wasd = ('w', 'a', 's', 'd')


def generate_column_names(keys):
    cols = []
    for key in keys:
        cols.append(key + '_press')
        cols.append(key + '_release')
    return cols


def build_dataset(record_seconds):
    recorder = Recorder(recorded_csv, recorded_avi, tree_farm_keys)
    recorder.run(minecraft_json, record_seconds)
    recorder.quit()
    print('successfully recorded video')

    efr = EmptyFrameRemover(minecraft_json)
    # Clean up json read/write in EmptyFrameRemover
    efr.remove_empty_frames(processed_csv, processed_avi, .1)
    print('successfully removed empty frames')


def build_model():
    km = KeyModel(json_address=minecraft_json,
                  initial_learn_rate=.004,
                  epochs=20,
                  batch_size=48,
                  keys=generate_column_names(tree_farm_keys),
                  mouse=True)

    km.build_model()


def make_predictions():
    predictor = Predictor(model_address='D:/Python Projects/gameBot/models/tree_farm',
                          json_address=minecraft_json)
    predictor.run_predictions()


make_predictions()
# check tf.memory() after each frame/batch?

# ['w_press', 'w_release'] no mouse: 80% memory 90% cpu
# What kills memory is the cpu working too hard