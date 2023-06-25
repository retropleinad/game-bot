import json

from recording import Recorder
from preprocessing import remove_empty_frames

from model import KeyModel

minecraft_json = 'D:/Python Projects/gameBot/saves/minecraft.json'
minecraft_all_keys = ('1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'space', 'w', 'a', 's', 'd', 'shift',
                      'lmouse', 'rmouse')

basic_keys = ('w', 'a', 's', 'd', 'lmouse')


def generate_column_names(keys):
    cols = []
    for key in keys:
        cols.append(key + '_press')
        cols.append(key + '_release')
    return cols


def build_dataset(record_seconds):
    save_data = json.load(open(minecraft_json))

    recorder = Recorder(save_data['recorded_csv_address'], minecraft_all_keys)
    recorder.run(record_seconds=record_seconds)
    recorder.quit()

    remove_empty_frames(input_csv_address=save_data['recorded_csv_address'],
                        output_csv_address=save_data['processed_csv_address'],
                        input_avi_address=save_data['recorded_avi_address'],
                        output_avi_address=save_data['processed_avi_address'],
                        json_save_address=minecraft_json,
                        search_cols=generate_column_names(basic_keys))


def build_model():
    save_data = json.load(open(minecraft_json))

    km = KeyModel(input_shape=[876, 1616, 3],
                  initial_learn_rate=.004,
                  epochs=10,
                  batch_size=3,
                  keys=generate_column_names(basic_keys),
                  mouse=True)

    km.build_model()