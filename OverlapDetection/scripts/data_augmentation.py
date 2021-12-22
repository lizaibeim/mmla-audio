import os.path
import random

import numpy as np
import pandas as pd
from pydub import AudioSegment

from overlap_features_generator import OverlapFeaturesGenerator

'''
This is for mannually synthesizing the overlapped segments
'''

Features = './timit/features'
DATASET = './timit/data'
LABELS = "./timit/labels.csv"
AUGMENTED_LABELS = "./timit/augmented_labels.csv"


def generate_overlap_segment(wav_list, overlap_path):
    """
    :param wav_list: path list of wav files to synthesize the overlapped segment
    :param overlap_path: output path to store the overlapped segment
    """
    canvas = AudioSegment.from_file(file=wav_list[0])
    time_durations = 1.5 if canvas.duration_seconds > 1.5 else canvas.duration_seconds
    canvas = canvas[:time_durations * 1000]

    for i in range(1, len(wav_list)):
        sound = AudioSegment.from_file(file=wav_list[i])
        index = random.randrange(int(time_durations * 10) - 2) * 100
        canvas = canvas.overlay(sound, position=index)

    canvas.export(out_f=overlap_path, format="wav")


def run_overlaps_generator():
    """
    :return: generate the same size of overlapped segments compared to original data set (TIMIT)
    """
    labels = pd.read_csv(LABELS).loc[:, ['speaker_id', 'path_from_data_dir_windows']]
    rows = []
    for i in range(0, 6300):
        row = []
        if i < 3150:
            speaker_num = 2
        elif 3150 <= i < 5040:
            speaker_num = 3
        elif 5040 <= i < 5985:
            speaker_num = 4
        else:
            speaker_num = 5

        wav_list = []
        speakers_id = ''
        for j in range(0, speaker_num):
            selected_speaker = random.randrange(0, 630)
            selected_sample = random.randrange(0, 10)
            index = selected_speaker * 10 + selected_sample
            wav_path = os.path.join(DATASET, labels['path_from_data_dir_windows'].values[index])
            wav_list.append(wav_path)
            speakers_id = labels['path_from_data_dir_windows'].values[index].split('\\\\')[
                2] if j == 0 else speakers_id + '_' + labels['path_from_data_dir_windows'].values[index].split('\\\\')[
                2]

        path_from_data_dir_winodws = 'OVERLAP\\\\' + 'Overlap' + str(i) + '.WAV.wav'
        row.append(speakers_id)
        row.append(path_from_data_dir_winodws)
        rows.append(row)
        overlap_path = os.path.join(DATASET, path_from_data_dir_winodws)
        generate_overlap_segment(wav_list, overlap_path)

    df = pd.DataFrame(np.array(rows), columns=['speaker_id', 'path_from_data_dir_windows'])
    frames = [labels, df]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(AUGMENTED_LABELS)


def run_overlap_features_generator():
    """
    :return: generate the overlapped features (ZCR) of all data
    """
    ofg = OverlapFeaturesGenerator(wl=25, hl=10)
    augmented_labels = pd.read_csv(AUGMENTED_LABELS).loc[:, ['speaker_id', 'path_from_data_dir_windows']]
    files_name = []
    overlap_degrees = []

    for i in range(len(augmented_labels['path_from_data_dir_windows'])):
        file_name = str.split(augmented_labels['path_from_data_dir_windows'].values[i], '\\')[-1][:-8] + '_' + \
                    augmented_labels['speaker_id'].values[i] + '.png'
        overlap_degree = 2 if str.split(augmented_labels['path_from_data_dir_windows'].values[i], '\\')[-1][
                              :7] == 'Overlap' else 1
        files_name.append(file_name)
        overlap_degrees.append(overlap_degree)
        wav_path = os.path.join(DATASET, augmented_labels['path_from_data_dir_windows'].values[i])
        ofg.generate_zcr_image(wav_path, './timit/features/mel_spectrum_zcr/', file_name)
    augmented_labels['image_file_name'] = files_name
    augmented_labels['overlap_degree'] = overlap_degrees
    augmented_labels.to_csv('./timit/augmented_overlap_labels.csv')


if __name__ == "__main__":
    run_overlaps_generator()
    run_overlap_features_generator()
