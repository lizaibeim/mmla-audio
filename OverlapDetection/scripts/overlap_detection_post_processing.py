import genericpath
import os
import wave

from pydub import AudioSegment
from datetime import datetime, timedelta
from overlap_features_generator import OverlapFeaturesGenerator
import tensorflow as tf
import numpy as np
import webrtcvad
from tensorflow.keras import backend as K

vad = webrtcvad.Vad(3)
Root_Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
overlap_degree_dict = {'0': 'non-overlapped', '1': 'overlapped'}


def split(src_dir, dst_dir, win_time_stride, step_time):
    """
    :param src_dir: directory of the clips to cut
    :param dst_dir: directory to store the new cut clips
    :param win_time_stride: the time duration of the new cut clip
    :param step_time: interval time between two cut clip, overlap_time = win_time - step_time
    :return:
    """
    files = os.listdir(src_dir)
    files = [src_dir + "\\" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        filename = files[i]
        f = wave.open(filename, 'rb')

        params = f.getparams()
        # Get digital parameters of audio file, channe , sampling width(bits per second), sampling rate,
        # sampling frames
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        f.close()

        wave_data = np.frombuffer(str_data, dtype=np.short)

        if nchannels > 1:
            wave_data.shape = -1, 2
            wave_data = wave_data.T
            temp_data = wave_data.T
        else:
            wave_data = wave_data.T
            temp_data = wave_data.T

        win_num_frames = int(framerate * win_time_stride)
        step_num_frames = int(framerate * step_time)

        print("window frames: ", win_num_frames, "step frames: ", step_num_frames)
        cut_num = int(((nframes - win_num_frames) / step_num_frames) + 1)  # how many segments for one wav file
        step_total_num_frames = 0

        for j in range(cut_num):
            file_abs_path = os.path.splitext(os.path.split(filename)[-1])[0]
            file_save_path = os.path.join(dst_dir, file_abs_path)

            if not os.path.exists(file_save_path):
                os.makedirs(file_save_path)

            out_file = os.path.join(file_save_path,
                                    os.path.splitext(os.path.split(filename)[-1])[0] + '_%d_%s_split.wav' % (
                                        j, framerate))
            start = step_num_frames * j
            end = step_num_frames * j + win_num_frames
            temp_data_temp = temp_data[start:end]
            step_total_num_frames = (j + 1) * step_num_frames
            temp_data_temp.shape = 1, -1
            temp_data_temp = temp_data_temp.astype(np.short)
            f = wave.open(out_file, 'wb')
            f.setnchannels(nchannels)
            f.setsampwidth(sampwidth)
            f.setframerate(framerate)
            f.writeframes(temp_data_temp.tostring())
            f.close()

        print("Total number of frames :", nframes, " Extract frames: ", step_total_num_frames)


def standardize_audio(source_path, target_path=None, format=None, dbfs=None, channels=1, sampwidth=2, sr=16000,
                      silence_remove=False):
    sound = AudioSegment.from_file(source_path, format)

    print(sound.frame_rate)

    if sr:
        sound = sound.set_frame_rate(sr)
    if dbfs:
        change_dBFS = dbfs - sound.dBFS
        sound = sound.apply_gain(change_dBFS)

    if not target_path:
        target_path = source_path[:-4] + '.wav'
        print(target_path)

    if silence_remove:
        # since record_on_pc also import speaker_identification, couldn't import on top of file
        from record_on_pc import frame_generator, vad_collector
        audio_data = sound.raw_data
        frames = frame_generator(30, audio_data, sound.frame_rate)
        frames = list(frames)
        segments = vad_collector(sound.frame_rate, 30, 300, vad, frames)

        wf = wave.open(target_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        for i, segments in enumerate(segments):
            wf.writeframes(segments)

        wf.close()

    else:
        sound.export(target_path, format='wav')


def post_anlysing():
    ofg = OverlapFeaturesGenerator(wl=25, hl=10)
    model_path = os.path.join(Root_Dir, 'timit/models/timit2.0')
    model = tf.keras.models.load_model(model_path)

    """standardize the conversation files under the experiment/recordings/post-time/original folder"""
    conversations_path = []
    standardized_conversations_path = []
    logs_path = []
    segments_dirs = []
    features_dirs = []

    for (dirpath, dirnames, filenames) in os.walk(os.path.join(Root_Dir, 'experiment/recordings/post-time/original')):
        for filename in filenames:
            filename_path = os.sep.join([dirpath, filename])
            conversations_path.append(filename_path)
            standardized_conversations_path.append(
                os.path.join(Root_Dir, 'experiment/recordings/post-time/standardized', filename)[:-4] + '.wav')
            logs_path.append(Root_Dir + '/experiment/logs/' + filename[:-4] + '.txt')
            segments_dirs.append(os.path.join(Root_Dir, 'experiment/recordings/post-time/segments', filename)[:-4])
            features_dirs.append(Root_Dir + '/experiment/recordings/post-time/features/' + filename[:-4] + '/')

    for i, feature_dir in enumerate(features_dirs):
        if not os.path.exists(feature_dir):
            os.mkdir(feature_dir)

    print(segments_dirs)
    print(logs_path)
    print(features_dirs)

    for i, onewav in enumerate(conversations_path):
        standardize_audio(onewav, standardized_conversations_path[i], dbfs=0, silence_remove=False)

    """split the standradized conversation files into several segments which is 1.5 seconds for each"""
    split('D:/MMLA_Audio/OverlapDetection/experiment/recordings/post-time/standardized',
          'D:/MMLA_Audio/OverlapDetection/experiment/recordings/post-time/segments', 1.5, 1.5)

    """features generation and prediction"""
    for i, segments_dir in enumerate(segments_dirs):
        _input_dict = {}
        count = 0
        time = datetime.today()
        for file in os.listdir(segments_dir):
            wav_file_path = os.path.join(segments_dir, file)
            out_name_png = str(count) + '.png'
            features_image_path = features_dirs[i] + out_name_png
            ofg.generate_zcr_image(wav_file_path, features_dirs[i], out_name_png)
            image = tf.io.read_file(features_image_path)
            features_data = [tf.image.decode_png(image, 3)]
            _input = tf.stack(features_data, axis=0).numpy().astype('float32')

            prob = model.predict(_input)
            key = str(np.argmax(prob, axis=1)[0])
            # print('[INFO] Predcition for the last 1.5 seconds: ', 'probability: ', prob, 'overlap degree: ',
            #       overlap_degree_dict[key])

            if count == 0:
                with open(logs_path[i], 'w') as f:
                    f.write('segment' + '\t' + 'overlapped degree' + '\t' + 'timestamp')
                    f.write('\n')
                    f.write(str(count) + '\t' + str(overlap_degree_dict[key]) + '\t' + str(time))
                    f.write('\n')

            else:
                with open(logs_path[i], 'a') as f:
                    time = time + timedelta(seconds=1.5)
                    f.write(str(count) + '\t' + str(overlap_degree_dict[key]) + '\t' + str(time))
                    f.write('\n')

            count += 1


if __name__ == "__main__":
    """path creation"""
    paths = [os.path.join(Root_Dir, 'experiment/recordings/'),
             os.path.join(Root_Dir, 'experiment/recordings/post-time/original/'),
             os.path.join(Root_Dir, 'experiment/recordings/post-time/features/'),
             os.path.join(Root_Dir, 'experiment/recordings/post-time/segments/'),
             os.path.join(Root_Dir, 'experiment/recordings/post-time/standardized/')]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    """
    Before running the scripts, you need to put the conversation audio files under the experiment/recordings/post-time/original/ folder
    """
    post_anlysing()
