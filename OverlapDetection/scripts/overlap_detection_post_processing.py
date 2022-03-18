import genericpath
import os
import wave

from pydub import AudioSegment
from datetime import datetime, timedelta
from overlap_features_generator import OverlapFeaturesGenerator
import overlap_degree_distribution as odd
import tensorflow as tf
import numpy as np
import noisereduce as nr
import soundfile as sf
import librosa
import webrtcvad
from tensorflow.keras import backend as K

vad = webrtcvad.Vad(3)
Root_Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
overlap_degree_dict = {'0': 'non-overlapped', '1': 'overlapped'}
NOISE_PATH = os.path.join(Root_Dir, 'experiment/Ambient_Noise.wav')


def segmentation(src_dir, dst_dir, win_time_stride, step_time):
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


def read_wave_file(filepath):
    wf = wave.open(filepath, 'rb')
    num_channels = wf.getnchannels()
    assert num_channels == 1
    sample_width = wf.getsampwidth()
    assert sample_width == 2
    sample_rate = wf.getframerate()
    # print(sample_rate)
    assert sample_rate in (8000, 16000, 32000, 48000)
    data = wf.readframes(wf.getnframes())
    return data, sample_rate


def standardize_audio(source_path, target_path=None, format=None, dbfs=None, channels=1, sampwidth=2, sample_rate=16000,
                      noise_reduced=0, silence_remove=False):
    """normalize the audio volume"""
    (y, sr) = librosa.load(source_path)
    max_peak = np.max(np.abs(y))

    ratio = 1 / max_peak

    y = y * ratio

    print('Previous peak: ', max_peak, 'Now peak: ', np.max(np.abs(y)))

    if not target_path:
        target_path = source_path[:-4] + '.wav'

    sf.write(target_path, y, sr, format='wav')

    sound = AudioSegment.from_file(source_path, format)

    if sample_rate:
        sound = sound.set_frame_rate(sample_rate)
    if dbfs:
        change_dBFS = dbfs - sound.dBFS
        sound = sound.apply_gain(change_dBFS)
    sound.export(target_path, format='wav')

    noise, _ = librosa.load(NOISE_PATH, sr=None)
    while noise_reduced > 0:
        noise_reduced -= 1
        y, _ = librosa.load(target_path, sr=None)
        noise_reduced_wav = nr.reduce_noise(y_noise=noise, y=y, sr=sample_rate, stationary=True)
        sf.write(target_path, noise_reduced_wav, sample_rate)

    if silence_remove:
        # since record_on_pc also import speaker_identification, couldn't import on top of file
        from record_on_pc import frame_generator, vad_collector
        audio_data, sr = read_wave_file(target_path)
        frames = frame_generator(30, audio_data, sr)
        frames = list(frames)
        segments = vad_collector(sr, 30, 300, vad, frames)

        wf = wave.open(target_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        for i, segments in enumerate(segments):
            wf.writeframes(segments)
        wf.close()


def post_anlysing():
    ofg = OverlapFeaturesGenerator(wl=25, hl=10)
    model_path = os.path.join(Root_Dir, 'timit/models/timit2.0')
    model = tf.keras.models.load_model(model_path)

    """standardize the conversation files under the experiment/recordings/post-time/whole folder"""
    conversations_path = []
    standardized_conversations_path = []
    logs_path = []
    segments_dirs = []
    features_dirs = []

    for (dirpath, dirnames, filenames) in os.walk(os.path.join(Root_Dir, 'experiment/recordings/post-time/whole')):
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

    # print(segments_dirs)
    # print(logs_path)
    # print(features_dirs)

    for i, onewav in enumerate(conversations_path):
        if onewav.split('\\')[-1].startswith('zoom'):
            print("[INFO]Processing zoom audio file .")
            standardize_audio(onewav, standardized_conversations_path[i], dbfs=0, noise_reduced=0, silence_remove=False)
        elif onewav.split('\\')[-1].startswith('audio'):
            print("[INFO]Processing recorded audio file.")
            standardize_audio(onewav, standardized_conversations_path[i], dbfs=0, noise_reduced=3, silence_remove=False)

    """segment the standradized conversation files into several segments which is 1.5 seconds for each"""
    src_dir = Root_Dir + '/experiment/recordings/post-time/standardized/'
    dst_dir = Root_Dir + '/experiment/recordings/post-time/segments/'
    segmentation(src_dir, dst_dir, 1.5, 1.5)

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
            print('[INFO] Prediction for segment ', count,  ': probability: ', prob, ' overlapped degree: ',
                  overlap_degree_dict[key])

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
    """
    Before running the scripts, you need to put the conversation audio files under the experiment/recordings/post-time/whole/ folder
    """
    post_anlysing()
    odd.visualization()
