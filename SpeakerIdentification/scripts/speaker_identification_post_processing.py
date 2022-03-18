import math
import os
import wave
from datetime import datetime
from datetime import timedelta

import librosa
import numpy as np
import noisereduce as nr
import scipy.io.wavfile as wav
import soundfile as sf
import tensorflow as tf
import webrtcvad
from pydub import AudioSegment
from pydub import effects
from python_speech_features import mfcc

import speaker_identification as si
import speaker_time_distribution as std

framerate = 16000  # sampling rate
num_samples = 2000  # sampling points for each chunk
channels = 1  # channels
sampwidth = 2  # sample width 2bytes
duration = 2.56  # recording voice for each 2.56 seconds
vad = webrtcvad.Vad(3)
np.random.seed(1)
Root_Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
NOISE_PATH = os.path.join(Root_Dir, 'experiment/Ambient_Noise.wav')


def delta(feat, N):
    NUMFRAMES = len(feat)
    # print("num frames: %d", NUMFRAMES)
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    # pad the row
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    for t in range(NUMFRAMES):
        # use partial finite difference to approximate the derivative
        delta_feat[t] = np.dot(np.arange(-N, N + 1), padded[t: t + 2 * N + 1]) / denominator
    return delta_feat


def trim_audio(source_path, target_path, start, end, format=None):
    """
    slice the audio segement from start to end
    :param start: start time / second
    :param end: stop time / second
    """
    sound = AudioSegment.from_file(source_path, format)
    sr = sound.frame_rate
    samples = sound.get_array_of_samples()
    shorter = sound._spawn(samples[int(start * sr):int(end * sr)])
    shorter.export(target_path, format="wav")


def segmentation(src_dir, dst_dir, win_time_stride, step_time):
    """
        :param src_dir: directory of the clips to cut
        :param dst_dir: directory to store the new cut clips
        :param win_time_stride: the time duration of the new cut clip
        :param step_time: difference time between two cut clip, overlap_time = win_time - step_time
        :return:
        """
    files = os.listdir(src_dir)
    files = [src_dir + "\\" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        filename = files[i]
        f = wave.open(filename, 'rb')

        params = f.getparams()
        # Get digital parameters of audio file, channels number, sampling width(bits per second), sampling rate,
        # sampling frames
        nchannels, _, _, nframes = params[:4]
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
            f.setnchannels(channels)
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
    if not target_path:
        target_path = source_path[:-4] + '.wav'

    """Normalize the audio volume dynamically to 1 as the the peak"""
    (y, sr) = librosa.load(source_path)
    max_peak = np.max(np.abs(y))

    ratio = 1 / max_peak

    y = y * ratio

    print('Previous peak: ', max_peak, 'Now peak: ', np.max(np.abs(y)))

    sf.write(target_path, y, sr, format='wav')

    "Equalize all parts of audio to its peak"
    # _sound = AudioSegment.from_file(target_path, "wav")
    # sound = effects.normalize(_sound)

    sound = AudioSegment.from_file(target_path, "wav")

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


def post_analysing():
    """Require the User update the regsitered speaker in /experiment/data/, later will generate the speaker_id_dict.json from data but not the existed one"""
    files = os.listdir(Root_Dir + '/experiment/corpus/')
    speaker_id_dict = {}

    for i in range(len(files)):
        speaker_id_dict[str(i)] = files[i][:-4]

    # print(speaker_id_dict)

    subdirectories = os.listdir(Root_Dir + '/experiment/recordings/post-time/segments/')

    for directory_name in subdirectories:
        "load model"
        model_path = Root_Dir + '/experiment/model'
        model = tf.keras.models.load_model(model_path)

        time = datetime.today()
        segment_no = -1
        segments_directory_path = Root_Dir + '/experiment/recordings/post-time/segments/' + directory_name
        whole_wav_path = Root_Dir + '/experiment/recordings/post-time/standardized/' + directory_name + '.wav'
        log_path = Root_Dir + '/experiment/logs/' + directory_name + '.txt'

        if os.path.exists(log_path):
            os.remove(log_path)
        else:
            print('file not exist')

        segments_wavs_path_lst = [segments_directory_path + '/' + file for file in os.listdir(segments_directory_path)]
        segments_wavs_path_lst.sort(key=lambda x: int(x.split('/')[-1].split('_')[-3]))
        # print(segments_wavs_path_lst)

        """find out the silent segments"""
        silent_index = []
        for wav_path in segments_wavs_path_lst:
            segment_no += 1
            data, sr = read_wave_file(wav_path)
            # print('sample rate ', sr, 'length', len(data))

            """check whether it is silent"""
            # since record_on_pc also import speaker_identification, couldn't import on top of file
            from record_on_pc import frame_generator, vad_collector

            frames = frame_generator(30, data, sr)
            frames = list(frames)
            segments = vad_collector(sr, 30, 300, vad, frames)
            wf = wave.open(wav_path, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)

            for i, segment in enumerate(segments):
                wf.writeframes(segment)
            wf.close()

            (rate, sig) = wav.read(wav_path)

            # print('after silence remove, sample rate', rate, 'length', len(sig))

            if len(sig) < 4000:
                silent_index.append(segment_no)

        test_x = []

        (rate, sig) = wav.read(whole_wav_path)
        mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, nfft=512)

        d_mfcc_feat = delta(mfcc_feat, 2)
        dd_mfcc_feat = delta(d_mfcc_feat, 2)

        finalfeature = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
        length = finalfeature.shape[0]
        segments_count = math.ceil(length / 256)
        finalfeature = np.concatenate((finalfeature, np.zeros((segments_count * 256 - length, 39))), axis=0)

        for i in range(segments_count):
            test_x.append(finalfeature[i * 256:(i + 1) * 256, :])

        test_x = np.asarray(test_x)

        "predict on test x"
        results = model.predict(test_x)
        segments = results.shape[0]

        for i in range(segments):
            if i in silent_index:
                speaker = 'silent'
                with open(log_path, 'a') as f:
                    time = time + timedelta(seconds=2.56)
                    if i == 0:
                        f.write('segment' + '\t' + 'speaker' + '\t' + 'timestamp')
                        f.write('\n')
                        # send_fruit_io(str(speaker_id_dict[key]), str(datetime.utcnow().isoformat()))

                    f.write(str(i) + '\t' + str(speaker) + '\t' + str(time))
                    f.write('\n')
                continue

            prob = results[i]

            # set a threshold for recognizing speaker
            # if np.amax(prob, axis=0) > 0.9:
            #     key = str(np.argmax(prob, axis=0))
            #     speaker = speaker_id_dict[key]
            # else:
            #     speaker = 'Unkown'

            key = str(np.argmax(prob, axis=0))
            speaker = speaker_id_dict[key]

            print('[RESULT] Prediction for segment', i, ': ', 'probability: ', prob, 'speaker: ', speaker)

            with open(log_path, 'a') as f:
                time = time + timedelta(seconds=2.56)
                if i == 0:
                    f.write('segment' + '\t' + 'speaker' + '\t' + 'timestamp')
                    f.write('\n')

                    # send_fruit_io(str(speaker_id_dict[key]), str(datetime.utcnow().isoformat()))

                f.write(str(i) + '\t' + str(speaker) + '\t' + str(time))
                f.write('\n')


if __name__ == "__main__":
    '''trim regsitry audio to one minute'''
    # trim_audio('D:\MMLA_Audio\SpeakerIdentification\experiment\corpus/eric.m4a', 'D:\MMLA_Audio\SpeakerIdentification\experiment\corpus/eric.wav', 10, 70)

    '''standardize the registered speakers' corpus'''
    files_path = []
    for (dirpath, dirnames, filenames) in os.walk(Root_Dir + '/experiment/corpus/'):
        for filename in filenames:
            filename_path = os.sep.join([dirpath, filename])
            files_path.append(filename_path)

    for i, onewav in enumerate(files_path):
        standardize_audio(onewav, dbfs=0, noise_reduced=0, silence_remove=True)
        # standardize_audio(onewav, noise_reduced=0, silence_remove=True)

    '''training model for the registered speakers'''
    acc = si.transfer_learning_on_experiment()

    '''standarize the conversation file'''
    conversation_list = os.listdir(Root_Dir + '/experiment/recordings/post-time/whole/')
    for audio_file_name in conversation_list:
        src_audio_path = os.path.join(Root_Dir + '/experiment/recordings/post-time/whole/', audio_file_name)
        dst_audio_path = os.path.join(Root_Dir + '/experiment/recordings/post-time/standardized/', audio_file_name[:-4] + '.wav')

        if audio_file_name.startswith('zoom'):
            print('[INFO]Processing zoom conversation file.')
            standardize_audio(src_audio_path, dst_audio_path, dbfs=0, noise_reduced=0, silence_remove=False)
        elif audio_file_name.startswith('audio'):
            print('[INFO]Processing recorded conversation file.')
            standardize_audio(src_audio_path, dst_audio_path, dbfs=0, noise_reduced=3, silence_remove=False)
        # standardize_audio(src_audio_path, dst_audio_path, noise_reduced=0, silence_remove=False)

    '''segment the converstaion file'''
    src_dir = Root_Dir + '/experiment/recordings/post-time/standardized/'
    dst_dir = Root_Dir + '/experiment/recordings/post-time/segments/'
    segmentation(src_dir, dst_dir, 2.56, 2.56)

    '''start analysing'''
    post_analysing()
    std.visualization()
