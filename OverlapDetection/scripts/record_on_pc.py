import collections
import json
import os
import sys
import time
import wave
from datetime import datetime

import cv2
import librosa
import noisereduce as nr
import numpy as np
import requests
import scipy.io.wavfile as wav
import soundfile as sf
import tensorflow as tf
import webrtcvad
from pyaudio import PyAudio, paInt16
from tensorflow.keras import backend as K
from skimage.metrics._structural_similarity import structural_similarity

from overlap_features_generator import OverlapFeaturesGenerator

# Get the root directory of Overlap detection
Root_Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
NOISE_PATH = os.path.join(Root_Dir, 'experiment/Ambient_Noise.wav')

framerate = 16000  # sampling rate
num_samples = 2000  # sampling points for each chunk
channels = 1  # channels
sampwidth = 2  # sample width 2bytes
dur = 2.56  # duration of one segment
vad = webrtcvad.Vad(3)
overlap_degree_dict = {'0': 'non-overlapped', '1': 'overlapped'}
url = ''
io_key = ''


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def recording(filename, duration, noise_reduced=False):
    filepath = os.path.join(Root_Dir, 'experiment/data/') + str(filename) + '.wav'

    pa = PyAudio()
    stream = pa.open(format=paInt16, channels=channels,
                     rate=framerate, input=True, frames_per_buffer=num_samples)
    my_buf = []
    t = time.time()
    if not noise_reduced:
        print('[INFO] Noise Recording, Please keep quiet...')
    else:
        print('[INFO] Recording...')

    while time.time() < t + duration:  # set the recording duration
        # loop of read，read 2000 frames each iteration (0.175s)
        string_audio_data = stream.read(num_samples)
        my_buf.append(string_audio_data)
    print('[INFO] Recording Done.')

    if not noise_reduced:
        save_wave_file(NOISE_PATH, my_buf, noise_reduce=False)

    else:
        save_wave_file(filepath, my_buf, noise_reduce=True)

    stream.close()


def compare_images(img1_path, img2_path):
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    s = structural_similarity(image1, image2, multichannel=True)
    # print(s)
    if s < 0.3:
        return True
    return False


def run_overlap_detection(silence_removed=False):
    ofg = OverlapFeaturesGenerator(wl=25, hl=10)
    quit_flag = False
    model_path = os.path.join(Root_Dir, 'timit/models/timit2.0')
    model = tf.keras.models.load_model(model_path)
    # with open('overlap_degree_dict.json') as f:
    #     overlap_degree_dict = json.load(f)
    # print(overlap_degree_dict)

    if not os.path.exists(os.path.join(Root_Dir, 'experiment/logs/')):
        os.mkdir(os.path.join(Root_Dir, 'experiment/logs/'))
    if not os.path.exists(os.path.join(Root_Dir, 'experiment/recordings/')):
        os.mkdir(os.path.join(Root_Dir, 'experiment/recordings/'))

    print('[INFO] Model loaded: start predicting...')
    pa = PyAudio()

    count = 0
    log_path = Root_Dir + '/experiment/logs/' + str(datetime.now()).replace(' ', '-').replace(':', '-')[:-7] + '.txt'
    p_dir = Root_Dir + '/experiment/recordings/' + str(datetime.now()).replace(' ', '-').replace(':', '-')[
                                                   :-7]
    c_png_dir = Root_Dir + '/experiment/recordings/' + str(datetime.now()).replace(' ', '-').replace(':', '-')[
                                                       :-7] + '/png/'
    c_wav_dir = Root_Dir + '/experiment/recordings/' + str(datetime.now()).replace(' ', '-').replace(':', '-')[
                                                       :-7] + '/wav/'
    os.mkdir(p_dir)
    os.mkdir(c_png_dir)
    os.mkdir(c_wav_dir)

    try:
        while not quit_flag:
            stream = pa.open(format=paInt16, channels=channels, rate=framerate, input=True,
                             frames_per_buffer=num_samples)
            print('[INFO] One iteration...')
            frames = []
            t = time.time()
            while time.time() < t + dur:  # set the predicting duration

                # loop of read，read 2000 frames each iteration (0.175s)
                string_audio_data = stream.read(num_samples)
                frames.append(string_audio_data)

            count += 1
            out_name_wav = str(count) + '.wav'
            out_name_png = str(count) + '.png'
            # filepath = c_wav_dir + out_name_wav
            noise_reduced_filepath = c_wav_dir + out_name_wav

            # save_wave_file(filepath, frames, noise_reduce=False, silence_remove=silence_removed)
            save_wave_file(noise_reduced_filepath, frames, noise_reduce=True, silence_remove=silence_removed)

            # features_image_path = filepath[:-4] + '.png'
            features_image_path2 = c_png_dir + out_name_png

            # ofg.generate_zcr_image(filepath, c_dir, out_name_png)
            ofg.generate_zcr_image(noise_reduced_filepath, c_png_dir, out_name_png)

            (rate, sig) = wav.read(noise_reduced_filepath)
            if len(sig) < 4000:
                print("[INFO] Prediction result for last 1.5 second: silent. \n")
                with open(log_path, 'a') as f:
                    if count == 1:
                        f.write('segment' + '\t' + 'overlapped degree' + '\t' + 'timestamp')
                        f.write('\n')

                    # send_fruit_io('silent', str(datetime.utcnow().isoformat()))
                    f.write(str(count) + '\t' + 'silent' + '\t' + str(datetime.today()))
                    f.write('\n')

                stream.close()
                continue

            image = tf.io.read_file(features_image_path2)
            features_data = [tf.image.decode_png(image, 3)]
            _input = tf.stack(features_data, axis=0).numpy().astype('float32')
            prob = model.predict(_input)
            key = str(np.argmax(prob, axis=1)[0])
            print('[INFO] Predcition for the last 1.5 seconds: ', 'probability: ', prob, 'overlap degree: ',
                  overlap_degree_dict[key])

            with open(log_path, 'a') as f:
                if count == 1:
                    f.write('segment' + '\t' + 'overlapped degree' + '\t' + 'timestamp')
                    f.write('\n')

                # send_fruit_io(str(overlap_degree_dict[key]), str(datetime.utcnow().isoformat()))
                f.write(str(count) + '\t' + str(overlap_degree_dict[key]) + '\t' + str(datetime.today()))
                f.write('\n')

            stream.close()

    except KeyboardInterrupt:
        print("[INFO] Exit the program now...")
        sys.exit(0)


def send_fruit_io(value, time):
    time = time[:-7] + 'Z'
    data = {'value': value, 'created_at': time}
    json_dump = json.dumps(data)

    requests.post(url, headers={'X-AIO-Key': io_key, 'Content-type': 'application/json'}, data=json_dump, )


def read_wave_file(filepath):
    wf = wave.open(filepath, 'rb')
    num_channels = wf.getnchannels()
    assert num_channels == 1
    sample_width = wf.getsampwidth()
    assert sample_width == 2
    sample_rate = wf.getframerate()
    assert sample_rate in (8000, 16000, 32000, 48000)
    data = wf.readframes(wf.getnframes())
    return data, sample_rate


def save_wave_file(filepath, data, noise_reduce=False, silence_remove=False):
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b''.join(data))
    wf.close()

    if noise_reduce:
        noise, sr = librosa.load(NOISE_PATH, sr=None)
        y, sr = librosa.load(filepath, sr=None)
        noise_reduced_wav = nr.reduce_noise(y_noise=noise, y=y, sr=sr, stationary=True)
        sf.write(filepath, noise_reduced_wav, 16000)

    if silence_remove:
        audio, sample_rate = read_wave_file(filepath)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, 300, vad, frames)

        wf = wave.open(filepath, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        for i, segments in enumerate(segments):
            wf.writeframes(segments)
        wf.close()


def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
    #     sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


# recording
def main():
    """pivot ambient noise, set it as ten seconds duration"""

    val = input('[INFO] Please be quiet for 10 seconds and the ambient noise will be collected, press [y] to start.')
    while val != 'y':
        val = input(
            '[INFO] Please be quiet for 10 seconds and the ambient noise will be collected, press [y] to start.')
    recording('Ambient_Noise', 10, False)

    """Run the speaker identification module"""
    val4 = input('[INFO] Ready for overlap detection, press [s] to start.')
    while val4 != 's':
        val4 = input('[INFO] Ready for overlap detection, press [s] to start.')
    run_overlap_detection(silence_removed=True)


if __name__ == '__main__':
    main()
