import collections
import json
import os
import sys
import time
import wave
from datetime import datetime

import keyboard
import librosa
import noisereduce as nr
import numpy as np
import requests
import soundfile as sf
import tensorflow as tf
import webrtcvad
from pyaudio import PyAudio, paInt16
from tensorflow.keras import backend as K

import speaker_identification as vi

Root_Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
NOISE_PATH = os.path.join(Root_Dir, 'experiment/Ambient_Noise.wav')

framerate = 16000  # sampling rate
num_samples = 2000  # sampling points for each chunk
channels = 1  # channels
sampwidth = 2  # sample width 2bytes
duration = 2.56  # recording voice for each 2.56 seconds
vad = webrtcvad.Vad(3)

url = ''
io_key = ''


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def recording(filename, duration, noise_reduced=False, silence_removed=False):
    if not os.path.exists(Root_Dir + '/experiment/data/'):
        os.mkdir(Root_Dir + '/experiment/data/')
    filepath = Root_Dir + '/experiment/data/' + str(filename) + '.wav'

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
        save_wave_file(NOISE_PATH, my_buf, noise_reduced=False, silence_removed=silence_removed)

    else:
        save_wave_file(filepath, my_buf, noise_reduced=True, silence_removed=silence_removed)

    stream.close()


def run_speaker_identification(noise_reduced=False, silence_removed=False):
    quit_flag = False
    model_path = Root_Dir + '/experiment/model'
    model = tf.keras.models.load_model(model_path)
    with open(Root_Dir + '/experiment/speaker_id_dict.json') as f:
        speaker_id_dict = json.load(f)
    print(speaker_id_dict)

    print('[INFO] Model loaded: start predicting...')

    if not os.path.exists(Root_Dir + '/experiment/logs/'):
        os.mkdir(Root_Dir + '/experiment/logs/')
    if not os.path.exists(Root_Dir + '/experiment/recordings/real-time/'):
        os.mkdir(Root_Dir + '/experiment/recordings/real-time/')

    pa = PyAudio()

    count = 0
    log_path = Root_Dir + '/experiment/logs/' + str(datetime.now()).replace(' ', '-').replace(':', '-')[:-7] + '.txt'
    run_dir = Root_Dir + '/experiment/recordings/real-time/' + str(datetime.now()).replace(' ', '-').replace(':', '-')[:-7]
    os.mkdir(run_dir)

    try:
        while not quit_flag:

            stream = pa.open(format=paInt16, channels=channels,
                             rate=framerate, input=True, frames_per_buffer=num_samples)
            print('[INFO] One iteration...')
            frames = []
            t = time.time()
            while time.time() < t + duration:  # set the predicting duration
                # if keyboard.is_pressed('q'):
                #     quit_flag = True
                #     print("Quit now...")
                #     # break

                # loop of read，read 2000 frames each iteration (0.175s)
                string_audio_data = stream.read(num_samples)
                frames.append(string_audio_data)

            count += 1
            filepath = run_dir + '/' + str(count) + '.wav'

            save_wave_file(filepath, frames, noise_reduced=noise_reduced, silence_removed=silence_removed)
            x = vi.input_feature_gen(filepath)
            if x == 'silent':
                print('Predcition for the last 2 seconds: silent')

                with open(log_path, 'a') as f:
                    if count == 1:
                        f.write('segment' + '\t' + 'speaker' + '\t' + 'timestamp')
                        f.write('\n')

                    # send_fruit_io('silent', str(datetime.utcnow().isoformat()))
                    f.write(str(count) + '\t' + 'silent' + '\t' + str(datetime.today()))
                    f.write('\n')

                stream.close()
                continue

            prob = model.predict(x)
            key = str(np.argmax(prob, axis=1)[0])
            # key = str(np.argmax(model.predict(x), axis=1)[0])
            print('[RESULT] Predcition for the last 2 seconds: ', 'probability: ', prob, 'speaker: ',
                  speaker_id_dict[key])

            with open(log_path, 'a') as f:
                if count == 1:
                    f.write('segment' + '\t' + 'speaker' + '\t' + 'timestamp')
                    f.write('\n')

                # send_fruit_io(str(speaker_id_dict[key]), str(datetime.utcnow().isoformat()))
                f.write(str(count) + '\t' + str(speaker_id_dict[key]) + '\t' + str(datetime.today()))
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


def save_wave_file(filepath, data, noise_reduced=False, silence_removed=False):
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b''.join(data))
    wf.close()

    if noise_reduced:
        noise, sr = librosa.load(NOISE_PATH, sr=None)
        y, sr = librosa.load(filepath, sr=None)
        noise_reduced_wav = nr.reduce_noise(y_noise=noise, y=y, sr=sr, stationary=True)
        sf.write(filepath, noise_reduced_wav, 16000)

    if silence_removed:
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

    if os.path.exists(NOISE_PATH):
        val = input(
                '[INFO] It seems like you have already recorded the ambient noise, do you want to recalibrate it?, press [y] to record it again or [n] not to record.')

        while val != 'y' and val != 'n':
            val = input(
                '[INFO] It seems like you have already recorded the ambient noise, do you want to recalibrate it?, press [y] to record it again or [n] not to record.')

        if val == 'y':
            recording('Ambient_Noise', 10, False)

    else:
        val5 = input(
            '[INFO] Please be quiet for 10 seconds and the ambient noise will be collected, press [y] to start.')
        while val5 != 'y':
            val5 = input(
                '[INFO] Please be quiet for 10 seconds and the ambient noise will be collected, press [y] to start.')
        recording('Ambient_Noise', 10, False)

    """After recording the noise, then start to register the user"""
    count = 0
    speakers = []
    while True:
        val2 = input('[INFO] Register for speaker, press [y] to start or [n] to finish.')
        if val2 != 'y' and val2 != 'n':
            print('[WARNING] Invalid input')
            continue

        if val2 == 'y':
            val3 = input('[INFO] Register for the ' + str(
                count + 1) + ' th speaker, please enter the speaker name or [n] to exit.')
            while val3 == '' or val3 in speakers:
                print(
                    '[WARNING] Invalid input, maybe you enter an empty speaker name or duplicate speaker name who is '
                    'already registered.')
                val3 = input(
                    '[INFO] Register for the ' + str(count + 1) + ' th speaker, please enter the speaker name.')

            if val3 == 'n':
                continue

            else:
                recording(val3, 60, True, True)
                count += 1
                speakers.append(val3)
                # check whether the valid duration is greater than 50 seconds or not and ask if need to re-record

        if val2 == 'n':
            break

    """Training on experiment dataset"""
    print('[INFO] Training...')
    acc = vi.transfer_learning_on_experiment()
    while acc < 0.80:
        print('[INFO] Retraining...')
        acc = vi.transfer_learning_on_experiment()
    print('[INFO] Training done, accuary is ', acc)

    """Run the speaker identification module"""
    val4 = input('[INFO] Ready for speaker identification, press [s] to start.')
    while val4 != 's':
        val4 = input('[INFO] Ready for speaker identification, press [s] to start.')

    run_speaker_identification(True, True)


if __name__ == '__main__':
    main()
