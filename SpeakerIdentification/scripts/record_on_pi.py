import collections
import json
import time
import wave
from math import ceil

import RPi.GPIO as GPIO
import keyboard
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import tensorflow as tf
import webrtcvad
from pyaudio import PyAudio

import speaker_identification as vi

BUTTON = 17
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 2
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 1  # refer to input device id
CHUNK = 2000
RECORD_SECONDS = 5  # recording voice for each 2.56 seconds, considering the blank space, set it as 4 seconds
WAVE_OUTPUT_FILENAME = "segment"
VOICE_IDENTIFIER_TFL = "./experiment/converted_model.tflite"
NOISE_PATH = "./experiment/Ambient_Noise.wav"
vad = webrtcvad.Vad(2)
p = PyAudio()

# RECORDING_DIR = "./recordings/"
# NOISE_REDUCED_RECORDING_DIR = './noise_reduced_recordings/'
# NOISE_REDUCE_TIMES = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON, GPIO.IN)


def open_stream():
    _stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX, )
    return _stream


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


"""
Recording for speaker registration
"""


def recording(filename, duration, noise_reduced=False, silence_remove=False):
    filepath = './experiment/data/' + str(filename) + '.wav'
    stream = open_stream()
    frames = []
    t = time.time()
    if not noise_reduced:
        print('[INFO] Noise Recording, Please keep quiet...')
    else:
        print('[INFO] Recording...')

    for i in range(0, ceil(RESPEAKER_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        a = np.fromstring(data, dtype=np.int16)[0::2]
        frames.append(a.tostring())

    print('[INFO] Recroding Done.')

    if not noise_reduced:
        save_wave_file(NOISE_PATH, frames, noise_reduce=False, silence_remove=silence_remove)

    else:
        save_wave_file(filepath, frames, noise_reduce=True, silence_remove=silence_remove)

    stream.close()


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
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
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
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
        wf.setframerate(RESPEAKER_RATE)
        for i, segments in enumerate(segments):
            wf.writeframes(segments)
        wf.close()


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


def run_speaker_identification(noise_reduced=False, silence_removed=False):
    quit_flag = False
    model_path = './experiment/model'
    model = tf.keras.models.load_model(model_path)
    with open('./experiment/speaker_id_dict.json') as f:
        speaker_id_dict = json.load(f)
    print(speaker_id_dict)

    print('[INFO] Model loaded: start predicting...')
    stream = open_stream()

    count = 0
    while not quit_flag:
        print('[INFO] One iteration...')
        frames = []
        t = time.time()
        for i in range(0, ceil(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):  # set the predicting duration
            if keyboard.is_pressed('q'):
                quit_flag = True
                print("Quit now...")
                # break

            # loop of read，read 2000 frames each iteration (0.175s)
            data = stream.read(CHUNK, exception_on_overflow=False)
            a = np.fromstring(data, dtype=np.int16)[0::2]
            frames.append(a.tostring())

        filepath = './experiment/run/' + str(count) + '.wav'

        save_wave_file(filepath, frames, noise_reduce=noise_reduced, silence_remove=silence_removed)
        x = vi.input_feature_gen(filepath)
        if x == 'silent':
            print('Predcition for the last 2 seconds: silent')
            count += 1
            with open('./experiment/logs.txt', 'a') as f:
                if count == 1:
                    f.write('segment' + '\t' + 'speaker')
                    f.write('\n')

                f.write(str(count) + '\t' + 'silent')
                f.write('\n')
            continue

        prob = model.predict(x)
        key = str(np.argmax(model.predict(x), axis=1)[0])
        print('[RESULT] Predcition for the last 2 seconds: ', 'probability: ', prob, 'speaker: ', speaker_id_dict[key])

        count += 1
        with open('./experiment/logs.txt', 'a') as f:
            if count == 1:
                f.write('segment' + '\t' + 'speaker')
                f.write('\n')

            f.write(str(count) + '\t' + str(speaker_id_dict[key]))
            f.write('\n')

    stream.close()


def run_speaker_identification_tfl(noise_reduced=False, silence_removed=False):
    print("[INFO] Loading tensorflow lite model...")
    # Load the TFLite model and allocate tensors.
    start_time = time.time()
    quit_flag = False
    interpreter = tf.lite.Interpreter(model_path=VOICE_IDENTIFIER_TFL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    end_time = time.time()
    print("[INFO] Time usage for loading tensorflow lite model:{:.5f} seconds\n".format(end_time - start_time))

    with open('./experiment/speaker_id_dict.json') as f:
        speaker_id_dict = json.load(f)
    print(speaker_id_dict)

    print('[INFO] Model loaded: start predicting...')
    stream = open_stream()

    count = 0
    while not quit_flag:
        print('[INFO] One iteration...')
        frames = []
        t = time.time()
        for i in range(0, ceil(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):  # set the predicting duration
            if keyboard.is_pressed('q'):
                quit_flag = True
                print("Quit now...")
                # break

            # loop of read，read 2000 frames each iteration (0.175s)
            data = stream.read(CHUNK, exception_on_overflow=False)
            a = np.fromstring(data, dtype=np.int16)[0::2]
            frames.append(a.tostring())

        filepath = './experiment/run/' + str(count) + '.wav'

        save_wave_file(filepath, frames, noise_reduce=noise_reduced, silence_remove=silence_removed)
        x = vi.input_feature_gen(filepath)
        if x == 'silent':
            print('Predcition for the last 2 seconds: silent')
            count += 1
            with open('./experiment/logs.txt', 'a') as f:
                if count == 1:
                    f.write('segment' + '\t' + 'speaker')
                    f.write('\n')

                f.write(str(count) + '\t' + 'silent')
                f.write('\n')
            continue

        _input_tfl = np.expand_dims(x, axis=0)
        interpreter.set_tensor(input_details[0]['index'], _input_tfl[0, :, :])
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])[0]
        interpreter.reset_all_variables()
        print("[INFO] Prediction result for last 2.5 second: %s. Time usage: %f s\n" % (result, end_time - start_time,))

        count += 1
        with open('./experiment/logs.txt', 'a') as f:
            if count == 1:
                f.write('segment' + '\t' + 'speaker')
                f.write('\n')

            f.write(str(count) + '\t' + result)
            f.write('\n')

    stream.close()


# recording
def main():
    """pivot ambient noise, set it as ten seconds duration"""

    val = input('[INFO] Please be quiet for 10 seconds and the ambient noise will be collected, press [Y] to start.')
    while val != 'y':
        val = input(
            '[INFO] Please be quiet for 10 seconds and the ambient noise will be collected, press [Y] to start.')
    recording('Ambient_Noise', 10, False, False)

    """After recording the noise, then start to register the user"""
    count = 0
    speakers = []
    while True:
        val2 = input('[INFO] Register for speaker, press [Y] to start or [N] to finish.')
        if val2 != 'y' and val2 != 'n':
            print('[WARNING] Invalid input')
            continue

        if val2 == 'y':
            val3 = input('[INFO] Register for the ' + str(
                count + 1) + ' th speaker, please enter the speaker name or [N] to exit.')
            while val3 == '' or val3 in speakers:
                print(
                    '[WARNING] Invalid input, maybe you enter an empty speaker name or duplicate speaker name who is already registered.')
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
    print('[INFO] Training done, accuary is ', acc)

    """Run the speaker identification module"""
    val4 = input('[INFO] Ready for speaker identification, press [S] to start.')
    while val4 != 's':
        val4 = input('[INFO] Ready for speaker identification, press [S] to start.')

    run_speaker_identification(True, True)


if __name__ == '__main__':
    main()
