import time
import wave
from math import ceil

import RPi.GPIO as GPIO
import cv2
import librosa
import noisereduce as nr
import numpy as np
import pyaudio
import soundfile as sf
import tensorflow as tf
from skimage.metrics._structural_similarity import structural_similarity

from overlap_features_generator import OverlapFeaturesGenerator

BUTTON = 17
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 2
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 1  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 10  # read from stream for each 1.5 seconds
WAVE_OUTPUT_FILENAME = "output"
OVERLAP_DETECTOR_TFL = "converted_model_ckt.tflite"
NOISE = "noise.wav"
RECORDING_DIR = "./recordings/"
NOISE_REDUCED_RECORDING_DIR = './noise_reduced_recordings/'
NOISE_REDUCE_TIMES = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON, GPIO.IN)

p = pyaudio.PyAudio()
noise, _ = librosa.load(NOISE, sr=None)


def compare_images(img1_path, img2_path):
    image1 = cv2.imread(img1_path)
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.imread(img2_path)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    s = structural_similarity(image1, image2, multichannel=True)
    if s < 0.3:
        print('The similarity is too low, which is ', s)
        return True
    return False


def open_stream():
    _stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX, )
    return _stream


print("[INFO] Loading tensorflow lite model...")
# Load the TFLite model and allocate tensors.
start_time = time.time()
interpreter = tf.lite.Interpreter(model_path=OVERLAP_DETECTOR_TFL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
end_time = time.time()
print("[INFO] Time usage for loading tensorflow lite model:{:.5f} seconds\n".format(end_time - start_time))

print("[INFO] Start recording")
ofg = OverlapFeaturesGenerator(wl=25, hl=10)
count = 0
on_record = True
shut_down = False
stream = open_stream()

while True:
    state = GPIO.input(BUTTON)
    frames = []
    if state:
        if not shut_down:
            on_record = True
            for i in range(0, ceil(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                # extract channel 0 data from 2 channels, if you want to extract channel 1, please change to [1::2]
                a = np.fromstring(data, dtype=np.int16)[0::2]
                frames.append(a.tostring())

            count += 1
            wav_file_path = RECORDING_DIR + WAVE_OUTPUT_FILENAME + str(count) + '.wav'
            wf = wave.open(wav_file_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
            wf.setframerate(RESPEAKER_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            print("[INFO] Recording for 1.5 second, predicting now...")
            start_time = time.time()
            wav_file_path_noise_reduced = NOISE_REDUCED_RECORDING_DIR + WAVE_OUTPUT_FILENAME + str(
                count) + '_noise_reduced.wav'
            for i in range(NOISE_REDUCE_TIMES):
                if i == 0:
                    y, sr = librosa.load(wav_file_path, sr=None)
                    noise_reduced_wav = nr.reduce_noise(y_noise=noise, y=y, sr=sr, stationary=True)
                    sf.write(wav_file_path_noise_reduced, noise_reduced_wav, 16000)

                else:
                    y, sr = librosa.load(wav_file_path_noise_reduced, sr=None)
                    noise_reduced_wav = nr.reduce_noise(y_noise=noise, y=y, sr=sr, stationary=True)
                    sf.write(wav_file_path_noise_reduced, noise_reduced_wav, 16000)

            features_image_path = wav_file_path[:-4] + '.png'
            features_image_path2 = wav_file_path_noise_reduced[:-4] + '.png'

            ofg.generate_zcr_image(wav_file_path, features_image_path)
            ofg.generate_zcr_image(wav_file_path_noise_reduced, features_image_path2)

            if compare_images(features_image_path, features_image_path2):
                end_time = time.time()
                print("[INFO] Prediction result for last 1.5 second: silent. Time usage: {:.5f} s\n".format(
                    end_time - start_time))
                continue

            image = tf.io.read_file(features_image_path2)
            features_data = [tf.image.decode_png(image, 3)]
            _input = tf.stack(features_data, axis=0).numpy().astype('float32')
            _input_tfl = np.expand_dims(_input, axis=0)

            interpreter.set_tensor(input_details[0]['index'], _input_tfl[0, :, :])
            interpreter.invoke()
            result = interpreter.get_tensor(output_details[0]['index'])[0]
            interpreter.reset_all_variables()

            print("[INFO] Prediction result for last 1.5 second: %s. Time usage: %f s\n" % (
            result, end_time - start_time,))

        else:
            if on_record:
                on_record = False
                print("[INFO] Recording suspend...")

    else:
        if on_record:
            if not shut_down:
                print("[INFO] Terminate recording...")
                stream.stop_stream()
                stream.close()
                shut_down = True
                time.sleep(2)

        if not on_record:
            if shut_down:
                print("[INFO] Try to restart recording...")
                p = pyaudio.PyAudio()
                stream = open_stream()
                shut_down = False
                time.sleep(2)
