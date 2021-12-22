import argparse
import os
import time

import numpy as np
import tensorflow as tf

from overlap_features_generator import OverlapFeaturesGenerator, weighted_categorical_crossentropy

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--audio", nargs='?', help="path to audio file")
ap.add_argument("-dir", "--audio_dir", nargs='?', help="path to audio files directory")
ap.add_argument("-tf", "--model_tf", nargs='?', help="path to pre-trained tensorflow model")
ap.add_argument("-tfl", "--model_tfl", nargs='?', help="path to pre-trained tensorflow lite model")
# args = vars(ap.parse_args())
args, leftovers = ap.parse_known_args()

ofg = OverlapFeaturesGenerator(wl=25, hl=10)

print("[INFO] Generating Intermediate Features...")
start_time = time.time()

_input = None
_input_tfl = None

if args.audio is not None:
    wav_file_path = args.audio
    features_image_path = wav_file_path[:-4] + '.png'
    ofg.generate_zcr_image(wav_file_path, features_image_path)
    image = tf.io.read_file(features_image_path)
    features_data = [tf.image.decode_png(image, 3)]
    _input = tf.stack(features_data, axis=0).numpy().astype('float32')
    _input_tfl = np.expand_dims(_input, axis=0)

_input_dict = {}
_input_tfl_dict = {}
if args.audio_dir is not None:
    wav_files_dir = args.audio_dir
    for file in os.listdir(wav_files_dir):
        wav_file_path = os.path.join(wav_files_dir, file)
        features_image_path = wav_file_path[:-4] + '.png'
        ofg.generate_zcr_image(wav_file_path, features_image_path)
        image = tf.io.read_file(features_image_path)
        features_data = [tf.image.decode_png(image, 3)]
        _input_dict[file] = tf.stack(features_data, axis=0).numpy().astype('float32')
        _input_tfl_dict[file] = np.expand_dims(_input_dict[file], axis=0)

end_time = time.time()
print("[INFO] Time usage for generating features:{:.5f} seconds\n".format(end_time - start_time))

if args.model_tf is not None:
    print("[INFO] Loading tensorflow model...")
    start_time = time.time()
    osd_model = tf.keras.models.load_model(args.model_tf, custom_objects={
        'loss': weighted_categorical_crossentropy([0.96431881, 0.36793327, 0.66774791])})
    end_time = time.time()
    print("[INFO] Time usage for loading tensorflow model:{:.5f} seconds\n".format(end_time - start_time))

    print("[INFO] Predicting sample...")
    start_time = time.time()
    if _input is not None:
        result = osd_model.predict(_input)
        print("[INFO] Prediction result for single input: %s" % (result,))

    if len(_input_dict) > 0:
        for key, value in sorted(_input_dict.items()):
            result = osd_model.predict(value)
            print("[INFO] Prediction result for %s: %s" % (key, result,))

    end_time = time.time()
    print("[INFO] Time usage for tensorflow inference:{:.5f} seconds\n".format(end_time - start_time))

if args.model_tfl is not None:
    print("[INFO] Loading tensorflow lite model...")
    # Load the TFLite model and allocate tensors.
    start_time = time.time()
    interpreter = tf.lite.Interpreter(model_path=args.model_tfl)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    end_time = time.time()
    print("[INFO] Time usage for loading tensorflow lite model:{:.5f} seconds\n".format(end_time - start_time))

    print('[INFO] Predicting sample on tensorflow lite..."')
    start_time = time.time()
    if _input_tfl is not None:
        interpreter.set_tensor(input_details[0]['index'], _input_tfl[0, :, :])
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])[0]
        interpreter.reset_all_variables()
        print("[INFO] Prediction result for single input: %s" % (result,))

    if len(_input_tfl_dict) > 0:
        for key, value in sorted(_input_tfl_dict.items()):
            interpreter.set_tensor(input_details[0]['index'], value[0, :, :])
            interpreter.invoke()
            result = interpreter.get_tensor(output_details[0]['index'])[0]
            interpreter.reset_all_variables()
            print("[INFO] Prediction result for %s: %s" % (key, result,))

    end_time = time.time()

    print("[INFO] Time usage of tensorflow lite inference:{:.5f} seconds".format(end_time - start_time))
