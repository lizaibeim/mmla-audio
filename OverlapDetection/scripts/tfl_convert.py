import platform

import numpy as np
import tensorflow as tf

from overlap_detector import weighted_categorical_crossentropy, images_loader, labels_loader

if __name__ == '__main__':

    # ZCR
    # labels_path = "D:\OverlapDetection\labelling_results\OverlapLabels.xlsx"
    # images_dir = "D:\OverlapDetection\mel_spectrum_zcr"
    # # path_to_store = "D:\OverlapDetection\\reslblstm_zcr_3classes_unaugmented_weighted_60epochs"
    # osd = OverlapDetector(images_dir, labels_path, imagetype='zcr')
    # # osd.train_model(path_to_store, epochs=60, batch_size=16, augmented=False, weighted=True)
    # osd.populate_model("D:\OverlapDetection\\ckt", augmented=False, weighted=True, images_dir=images_dir, labels_path=labels_path)
    # osd.evaluation('val')
    # osd.evaluation('test')

    # osd.populate_model("D:\OverlapDetection\\resBlstm_zcr_3classes_unaugmented_weighted_60epochs", augmented=False, weighted=True, images_dir=images_dir,
    #                    labels_path=labels_path)
    # osd.evaluation('val')
    # osd.evaluation('test')
    # osd.populate_model("D:\OverlapDetection\\models\\reslstm_zcr_3classes_unaugmented_weighted_60epochs", augmented=False, weighted=True, images_dir=images_dir,
    #                    labels_path=labels_path)
    # osd.evaluation('val')
    # osd.evaluation('test')

    print(tf.__version__)
    print(platform.python_version())

    save_model_dir = "./ckt"
    osd_model = tf.keras.models.load_model(save_model_dir, custom_objects={
        'loss': weighted_categorical_crossentropy([0.96431881, 0.36793327, 0.66774791])})

    run_model = tf.function(lambda x: osd_model(x))
    # FIX THE INPUT SIZE FOR TENSORFLOW LITE CONVERSION
    BATCH_SIZE = 1
    MELS = 128
    TIMESERIES = 151
    CHANNEL = 3

    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, MELS, TIMESERIES, CHANNEL], osd_model.input[0].dtype))
    osd_model.save("./ckt_tfl", save_format='tf', signatures=concrete_func)

    print(osd_model.summary())

    converter = tf.lite.TFLiteConverter.from_saved_model("./ckt_tfl")
    tflite_model = converter.convert()
    with open('converted_model_ckt.tflite', 'wb') as f:
        f.write(tflite_model)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path='./converted_model_ckt.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)

    # Test data
    test_images_dir = "./multisimo/mel_spectrum_zcr_test"
    test_labels_path = "./multisimo/labelling_results/OverlapLabelsTest.xlsx"
    test_x = images_loader(test_images_dir, 3)
    test_y = labels_loader(test_labels_path)

    # Test the model on input data
    input_shape = input_details[0]['shape']

    result = []
    predictions_class = np.argmax(osd_model.predict(test_x), axis=1)
    for i in range(test_x.shape[0]):
        interpreter.set_tensor(input_details[0]['index'], test_x[i:i + 1, :, :])
        interpreter.invoke()
        result.append(interpreter.get_tensor(output_details[0]['index'])[0])

        # TFLite is stateful, need to reset, clean up internal states
        interpreter.reset_all_variables()

    print(result)
    predictions_class_tfl = np.argmax(result, axis=1)

    for i in range(predictions_class.shape[0]):
        np.testing.assert_almost_equal(predictions_class[i], predictions_class_tfl[i])
        print(predictions_class[i], predictions_class_tfl[i])
