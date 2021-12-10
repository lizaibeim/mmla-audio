import json
import math
import os
import sys
import time

import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, add, Input
from tensorflow.keras.layers import LSTM, Dense, Activation, Conv1D, MaxPool1D, Dropout, \
    BatchNormalization, AveragePooling1D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import RMSprop

np.random.seed(0)


# attain the file path list of the dataset
def get_wav_files(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)
    return wav_files


def make_feature_thch30(wav_files):
    train_x = []
    train_y = []
    paragraph_label = []
    begin_time = time.time()
    for i, onewav in enumerate(wav_files):

        if i % 5 == 4:
            gaptime = time.time() - begin_time
            percent = float(i) * 100 / len(wav_files)
            eta_time = gaptime * 100 / (percent + 0.01) - gaptime
            strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
            str_log = ("%.2f %% %s %s/%s \t used:%ds eta:%d s" % (
                percent, strprogress, i, len(train_y), gaptime, eta_time))
            sys.stdout.write('\r' + str_log)

        label = onewav.split("\\")[1].split('_')[0]

        (rate, sig) = wav.read(onewav)
        # print(rate)
        mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, nfft=512)
        # mfcc_feat = mfcc(scipy.signal.resample(sig, len(sig) // 2), rate // 2)

        d_mfcc_feat = delta(mfcc_feat, 2)
        dd_mfcc_feat = delta(d_mfcc_feat, 2)

        finalfeature = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
        # print(finalfeature.shape)
        train_x.append(finalfeature)
        train_y.append(label[1:])
        paragraph_label.append(label[0])

    # yy = LabelBinarizer().fit_transform(train_y)
    yy = binarizer(train_y, dim=5)

    # define the input feature shape (1561, 39) for each wav.file, padded
    for i in range(len(train_x)):
        length = train_x[i].shape[0]
        if length < 256:
            train_x[i] = np.concatenate((train_x[i], np.zeros((256 - length, 39))), axis=0)
        else:
            train_x[i] = train_x[i][:256, :]
            print('Oops exceeds,', train_x[i].shape)
    # train_x = [np.concatenate((j, np.zeros((1561-j.shape[0], 39)))) for j in train_x]
    train_x = np.asarray(train_x)
    train_y = np.asarray(yy)
    paragraph_label = np.asarray(paragraph_label)
    print(paragraph_label)

    return train_x, train_y, paragraph_label


# rewrite one-hot binarizer
def binarizer(str_list, dim, speakers_count_dict={}):
    count = 0
    for i in range(len(str_list)):
        if str_list[i] not in speakers_count_dict:
            # print(str_list[i], 'not in dict', 'count', count)
            speakers_count_dict[str_list[i]] = count
            count += 1

        label_bin_arr = np.zeros((1, dim))
        label_bin_arr[0, speakers_count_dict[str_list[i]]] = 1

        if i == 0:
            result = label_bin_arr
        else:
            result = np.concatenate((result, label_bin_arr))

    return result


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


# this functions is used to count the number of audio clips of different users
def statistics(wav_files):
    statistics = {}
    for i, onewav in enumerate(wav_files):

        label = onewav.split("\\")[1].split('_')[0][1:]
        if label not in statistics:
            statistics[label] = 1
        else:
            statistics[label] = statistics[label] + 1

    return statistics


def res_unit(x, filters, pool=False, regularized=0.0):
    res = x
    if pool:
        x = MaxPool1D(2, padding="same")(x)
        res = Conv1D(filters=filters, kernel_size=1, strides=2, padding="same")(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    if regularized > 0:
        out = Conv1D(filters=filters, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(regularized))(
            out)
    else:
        out = Conv1D(filters=filters, kernel_size=3, padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    if regularized > 0:
        out = Conv1D(filters=filters, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(regularized))(
            out)
    else:
        out = Conv1D(filters=filters, kernel_size=3, padding="same")(out)
    out = add([res, out])

    return out


def res_model(input_shape):
    mfcc_features = Input(input_shape)
    net = Conv1D(filters=32, kernel_size=4, padding="same")(mfcc_features)
    net = res_unit(net, 32, pool=True)
    net = res_unit(net, 32)
    net = res_unit(net, 32)

    net = res_unit(net, 64, pool=True)
    net = res_unit(net, 64, regularized=0.1)
    net = res_unit(net, 64, regularized=0.1)

    net = res_unit(net, 128, pool=True)
    net = res_unit(net, 128, regularized=0.2)
    net = res_unit(net, 128, regularized=0.2)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.25)(net)

    net = AveragePooling1D(4)(net)
    net = Bidirectional(LSTM(256))(net)
    net = Dropout(0.2)(net)
    net = Dense(630, activation='softmax')(net)

    model = Model(inputs=mfcc_features, outputs=net)
    return model


def train(x_train, y_train, x_validate, y_validate):
    # class_dim = y_train.shape[1]
    # model = Sequential()
    # model.add(Conv1D(32, 4, input_shape=(1024, 39)))
    # model.add(MaxPool1D(4))
    # model.add(BatchNormalization())
    # model.add(Conv1D(64, 4))
    # model.add(MaxPool1D(4))
    # model.add(BatchNormalization())
    # model.add(Bidirectional(LSTM(256)))
    # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.25))
    # model.add(Dense(class_dim, activation='softmax'))
    #
    model = res_model(x_train[0].shape)

    # learning_rate_adjust = [
    #     CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
    # ]
    print(model.summary())
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./ckt',
                                                    monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                                    mode='max', period=10)
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.0001), metrics=[categorical_accuracy])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate),
                        callbacks=[checkpoint, early_stopping],
                        batch_size=32, epochs=200)

    return model, history


def matrix_build_pca(feature_data):
    matrix_list = []
    width = feature_data[0].shape[0] * feature_data[0].shape[1]
    for i in range(len(feature_data)):
        matrix_list.append(feature_data[i].flatten())

    matrix = np.array(matrix_list)
    np.savez('feature_matrix', matrix)
    return matrix


def make_feature_timit(wav_files, speaker_ids):
    DATASET = './timit/data'
    train_x = []
    train_y = []

    begin_time = time.time()
    for i in range(len(wav_files)):

        if i % 5 == 4:
            gaptime = time.time() - begin_time
            percent = float(i) * 100 / len(wav_files)
            eta_time = gaptime * 100 / (percent + 0.01) - gaptime
            strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
            str_log = ("%.2f %% %s %s/%s \t used:%ds eta:%d s" % (
                percent, strprogress, i, len(train_y), gaptime, eta_time))
            sys.stdout.write('\r' + str_log)

        label = speaker_ids.values[i]
        onewav = os.path.join(DATASET, wav_files.values[i])

        (rate, sig) = wav.read(onewav)
        mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, nfft=512)
        # fbank_feat = fbank(sig, rate, winlen=0.025, winstep=0.01, nfft=512)[1].reshape(-1, 1)

        # mfcc_feat = mfcc(scipy.signal.resample(sig, len(sig) // 2), rate // 2)

        d_mfcc_feat = delta(mfcc_feat, 2)
        dd_mfcc_feat = delta(d_mfcc_feat, 2)

        finalfeature = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
        # print(finalfeature.shape)
        train_x.append(finalfeature)
        train_y.append(label)

    # yy = LabelBinarizer().fit_transform(train_y)
    yy = binarizer(train_y, dim=630)

    # define the input feature shape (1561, 39) for each wav.file, padded
    dim = train_x[0].shape[1]
    for i in range(len(train_x)):
        length = train_x[i].shape[0]
        if length < 256:
            train_x[i] = np.concatenate((train_x[i], np.zeros((256 - length, dim))), axis=0)
        else:
            train_x[i] = train_x[i][:256, :]
            print('Oops exceeds,', train_x[i].shape)
    # train_x = [np.concatenate((j, np.zeros((1561-j.shape[0], 39)))) for j in train_x]
    train_x = np.asarray(train_x)
    train_y = np.asarray(yy)

    return train_x, train_y


def make_feature_experiment(wav_files):
    train_x = []
    train_y = []
    begin_time = time.time()
    for i, onewav in enumerate(wav_files):

        if i % 5 == 4:
            gaptime = time.time() - begin_time
            percent = float(i) * 100 / len(wav_files)
            eta_time = gaptime * 100 / (percent + 0.01) - gaptime
            strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
            str_log = ("%.2f %% %s %s/%s \t used:%ds eta:%d s" % (
                percent, strprogress, i, len(train_y), gaptime, eta_time))
            sys.stdout.write('\r' + str_log)

        # for windows
        label = onewav.split('\\')[1][:-4]

        # for raspberry pi
        # label = onewav.split('/')[3][:-4]

        (rate, sig) = wav.read(onewav)
        mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, nfft=512)

        d_mfcc_feat = delta(mfcc_feat, 2)
        dd_mfcc_feat = delta(d_mfcc_feat, 2)

        finalfeature = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
        length = finalfeature.shape[0]
        segments_no = math.ceil(length / 256)
        finalfeature = np.concatenate((finalfeature, np.zeros((segments_no * 256 - length, 39))), axis=0)

        for i in range(segments_no):
            train_x.append(finalfeature[i * 256:(i + 1) * 256, :])
            train_y.append(label)

    # yy = LabelBinarizer().fit_transform(train_y)
    dimension = len(set(train_y))
    values = train_y
    # print('dimension: ', dimension)

    yy = binarizer(train_y, dim=dimension)

    train_x = np.asarray(train_x)
    train_y = np.asarray(yy)
    speaker_id = {str(np.argmax(train_y[i])): values[i] for i in range(len(values))}

    # speaker_id = {''.join(map(str, train_y[i].astype(int).tolist())): values[i] for i in range(len(values))}
    # print(speaker_id)
    # print(train_y)
    return train_x, train_y, speaker_id


def input_feature_gen(wav_path):
    lst = []
    (rate, sig) = wav.read(wav_path)
    if len(sig) < 4000:
        return 'silent'

    # fixed_length = int(2.57 * 16000)
    # pad = fixed_length - len(sig)
    #
    # if pad < 0:
    #     sig = sig[:fixed_length]
    # else:
    #     sig = np.pad(sig, (0, pad), 'constant')

    mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, nfft=512)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    finalfeature = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)

    length = finalfeature.shape[0]
    if length < 256:
        finalfeature = np.concatenate((finalfeature, np.zeros((256 - length, 39))), axis=0)
    else:
        finalfeature = finalfeature[:256, :]

    lst.append(finalfeature)
    return np.asarray(lst)


def transfer_learning(_x, _y, seed, _test_split_ratio, base_model_path, final_model_path):
    base_model = load_model(base_model_path)
    sliced_base_model = Model(base_model.input, base_model.layers[-2].output)
    sliced_base_model.trainable = False
    inputs = sliced_base_model.input
    x = sliced_base_model(inputs, training=False)
    dim = np.unique(_y, axis=0).shape[0]
    outputs = Dense(dim, activation='softmax', name='customized_dense')(x)
    model = Model(inputs, outputs)

    model.compile(loss="categorical_crossentropy",
                  optimizer=RMSprop(lr=0.0001),
                  metrics=[categorical_accuracy]
                  )
    epochs = 200

    # print(_x.shape)

    x_train, y_train = _x, _y
    if _test_split_ratio > 0:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=_test_split_ratio, stratify=_y,
                                                            random_state=seed)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train,
                                                                random_state=seed)
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=30)
    model.fit(x_train, y_train, validation_data=(x_validate, y_validate), callbacks=[early_stopping],
              batch_size=32, epochs=epochs)

    if _test_split_ratio > 0:
        loss, accuracy = model.evaluate(x_test, y_test)
        print("Loss on test set:", loss, "Accuracy: ", accuracy)

    model.trainable = True
    model.compile(loss="categorical_crossentropy",
                  optimizer=RMSprop(lr=1e-5),
                  metrics=[categorical_accuracy]
                  )

    epochs = 10
    model.fit(x_train, y_train, validation_data=(x_validate, y_validate), batch_size=32,
              epochs=epochs)

    if _test_split_ratio > 0:
        loss, accuracy = model.evaluate(x_test, y_test)
        print("After fine tuning, Loss on test set:", loss, "Accuracy: ", accuracy)

    else:
        loss, accuracy = model.evaluate(x_validate, y_validate)

    model.save(final_model_path)

    '''Do the tensorflow lite convert'''
    # run_model = tf.function(lambda x: model(x))
    # # fix the input size for tfl model
    # batch_size = 1
    # time_series = 256
    # mfcc_mels = 39
    #
    # concrete_func = run_model.get_concrete_function(
    #     tf.TensorSpec([batch_size, time_series, mfcc_mels], model.input[0].dtype)
    # )
    #
    # tfl_model_path = final_model_path[:-5] + 'model_tfl'
    # tfl_file_path = final_model_path[:-5] + 'converted_model.tflite'
    # model.save(tfl_model_path, save_format='tf', signatures=concrete_func)
    # converter = tf.lite.TFLiteConverter.from_saved_model('./experiment/model_tfl')
    # tflite_model = converter.convert()
    # with open(tfl_file_path, 'wb') as f:
    #     f.write(tflite_model)

    return accuracy


def transfer_learning_on_experiment():
    wav_files = get_wav_files("./experiment/data")

    x, y, speaker_id = make_feature_experiment(wav_files)
    np.savez("./experiment/experiment_feature", x, y)
    with open('./experiment/speaker_id_dict.json', 'w') as f:
        json.dump(speaker_id, f)

    input_feature = np.load('./experiment/experiment_feature.npz', allow_pickle=True)
    x = input_feature['arr_0']
    y = input_feature['arr_1']

    test_split_ratio = 0
    random_seed = 0
    base_model_path = './timit/model'
    final_model_path = './experiment/model'
    acc = transfer_learning(x, y, random_seed, test_split_ratio, base_model_path, final_model_path)

    return acc


if __name__ == '__main__':
    '''
    First training on TIMIT, generating the features
    '''
    # wav_files_path = pd.read_csv('./timit/labels.csv').loc[:, 'path_from_data_dir_windows']
    # speaker_ids = pd.read_csv('./timit/labels.csv').loc[:, 'speaker_id']
    # _x, _y = make_feature_timit(wav_files_path, speaker_ids)
    # np.savez('./timit/input_feature', _x, _y)

    '''
    After the first training, just load the feautre directly
    '''
    # speakers_count_dict = np.load('speakers_count_dict.npy', allow_pickle=True).item()
    # input_feature = np.load('./timit/input_feature.npz')
    # _x = input_feature['arr_0']
    # _y = input_feature['arr_1']
    #
    # print(_x.shape, _y.shape)
    #
    # x_train, x_test, y_train, y_test = train_test_split(_x, _y, test_size=0.2, random_state=0, stratify=_y)
    # x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
    #
    # trained_model, history = train(x_train, y_train, x_validate, y_validate)
    # trained_model.save('./cnnlstm')
    # epochs = range(len(history.history['categorical_accuracy']))
    # plt.figure()
    # plt.plot(epochs, history.history['categorical_accuracy'], 'b', label='categorical_accuracy')
    # plt.plot(epochs, history.history['val_categorical_accuracy'], 'r', label='val_categorical_accuracy')
    # plt.title("Accuracy on Training and Validation Data")
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.savefig('acc.png', dpi=500, bbox_inches='tight')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(epochs, history.history['loss'], 'b', label='loss')
    # plt.plot(epochs, history.history['val_loss'], 'r', label='val_loss')
    # plt.title("Loss on Training and Validation Data")
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.savefig('loss.png', dpi=500, bbox_inches='tight')
    # plt.show()

    '''After the first general traning, just load the pre-trained general model'''
    # trained_model = load_model('./ckt')
    # print(trained_model.summary())
    # loss, accuracy = trained_model.evaluate(x_test, y_test)
    # print("Loss on test set:", loss, "Accuracy: ", accuracy)

    '''Transfer learning on smaller dataset'''
    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.

    # wav_files = get_wav_files("./small_dataset/data")
    # _x, _y, paragraph_label = make_feature_thch30(wav_files)
    # np.savez("./small_dataset/small_dataset_feature", _x, _y)

    '''Transfer learning on experimental dataset'''
    wav_files = get_wav_files("./experiment/data")

    x, y, speaker_id = make_feature_experiment(wav_files)
    np.savez("./experiment/experiment_feature", x, y)
    with open('./experiment/speaker_id_dict.json', 'w') as f:
        json.dump(speaker_id, f)

    input_feature = np.load('./experiment/experiment_feature.npz', allow_pickle=True)
    x = input_feature['arr_0']
    y = input_feature['arr_1']

    test_split_ratio_lst = [0]
    acc_lst_lst = []
    base_model_path = './timit/model'
    final_model_path = './experiment/model'
    with open("./experiment/acc.txt", "w") as f:
        f.write("split\trandom_state\tacc\n")
        for test_split_ratio in test_split_ratio_lst:
            acc_lst = []
            for i in range(0, 1):
                print(
                    "################################################################################################")
                print("test split, ", test_split_ratio, "random seed, ", i)
                print(
                    "################################################################################################")

                acc = transfer_learning(x, y, i, test_split_ratio, base_model_path, final_model_path)
                f.write(str(test_split_ratio) + "\t")
                f.write(str(i) + "\t")
                f.write(str(acc) + "\n")
        f.close()
