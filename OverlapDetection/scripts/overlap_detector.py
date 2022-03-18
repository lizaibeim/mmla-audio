import os
import pandas as pd
import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from cosine_annealing import CosineAnnealingScheduler
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Bidirectional, add, Input, LSTM, Dense, Activation, Conv2D, MaxPool2D, Flatten, \
    Dropout, BatchNormalization, Lambda, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
# from keras import backend as K
from sklearn.preprocessing import OneHotEncoder

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)


def images_loader(image_dir, channels):
    """
    :param image_dir: absolute path of the mel-spectrogram images
    :param channels: image channels (gray-scale: 1, rgb: 3)
    :return: np array of image data
    """
    names = os.listdir(image_dir)
    # sort images file by sessions, then segments in ascending order
    names.sort(key=lambda name: (int(name.segment('.')[0].segment('_')[0][1:3]), int(name.segment('.')[0].segment('_')[3])))
    images_path = [os.path.join(image_dir, name) for name in names]

    images_data = []
    for image_file in images_path:
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image, channels)
        images_data.append(image)

    # convert to tensor with shape (samples, image.shape[0], image.shape[1])
    features_data = tf.stack(images_data, axis=0)

    return features_data.numpy().astype('float32')


def labels_loader(label_path):
    """
    :param label_path: absolute path of the labels file (.xlsx)
    :return: np array of label
    """
    df = pd.read_excel(label_path)
    # sort labels by session, then segment in ascending order
    df.sort_values(by=['Sessions', 'Segments'], inplace=True)
    labels = np.array(df['Overlap'])
    enc = OneHotEncoder()
    rlabels = np.reshape(labels, (-1, 1))
    y = enc.fit_transform(rlabels).toarray()

    return y.astype('float32')


def weighted_categorical_crossentropy(weights):
    """
    :param weights: weights array
    :return: loss function
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        weight_loss = y_true * K.log(y_pred) * weights
        weight_loss = -K.sum(weight_loss, -1)

        return weight_loss

    return loss


class OverlapDetector:

    def __init__(self, images_dir, labels_path, *, imagetype='rgb'):
        """
        :param images_dir: absolute path of the directory storing the images
        :param labels_path: absolute path of the label .xlsx file
        :param imagetype: the image_type of input images, 'gray' or 'rgb'
        """
        self.channel = 1 if imagetype == 'gray' else 3
        self.images_dir = images_dir
        self.labels_path = labels_path
        self.__train_labels_path = None
        self.__train_images_dir = None
        self.__test_labels_path = None
        self.__test_images_dir = None
        self.__train_aug_labels_path = None
        self.__train_aug_images_dir = None
        self.__weights_factor = None
        self.__model_path = None
        self.__augmented = False
        self.__weighted = False
        self.__x = None
        self.__y = None
        self.__x_train = None
        self.__y_train = None
        self.__x_test = None
        self.__y_test = None
        self.__split_train_test()

    @property
    def augmented(self):
        return self.__augmented

    @property
    def weighted(self):
        return self.__weighted

    @property
    def model_path(self):
        return self.__model_path

    @staticmethod
    def cal_weighted_penalty(y):
        """
        calculate the weight factors for each class, the number of classes is set to 3 by default
        """
        quantity = np.zeros(y.shape[1])

        for label_prob in y:
            for i in range(y.shape[1]):
                if label_prob[i] == 1:
                    quantity[i] += 1

        weights = np.zeros(y.shape[1])
        for i in range(y.shape[1]):
            weights[i] = 1 - (quantity[i] / np.sum(quantity))

        print(weights)
        return weights

    @staticmethod
    def augment_images(in_images_dir, in_labels_path, out_images_dir, out_labels_path):
        """
        :param in_images_dir: the directory of images to be augmented
        :param in_labels_path: the label path of images to be augments
        :param out_images_dir: the directory of augmented images to store
        :param out_labels_path: the label path of augmented images
        """
        if not os.path.isdir(out_images_dir):
            os.mkdir(out_images_dir)

        # read origin labels and sort
        df = pd.read_excel(in_labels_path)
        df.sort_values(by=['Sessions', 'Segments'], inplace=True)
        labels = np.array(df['Overlap'])

        # calculated the ratio for each class to duplicate
        count = [0, 0, 0]
        ratio = [0, 0, 0]
        for label in labels:
            count[label] += 1

        base = max(count)
        for m in range(3):
            ratio[m] = base / count[m] - 1

        # images path list for each class
        images_c0 = []
        images_c1 = []
        images_c2 = []
        images_classes_path = []

        names = os.listdir(in_images_dir)
        # sort images file by sessions, then segments in ascending order aligned with labels' order
        names.sort(
            key=lambda x: (int(x.segment('.')[0].segment('_')[0].replace('S', '')), int(x.segment('.')[0].segment('_')[3])))

        for i in range(len(df)):
            if df.iloc[i]['Overlap'] == 0:
                images_c0.append(names[i])
            elif df.iloc[i]['Overlap'] == 1:
                images_c1.append(names[i])
            elif df.iloc[i]['Overlap'] == 2:
                images_c2.append(names[i])
        images_classes_path.append(images_c0)
        images_classes_path.append(images_c1)
        images_classes_path.append(images_c2)

        # duplicate original images to augment folder first
        for images_classes in images_classes_path:
            for name in images_classes:
                image_path = os.path.join(in_images_dir, name)
                src = cv.imread(image_path)
                os.chdir(out_images_dir)
                cv.imwrite(name, src)

        # traverse each class, augment images
        for k in range(3):
            images_classes = images_classes_path[k]

            # duplicate images for ratio[k] times
            for i in range(round(ratio[k])):

                for name in images_classes:
                    image_path = os.path.join(in_images_dir, name)
                    src = cv.imread(image_path)

                    # down-sampling image and then up-sampling for i+1 times
                    for j in range(i + 1):
                        src = cv.pyrDown(src)
                        src = cv.pyrUp(src)

                    image_augmented = src[:, :-1]
                    session = name.split('.')[0].split('_')[0]
                    segment = name.split('.')[0].split('_')[3]
                    new_segment = str(1000 + int(segment)) + str(i)
                    write_name = session + '_audio_MONO_' + new_segment + '_16000_split.png'
                    os.chdir(out_images_dir)
                    cv.imwrite(write_name, image_augmented)
                    df = df.append([{'Sessions': session, 'Segments': int(new_segment), 'Overlap': k}],
                                   ignore_index=True)

        sorted_df = df.sort_values(by=['Sessions', 'Segments'], ascending=[True, True])
        sorted_df.to_excel(out_labels_path, index=None)

    @staticmethod
    def res_block(x, filters, pool=False, regularized=0.0):

        res = x
        if pool:
            res = Conv2D(filters=filters, kernel_size=1, strides=2, padding="same")(res)

        out = BatchNormalization()(x)
        out = Activation("elu")(out)

        if regularized > 0:
            out = Conv2D(filters=filters, kernel_size=3, padding="same",
                         kernel_regularizer=regularizers.l2(regularized))(
                out)
        else:
            out = Conv2D(filters=filters, kernel_size=3, padding="same")(out)

        out = BatchNormalization()(out)
        out = Activation("elu")(out)

        if regularized > 0:
            out = Conv2D(filters=filters, kernel_size=(4, 1), padding="same",
                         kernel_regularizer=regularizers.l2(regularized))(out)
        else:
            out = Conv2D(filters=filters, kernel_size=(4, 1), padding="same")(out)
        if pool:
            out = MaxPool2D(pool_size=2, padding="same")(out)
        out = add([res, out])

        return out

    def __split_train_test(self):
        """
        segment the oringinal dataset into train test dataset with ratio 4:1
        """

        df = pd.read_excel(self.labels_path)
        # sort labels by session, then segment in ascending order
        df.sort_values(by=['Sessions', 'Segments'], inplace=True)
        labels = np.array(df['Overlap'])

        self.__x = images_loader(self.images_dir, self.channel)
        self.__y = labels_loader(self.labels_path)

        print(self.__x.shape)
        print(self.__y.shape)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        train_index, test_index = list(kf.split(self.__x, labels))[0]

        df = pd.read_excel(self.labels_path)
        # sort labels by session, then segment in ascending order
        df.sort_values(by=['Sessions', 'Segments'], inplace=True)

        train_df = df.iloc[train_index, :]
        test_df = df.iloc[test_index, :]
        self.__train_labels_path = self.labels_path[:-5] + 'Train.xlsx'
        self.__test_labels_path = self.labels_path[:-5] + 'Test.xlsx'
        train_df.to_excel(self.__train_labels_path, index=None)
        test_df.to_excel(self.__test_labels_path, index=None)

        names = os.listdir(self.images_dir)
        # sort images file by sessions, then segments in ascending order
        names.sort(key=lambda name: (int(name.segment('.')[0].segment('_')[0][1:3]), int(name.segment('.')[0].segment('_')[3])))
        # images_path = [os.path.join(self.images_dir, name) for name in names]

        self.__train_images_dir = self.images_dir + '_train'
        self.__test_images_dir = self.images_dir + '_test'
        if not os.path.isdir(self.__train_images_dir):
            os.mkdir(self.__train_images_dir)
        if not os.path.isdir(self.__test_images_dir):
            os.mkdir(self.__test_images_dir)

        for i in train_index:
            image_path = os.path.join(self.images_dir, names[i])
            src = cv.imread(image_path)
            os.chdir(self.__train_images_dir)
            cv.imwrite(names[i], src)
        for j in test_index:
            image_path = os.path.join(self.images_dir, names[j])
            src = cv.imread(image_path)
            os.chdir(self.__test_images_dir)
            cv.imwrite(names[j], src)

    def __augment_data(self, images_dir, labels_path):
        """
        augment images and labels based on the ratio of each class
        """
        self.__train_aug_labels_path = labels_path[:-5] + 'Aug.xlsx'
        self.__train_aug_images_dir = images_dir + '_augmented'
        self.__augmented = True
        if os.path.exists(labels_path[:-5] + 'Aug.xlsx') and os.path.isdir(images_dir + '_augmented'):
            print("Already augment")
            return
        self.augment_images(images_dir, labels_path, self.__train_aug_images_dir, self.__train_aug_labels_path)
        print("Augment the images and labels done")

    def __load_model(self):
        """
        :return: the pretrained model of the overlap detector stored in model path
        """
        assert self.__model_path is not None, 'You should train your initial model or populate an outer model into ' \
                                              'the overlap detector '
        if self.__weighted:
            pretrained_model = tf.keras.models.load_model(self.__model_path, custom_objects={
                'loss': weighted_categorical_crossentropy(self.__weights_factor)})
        else:
            pretrained_model = tf.keras.models.load_model(self.__model_path,
                                                          custom_objects={'loss': 'categorical_crossentropy'})
        return pretrained_model

    def __load_train_data(self, with_augmented=False):
        """
        load train data from train (aug)images directory and train (aug)label path
        :param with_augmented: whether load augmented data or not
        """
        load_images_dir = self.__train_images_dir
        load_labels_path = self.__train_labels_path
        if with_augmented:
            # check whether the augmented images and labels keep aligned with the original images and labels
            self.__augment_data(load_images_dir, load_labels_path)
            load_images_dir = self.__train_aug_images_dir
            load_labels_path = self.__train_aug_labels_path

        self.__x_train = images_loader(load_images_dir, self.channel)
        self.__y_train = labels_loader(load_labels_path)
        self.__weights_factor = self.cal_weighted_penalty(self.__y_train)

    def __load_test_data(self):
        """
        load test data from test image directory and test label path
        """
        self.__x_test = images_loader(self.__test_images_dir, self.channel)
        self.__y_test = labels_loader(self.__test_labels_path)

    def __ResBLSTM(self, input_shape, class_dim):
        input_features = Input(input_shape)
        net = Conv2D(filters=16, kernel_size=1, padding="same")(input_features)

        net = self.res_block(net, 32, pool=True)
        net = self.res_block(net, 32)
        net = self.res_block(net, 32)

        net = self.res_block(net, 64, pool=True)
        net = self.res_block(net, 64)
        net = self.res_block(net, 64)

        net = self.res_block(net, 128, pool=True)
        net = self.res_block(net, 128)
        net = self.res_block(net, 128)

        net = Lambda(lambda x: K.mean(x, axis=1))(net)
        # forward_layer = LSTM(256, return_sequences=True)
        # backward_layer = LSTM(256, activation='relu', return_sequences=True, go_backwards=True)
        forward_layer = LSTM(256)
        backward_layer = LSTM(256, go_backwards=True)
        net = Bidirectional(forward_layer, backward_layer=backward_layer)(net)
        # net = Flatten()(net)
        net = Dropout(0.25)(net)
        net = LeakyReLU()(net)
        net = Dense(class_dim, activation="softmax")(net)
        model = Model(inputs=input_features, outputs=net)

        return model

    def __train_model(self, x_train, y_train, x_validate, y_validate, input_shape, epochs, batch_size,
                      weights_factor=None):
        class_dim = y_train.shape[1]

        # first layer, the input shape equals to the shape of one image file, regardless of the batch size
        model = self.__ResBLSTM(input_shape, class_dim)

        loss = weighted_categorical_crossentropy(
            weights_factor) if weights_factor is not None else "categorical_crossentropy"

        model.compile(loss=loss, optimizer=Adadelta(lr=0.001),
                      metrics=[tf.keras.metrics.categorical_accuracy, tf.keras.metrics.AUC()])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        learning_rate_adjust = [
            CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
        ]

        # checkpoint storage path is a placeholder
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='D:\SpectrumAnalysis\ckt',
                                                        monitor='val_categorical_accuracy', verbose=1,
                                                        save_best_only=True,
                                                        mode='max', period=1)

        print(model.summary())

        history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate),
                            callbacks=[early_stopping, learning_rate_adjust, checkpoint],
                            batch_size=batch_size, epochs=epochs)
        return model, history

    def train_model(self, save_path, epochs, batch_size, weighted=False, augmented=False):
        """
        train the overlap detector for the first time
        :param save_path: path of model to store
        :param epochs: number of epochs to train
        :param batch_size: number of batch for each parameter update
        :param weighted: specify use weighted categorical entropy loss function or not, False is set by default
        :param augmented: specify use augmented stratety or not, False is set by default
        """

        self.__load_train_data(augmented)
        self.__load_test_data()
        x_train, x_val, y_train, y_val = self.__x_train, self.__x_test, self.__y_train, self.__y_test  # assign test dataset to validation dataset
        # x_train, x_val, y_train, y_val = train_test_split(self.__train_x, self.__train_y, test_size=0.2, random_state=0,
        #                                                   stratify=self.__train_y)

        input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        print('input shape', x_train.shape)
        self.__weighted = weighted
        weights_factor = self.__weights_factor if weighted else None
        print('weight ', weights_factor)
        trained_model, history = self.__train_model(x_train, y_train, x_val, y_val, input_shape, epochs,
                                                    batch_size, weights_factor)

        run_model = tf.function(lambda x: trained_model(x))
        # FIX THE INPUT SIZE FOR TENSORFLOW LITE CONVERSION
        BATCH_SIZE = 1
        MELS = 128
        TIMESERIES = 151
        CHANNEL = 3

        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([BATCH_SIZE, MELS, TIMESERIES, CHANNEL], trained_model.input[0].dtype))
        trained_model.save(save_path, save_format='tf', signatures=concrete_func)

        self.__model_path = save_path

    def populate_model(self, load_model_path, weighted, augmented, images_dir, labels_path):
        """
        populate a pretrained model of the overlap detector will overwrite the current referred model, the pretrained model
        is trianed with the same images and labels
        :param labels_path: directory of images (input features)
        :param images_dir: path of labels
        :param load_model_path: the new pretrained model path
        :param weighted: specify whether the model to populate is weighted penalized or not
        :param augmented: specify whether the model to populate is augmented or not
        """

        self.__model_path = load_model_path
        self.__weighted = weighted
        self.__augmented = augmented
        self.images_dir = images_dir
        self.labels_path = labels_path
        self.__load_train_data(augmented)
        self.__load_test_data()

    def continue_train_model(self, save_path, epochs, batch_size):
        """
        continue training the model of the current osd, the weighted, augmented, train, test dataset\
        are unchanged
        :param save_path: new path to store the model
        :param epochs: training epoches
        :param batch_size: training batch size
        """
        pretrained_model = self.__load_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        learning_rate_adjust = [
            CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
        ]
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='D:\SpectrumAnalysis\ckt',
                                                        monitor='val_categorical_accuracy', verbose=1,
                                                        save_best_only=True,
                                                        mode='max', period=1)

        x_train, x_val, y_train, y_val = train_test_split(self.__x_train, self.__y_train, test_size=0.2, random_state=0,
                                                          stratify=self.__y_train)

        pretrained_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                             callbacks=[early_stopping, learning_rate_adjust, checkpoint],
                             batch_size=batch_size, epochs=epochs)

        if save_path is not None:
            pretrained_model.save(save_path, save_format='tf')
        else:
            pretrained_model.save(self.__model_path, save_format='tf')

        # return pretrained_model

    def evaluation(self, options):
        """
        evaluate the performance of model on test dataset
        """
        pretrained_model = self.__load_model()

        if options == 'test':
            x_eval, y_eval = self.__x_test, self.__y_test
        if options == 'val':
            _, x_eval, _, y_eval = train_test_split(self.__x_train, self.__y_train, test_size=0.2, random_state=0,
                                                    stratify=self.__y_train)

        predictions_class = np.argmax(pretrained_model.predict(x_eval), axis=1)
        predictions_prob = pretrained_model.predict(x_eval)

        confusion_matrix = np.zeros((3, 3))
        for i in range(predictions_prob.shape[0]):
            index_row = np.where(y_eval[i] == np.amax(y_eval[i]))
            index_col = int(predictions_class[i])
            confusion_matrix[index_row, index_col] += 1

        tp = confusion_matrix[2, 2]
        fn = confusion_matrix[2, 0] + confusion_matrix[2, 1]
        fp = confusion_matrix[0, 2] + confusion_matrix[1, 2]

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        # rows = ground truth; cols = predicted
        print(confusion_matrix)
        print('Overlapped recall: ', recall, " precision: ", precision)
