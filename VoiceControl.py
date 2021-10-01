import sounddevice
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as save
import pandas as pd
import glob
import random
import itertools

from graphics import GraphWin
from scipy.io.wavfile import write
from python_speech_features import logfbank, mfcc
from PIL import Image
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, Concatenate, Input, Activation, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, LSTM, TimeDistributed, Flatten
from keras.models import Model, load_model
from keras.utils import to_categorical, plot_model
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.regularizers import l2, l1_l2

data_path = "G:\\CODE\\Drone\\VoiceData\\"
process_data_path = "G:\\CODE\\Drone\\ProcessVoiceData\\"
sample_rate = 44100  # The rate at which the sample is taken. The value means 44.1kH for audio
second = 2  # The amount of time that the audio is recorded in seconds
num_data = 100  # Number of data per action
action_path = ['Background_Noise\\', 'Take_Off\\', 'Fly_Backward\\', 'Fly_Forward\\', 'Fly_Left\\', 'Fly_Right\\', 'Land\\', 'Spin\\', 'Turn_Left\\', 'Turn_Right\\', 'Full_Speed\\', 'Slow_Speed\\']  # The path to the action
model_path = "G:\\CODE\\Drone\\Model\\VoiceControlModel.h5"  # The path of the model
action_dict = {'Background_Noise': 0, 'Take_Off': 1, 'Fly_Backward': 2, 'Fly_Forward': 3, 'Fly_Left': 4, 'Fly_Right': 5, 'Land': 6, 'Spin': 7, 'Turn_Left': 8, 'Turn_Right': 9, 'Full_Speed': 10, 'Slow_Speed': 11}

height = 199  # Height of image
width = 199  # Width of image
channel = 3  # Channel of image. 3 for RGB and 1 for grayscale
batch_size = 32  # Size of data in batches
epoch = 300  # Total number of epoch
split = 0.2  # The percentage of data used for validation
factor = 5  # The number of times generated data is larger than raw data
learning_rate = 0.0005  # Learning rate for the optimizer
decay = 0.0001  # Decay rate for the optimizer
color_map = 'hot'  # value for cmap
l1_regulate = 0.000001  # This value is used to regularize to prevent over-fitting
l2_regulate = 0.00001  # This value is used to regularize to prevent over-fitting
patience = 20  # This value is used in early stopping. It means the number of epoch to wait after there is no improvement in the model


# This function allows user to collect sound data
def collect_data():
    for path in action_path:
        input('\nCurrent class: ' + path[:-1] + '.\nPress any key to continue')
        for i in range(num_data):
            print("Start recording")
            time.sleep(0.2)
            record_audio = sounddevice.rec(int(second * sample_rate), samplerate=sample_rate, channels=2)
            sounddevice.wait()
            write(data_path + path + str(i) + '.wav', rate=sample_rate, data=record_audio)
            print("Stop recording")
            time.sleep(1)


# This function aims to reduce the 'dead' sound which means the sound under the threshold
def envelop(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()

    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask


# This function plot data for further analysis
def plot_pre_process_data(data):
    num_row = 3
    num_col = 4
    i = 0

    fig, axes = plt.subplots(nrows=num_row, ncols=num_col, sharey='all', figsize=(7, 3))
    fig.suptitle('Filter Bank', size=10)

    for x in range(num_row):
        for y in range(num_col):
            axes[x, y].set_title(list(data.keys())[i], size=10)
            axes[x, y].imshow(list(data.values())[i], interpolation='nearest', cmap=color_map)
            i += 1

            if i >= len(list(data.keys())):
                break

    plt.show()
    plt.close()


# This function combines the audio files
def combine_audio_file():
    main_path = glob.glob(data_path + 'Speed\\*')
    secondary_path = glob.glob(data_path + 'Numbers\\*')

    for i in range(len(secondary_path)):
        sub_path = glob.glob(secondary_path[i] + '\\*')

        for j in range(len(main_path)):
            signal1, rate1 = librosa.load(path=main_path[j], sr=sample_rate)
            signal2, rate2 = librosa.load(path=sub_path[j], sr=sample_rate)

            combine = np.append(signal1, signal2)
            write(data_path + 'Speed' + str(i+1) + '\\' + str(j) + '.wav', rate=sample_rate, data=combine)


# This function pre_process data
def pre_process_data():
    data_list = {}  # A dictionary for filter bank

    for path in action_path:
        for i in range(num_data):
            # Get the audio signal and clean it
            signal, rate = librosa.load(path=data_path + path + str(i) + '.wav', sr=sample_rate)
            mask = envelop(signal, rate, 0.0001)
            signal = signal[mask]

            # Determine the filter bank
            bank = logfbank(signal, sample_rate, nfilt=52, nfft=1103)
            mel = mfcc(signal, sample_rate, numcep=52, nfilt=52, nfft=1103)

            # Save bank filter and mel frequency into 2 separate images
            save.imsave(process_data_path + path + str(i) + 'a.png', bank, cmap=color_map)
            img1 = Image.open(process_data_path + path + str(i) + 'a.png')
            save.imsave(process_data_path + path + str(i) + 'b.png', mel, cmap=color_map)
            img2 = Image.open(process_data_path + path + str(i) + 'b.png')

            # Combine both bank filter and mel frequency images
            img3 = np.concatenate((img1, img2), axis=1)
            save.imsave(process_data_path + path + str(i) + '.png', img3, cmap=color_map)
            data_list[path[:-1]] = img3

            # Load, resize, and save the final image
            img = Image.open(process_data_path + path + str(i) + '.png')
            img = img.resize((width, height), Image.ANTIALIAS)
            img.save(process_data_path + path + str(i) + '.png')

    plot_pre_process_data(data_list)


# Stem module of the Inception_ResNet_v2
def stem(x):
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)

    x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x2 = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)

    x = Concatenate(axis=3)([x1, x2])

    x3 = Conv2D(64, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x3 = Conv2D(96, kernel_size=(3, 3), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x3)

    x4 = Conv2D(64, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x4 = Conv2D(64, kernel_size=(7, 3), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x4)
    x4 = Conv2D(64, kernel_size=(1, 7), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x4)
    x4 = Conv2D(96, kernel_size=(3, 3), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x4)

    x = Concatenate(axis=3)([x3, x4])

    x5 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    x6 = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)

    x = Concatenate(axis=3)([x5, x6])
    x = Activation('relu')(x)

    return x


# InceptionA module of the Inception_ResNet_v2
def InceptionA(x):
    x1 = Conv2D(32, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)

    x2 = Conv2D(32, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x2 = Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x2)

    x3 = Conv2D(32, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x3 = Conv2D(48, kernel_size=(3, 3), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x3)
    x3 = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x3)

    x4 = Concatenate(axis=3)([x1, x2, x3])
    x4 = Conv2D(384, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x4)

    inception_a = Concatenate(axis=3)([x, x4])
    inception_a = Activation('relu')(inception_a)

    return inception_a


# InceptionB module of the Inception_ResNet_v2
def InceptionB(x):
    x1 = Conv2D(192, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)

    x2 = Conv2D(128, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x2 = Conv2D(160, kernel_size=(1, 7), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x2)
    x2 = Conv2D(192, kernel_size=(7, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x2)

    x3 = Concatenate(axis=3)([x1, x2])
    x3 = Conv2D(1154, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x3)

    inception_b = Concatenate(axis=3)([x, x3])
    inception_b = Activation('relu')(inception_b)

    return inception_b


# InceptionC module of the Inception_ResNet_v2
def InceptionC(x):
    x1 = Conv2D(192, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)

    x2 = Conv2D(192, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x2 = Conv2D(224, kernel_size=(1, 3), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x2)
    x2 = Conv2D(256, kernel_size=(3, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x2)

    x3 = Concatenate(axis=3)([x1, x2])
    x3 = Conv2D(2048, kernel_size=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x3)

    inception_c = Concatenate(axis=3)([x, x3])
    inception_c = Activation('relu')(inception_c)

    return inception_c


# ReductionA module of the Inception_ResNet_v2
def ReductionA(x):
    x1 = Conv2D(192, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x1 = Conv2D(224, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x1)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x1)

    x2 = Conv2D(192, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x1, x2, x3])
    x = Activation('relu')(x)
    return x


# ReductionB module of the Inception_ResNet_v2
def ReductionB(x):
    x1 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x1)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x1)

    x2 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x2)

    x3 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x3 = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x3)

    x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x1, x2, x3, x4])
    x = Activation('relu')(x)
    return x


# This function builds a recurrent model
def Recurrent_Model(input_data):
    x = LSTM(256, return_sequences=True, kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(input_data)
    x = LSTM(256, return_sequences=True, kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x = LSTM(256, return_sequences=True, kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x = Dropout(0.3)(x)

    x = TimeDistributed(Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate)))(x)
    x = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate)))(x)
    x = TimeDistributed(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate)))(x)
    x = TimeDistributed(Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate)))(x)

    x = Flatten()(x)

    return x


# This function builds the neural network based on the Inception_ResNet_v2
def InceptionResnetV2(input_data):
    input_layer = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(input_data)

    # The structure follows the structure of the Inception_ResNet_v2
    x = stem(input_layer)

    for i in range(4):
        x = InceptionA(x)

    x = ReductionA(x)
    x = Dropout(0.3)(x)

    for i in range(7):
        x = InceptionB(x)

    x = ReductionB(x)
    x = Dropout(0.3)(x)

    for i in range(3):
        x = InceptionC(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)

    x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_regulate, l2=l2_regulate), bias_regularizer=l2(l2_regulate))(x)

    return x


# This function builds a full complex neural network model
def build_model():
    input_ir2 = Input((height, width, channel))  # Input data for Inception Resnet V2
    input_regression = Input((height, width))  # Input data for Regression

    ir2_model = InceptionResnetV2(input_ir2)  # Build Inception Resnet V2 Model
    recurrent_model = Recurrent_Model(input_regression)  # Build Regression Model

    # Make output layer
    output = Concatenate()([ir2_model, recurrent_model])
    output = Dense(len(action_path), activation='softmax', activity_regularizer=l2(l2_regulate))(output)

    # Build model
    model = Model(inputs=[input_ir2, input_regression], outputs=output)
    return model


# This function loads data
def load_data():
    path = []
    train_path = []
    validate_path = []
    x_train = np.zeros((int(len(action_path) * num_data * (1 - split)), height, width, channel))
    y_train = []
    x_test = np.zeros((int(len(action_path) * num_data * split), height, width, channel))
    y_test = []
    signal_train = np.zeros((int(len(action_path) * num_data * (1 - split)), height, width, ))
    signal_test = np.zeros((int(len(action_path) * num_data * split), height, width, ))

    # Get the path to data
    for sub_path in action_path:
        for i in range(num_data):
            path.append(process_data_path + sub_path + str(i) + '.png')

        random.shuffle(path)
        train_path.extend(path[:int(len(path) * (1 - split))].copy())
        validate_path.extend(path[int(len(path) * (1 - split)):].copy())
        path.clear()

    # Shuffle data
    random.shuffle(train_path)
    random.shuffle(validate_path)

    # Load data and labels for training
    for i in range(len(train_path)):
        x_train[i] = img_to_array(load_img(path=train_path[i], color_mode='rgb', target_size=(height, width)))
        y_train.append(action_dict[train_path[i].split('\\')[4]])

    # Load data and labels for validating
    for i in range(len(validate_path)):
        x_test[i] = img_to_array(load_img(path=validate_path[i], color_mode='rgb', target_size=(height, width)))
        y_test.append(action_dict[validate_path[i].split('\\')[4]])

    # Load signal training data for recurrent model
    for i in range(len(train_path)):
        signal, rate = librosa.load(path=data_path + train_path[i].split('.')[0].split('\\')[-2] + '\\' + train_path[i].split('.')[0].split('\\')[-1] + '.wav', sr=sample_rate)
        bank = logfbank(signal, sample_rate, nfilt=52, nfft=1103)
        mel = mfcc(signal, sample_rate, numcep=52, nfilt=52, nfft=1103)

        data = np.concatenate((bank, mel), axis=1)
        data = data.flatten()

        if len(data) > height * width:
            new_data = data[:height * width]
        else:
            extra_data = np.zeros((height * width - len(data)))
            new_data = np.concatenate((data, extra_data))

        new_data = new_data.astype('float32')
        max_data = max(new_data)
        min_data = min(new_data)

        new_data = (new_data - min_data) / (max_data - min_data)  # Normalize data
        signal_train[i] = new_data.reshape((height, width))

    # Load signal testing data for recurrent model
    for i in range(len(validate_path)):
        signal, rate = librosa.load(path=data_path + validate_path[i].split('.')[0].split('\\')[-2] + '\\' + validate_path[i].split('.')[0].split('\\')[-1] + '.wav', sr=sample_rate)
        bank = logfbank(signal, sample_rate, nfilt=52, nfft=1103)
        mel = mfcc(signal, sample_rate, numcep=52, nfilt=52, nfft=1103)

        data = np.concatenate((bank, mel), axis=1)
        data = data.flatten()

        if len(data) > height * width:
            new_data = data[:height * width]
        else:
            extra_data = np.zeros((height * width - len(data)))
            new_data = np.concatenate((data, extra_data))

        new_data = new_data.astype('float32')
        max_data = max(new_data)
        min_data = min(new_data)

        new_data = (new_data - min_data) / (max_data - min_data)  # Normalize data
        signal_test[i] = new_data.reshape((height, width))

    # Convert data to single precision and normalize data
    x_train = x_train.astype('float32')
    x_train /= 255.0
    x_test = x_test.astype('float32')
    x_test /= 255.0
    signal_train = signal_train.astype('float32')
    signal_test = signal_test.astype('float32')

    # One-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test, signal_train, signal_test


# This function plot the performance of model during training for further analysis
def plot_model_performance(result):
    fig = plt.figure()

    loss = fig.add_subplot(2, 1, 1)
    loss.plot(result.history['loss'], label='loss')
    loss.plot(result.history['val_loss'], label='val_loss')

    acc = fig.add_subplot(2, 1, 2)
    acc.plot(result.history['accuracy'], label='accuracy')
    acc.plot(result.history['val_accuracy'], label='val_accuracy')

    plt.savefig("G:\\CODE\\Drone\\Model\\VoiceAnalysis.png")


# This function calculates the learning rate for the optimizer
def learning_rate_calculation(lr, decay, epoch):
    result = '1: ' + str(lr) + '\n'

    for i in range(1, epoch):
        lr = lr * 1 / (1 + decay * i)
        result += str(i) + ': ' + str(lr) + '\n'

    print(result)


# This function trains the model
def train():
    x_train, y_train, x_test, y_test, signal_train, signal_test = load_data()  # Load data
    model = build_model()  # Build model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(learning_rate=learning_rate, decay=decay, momentum=0.8, nesterov=True))  # learning_rate = learning_rate * (1 / (1 + decay * epoch))
    print(model.summary())
    plot_model(model, model_path.split('.')[0] + 'Analysis.png')

    result = model.fit(x=[x_train, signal_train], y=y_train, validation_data=([x_test, signal_test], y_test), batch_size=batch_size, epochs=epoch,
                       callbacks=[ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'), EarlyStopping(monitor='val_accuracy', mode='max', patience=patience)])

    plot_model_performance(result)


# This function evaluates the model
def evaluate():
    x_train, y_train, x_test, y_test, signal_train, signal_test = load_data()
    model = load_model(model_path)

    img_data = np.concatenate((x_train, x_test), axis=0)
    signal_data = np.concatenate((signal_train, signal_test), axis=0)
    label = np.concatenate((y_train, y_test), axis=0)

    loss, acc = model.evaluate([img_data, signal_data], label, batch_size=1, verbose=0)
    print('\nAccuracy: ' + str(int(acc * 1000) / 10) + '%')
    print('Loss: ' + str(int(loss * 1000) / 1000))


# This function create confusion matrix for further analysis
def create_confusion_matrix():
    x_train, y_train, x_test, y_test, signal_train, signal_test = load_data()
    model = load_model(model_path)
    predict1 = model.predict([x_train, signal_train], batch_size=1)
    predict2 = model.predict([x_test, signal_test], batch_size=1)
    predict = np.concatenate((predict1, predict2), axis=0)
    true_label = np.concatenate((y_train, y_test), axis=0)
    cm = confusion_matrix(np.argmax(true_label, axis=1), np.argmax(predict, axis=1))

    plt.imshow(cm, interpolation='nearest', cmap=color_map, aspect='auto')
    plt.title('Confusion Matrix')
    plt.colorbar()

    labels = []
    for label in action_path:
        labels.append(label.split('\\')[0])

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)

    thres = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thres else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# This function makes prediction
def prediction():
    gui = GraphWin()
    predict_data = np.zeros((1, height, width, channel))
    predict_signal = np.zeros((1, height, width))
    model = load_model(model_path)

    while True:
        key = gui.checkKey()

        if key:
            print(key)
            if key == 'Escape':
                break

            if key == 'c':
                print('Start Recording')
                record_audio = sounddevice.rec(int(second * sample_rate), samplerate=sample_rate, channels=2)
                sounddevice.wait()
                print('Stop Recording')
                write('G:\\CODE\\Drone\\predict.wav', rate=sample_rate, data=record_audio)

                signal, rate = librosa.load(path='G:\\CODE\\Drone\\predict.wav', sr=sample_rate)
                mask = envelop(signal, rate, 0.0001)
                signal = signal[mask]

                # Determine the filter bank and mel frequency
                bank = logfbank(signal, sample_rate, nfilt=52, nfft=1103)
                mel = mfcc(signal, sample_rate, numcep=52, nfilt=52, nfft=1103)

                # Get signal data
                data = np.concatenate((bank, mel), axis=1)
                data = data.flatten()

                if len(data) > height * width:
                    new_data = data[:height * width]
                else:
                    extra_data = np.zeros((height * width - len(data)))
                    new_data = np.concatenate((data, extra_data))

                new_data = new_data.astype('float32')
                max_data = max(new_data)
                min_data = min(new_data)

                new_data = (new_data - min_data) / (max_data - min_data)  # Normalize data
                predict_signal[0] = new_data.reshape((height, width))

                # Save the processed data
                save.imsave('G:\\CODE\\Drone\\predict_a.png', bank, cmap=color_map)
                img1 = Image.open('G:\\CODE\\Drone\\predict_a.png')
                save.imsave('G:\\CODE\\Drone\\predict_b.png', mel, cmap=color_map)
                img2 = Image.open('G:\\CODE\\Drone\\predict_b.png')
                img3 = np.concatenate((img1, img2), axis=1)
                save.imsave('G:\\CODE\\Drone\\predict.png', img3, cmap=color_map)

                # Load, resize, and save the final image
                img = Image.open('G:\\CODE\\Drone\\predict.png')
                img = img.resize((width, height), Image.ANTIALIAS)
                img.save('G:\\CODE\\Drone\\predict.png', cmap=color_map)

                # Load and prepare data
                predict_data[0] = img_to_array(load_img(path='G:\\CODE\\Drone\\predict.png', color_mode='rgb', target_size=(height, width)))

                predict_data = predict_data.astype('float32')
                predict_data /= 255.0

                # Make prediction
                model_prediction = model.predict([predict_data, predict_signal], batch_size=batch_size)
                print(model_prediction)
                result = np.argmax(model_prediction[0])

                # Interpret result
                for key, value in action_dict.items():
                    if value == result:
                        print(key)
                        break


if __name__ == '__main__':
    print("Enter 1 to collect and pre-process data, 2 to train the neural networks, 3 to evaluate model, 4 to predict result, 5 to calculate learning rate decay after each epoch, 6 to create confusion matrix")

    while True:
        user = input("Enter: ")
        flag = True

        if user == '1':
            collect_data()
            combine_audio_file()
            pre_process_data()
        elif user == '2':
            train()
        elif user == '3':
            evaluate()
        elif user == '4':
            prediction()
        elif user == '5':
            learning_rate_calculation(learning_rate, decay, epoch)
        elif user == '6':
            create_confusion_matrix()
        else:
            print('Invalid choice')
            flag = False

        if flag:
            break

    print('done')
