import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import collections

from keras.layers import Conv2D, Concatenate, Input, Activation, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array

# These values are specified within the paper
height = 299
width = 299
channel = 1

# These values are used for hot-encode label
left_right_label = ['-1', '-50', '-35', '-20', '0', '20', '35', '50']
up_down_label = ['-1', '-50', '-35', '-20', '0', '20', '35', '50']
for_back_label = ['-1', '-35', '0', '35']

# Path of the AI model
model_path = "G:\\CODE\\Drone\\Model\\DroneAI.h5"


# Stem module of the Inception_ResNet_v2
def stem(x):
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x2 = Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x1, x2])

    x3 = Conv2D(64, kernel_size=(1, 1), padding='same')(x)
    x3 = Conv2D(96, kernel_size=(3, 3), padding='valid')(x3)

    x4 = Conv2D(64, kernel_size=(1, 1), padding='same')(x)
    x4 = Conv2D(64, kernel_size=(7, 3), padding='same')(x4)
    x4 = Conv2D(64, kernel_size=(1, 7), padding='same')(x4)
    x4 = Conv2D(96, kernel_size=(3, 3), padding='valid')(x4)

    x = Concatenate(axis=3)([x3, x4])

    x5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x6 = Conv2D(192, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x5, x6])
    x = Activation('relu')(x)
    return x


# InceptionA module of the Inception_ResNet_v2
def InceptionA(x):
    x1 = Conv2D(32, kernel_size=(1, 1), padding='same')(x)

    x2 = Conv2D(32, kernel_size=(1, 1), padding='same')(x)
    x2 = Conv2D(32, kernel_size=(3, 3), padding='same')(x2)

    x3 = Conv2D(32, kernel_size=(1, 1), padding='same')(x)
    x3 = Conv2D(48, kernel_size=(3, 3), padding='same')(x3)
    x3 = Conv2D(64, kernel_size=(3, 3), padding='same')(x3)

    x4 = Concatenate(axis=3)([x1, x2, x3])
    x4 = Conv2D(384, kernel_size=(1, 1), padding='same')(x4)

    inception_a = Concatenate(axis=3)([x, x4])
    inception_a = Activation('relu')(inception_a)

    return inception_a


# InceptionB module of the Inception_ResNet_v2
def InceptionB(x):
    x1 = Conv2D(192, kernel_size=(1, 1), padding='same')(x)

    x2 = Conv2D(128, kernel_size=(1, 1), padding='same')(x)
    x2 = Conv2D(160, kernel_size=(1, 7), padding='same')(x2)
    x2 = Conv2D(192, kernel_size=(7, 1), padding='same')(x2)

    x3 = Concatenate(axis=3)([x1, x2])
    x3 = Conv2D(1154, kernel_size=(1, 1), padding='same')(x3)

    inception_b = Concatenate(axis=3)([x, x3])
    inception_b = Activation('relu')(inception_b)

    return inception_b


# InceptionC module of the Inception_ResNet_v2
def InceptionC(x):
    x1 = Conv2D(192, kernel_size=(1, 1), padding='same')(x)

    x2 = Conv2D(192, kernel_size=(1, 1), padding='same')(x)
    x2 = Conv2D(224, kernel_size=(1, 3), padding='same')(x2)
    x2 = Conv2D(256, kernel_size=(3, 1), padding='same')(x2)

    x3 = Concatenate(axis=3)([x1, x2])
    x3 = Conv2D(2048, kernel_size=(1, 1), padding='same')(x3)

    inception_c = Concatenate(axis=3)([x, x3])
    inception_c = Activation('relu')(inception_c)

    return inception_c


# ReductionA module of the Inception_ResNet_v2
def ReductionA(x):
    x1 = Conv2D(192, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x1 = Conv2D(224, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x1)

    x2 = Conv2D(192, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x1, x2, x3])
    x = Activation('relu')(x)
    return x


# ReductionB module of the Inception_ResNet_v2
def ReductionB(x):
    x1 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x1)

    x2 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x2)

    x3 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x3 = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x3)

    x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x1, x2, x3, x4])
    x = Activation('relu')(x)
    return x


def sub_networks(x):
    x = Conv2D(256, kernel_size=(7, 7), strides=(4, 4), padding='same')(x)
    x = Conv2D(256, kernel_size=(7, 7), strides=(3, 3), padding='same')(x)
    x = Conv2D(256, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = Conv2D(256, kernel_size=(7, 7), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, kernel_size=(5, 5), strides=(3, 3), padding='same')(x)
    x = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)

    return x


# This function build the neural network based on the Inception_ResNet_v2
def InceptionResnetV2():
    input_data = Input((height, width, channel))  # Input data
    input_layer = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_data)

    # The structure follows the structure of the Inception_ResNet_v2
    x = stem(input_layer)

    for i in range(5):
        x = InceptionA(x)

    x = ReductionA(x)

    for i in range(10):
        x = InceptionB(x)

    x = ReductionB(x)

    for i in range(5):
        x = InceptionC(x)

    x = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Split the neural networks for multi-output with corresponding to the drone movements which are left_right, up_down, for_back. NOTE: turn is ignored
    # Build left_right model
    left_right = sub_networks(input_layer)
    left_right = Concatenate(axis=3)([x, left_right])
    left_right = GlobalAveragePooling2D()(left_right)
    left_right = Dropout(rate=0.25)(left_right)

    left_right = Dense(256, activation='relu')(left_right)
    left_right = Dense(126, activation='relu')(left_right)
    left_right = Dense(64, activation='relu')(left_right)
    left_right = Dense(len(left_right_label), activation='softmax', name='left_right')(left_right)

    # Build up_down model
    up_down = sub_networks(input_layer)
    up_down = Concatenate(axis=3)([x, up_down])
    up_down = GlobalAveragePooling2D()(up_down)
    up_down = Dropout(rate=0.25)(up_down)

    up_down = Dense(256, activation='relu')(up_down)
    up_down = Dense(126, activation='relu')(up_down)
    up_down = Dense(64, activation='relu')(up_down)
    up_down = Dense(len(up_down_label), activation='softmax', name='up_down')(up_down)

    # Build up_down model
    for_back = sub_networks(input_layer)
    for_back = Concatenate(axis=3)([x, for_back])
    for_back = GlobalAveragePooling2D()(for_back)
    for_back = Dropout(rate=0.25)(for_back)

    for_back = Dense(256, activation='relu')(for_back)
    for_back = Dense(128, activation='relu')(for_back)
    for_back = Dense(64, activation='relu')(for_back)
    for_back = Dense(len(for_back_label), activation='softmax', name='for_back')(for_back)

    # Build and compile the model
    model = Model(inputs=input_data, outputs=[left_right, up_down, for_back])
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, decay=0.0005), metrics=['accuracy'], weighted_metrics=['accuracy'])

    # This is used for analyzing the structure of the model
    plot_model(model, "G:\\CODE\\Drone\\Model\\DroneModel.png")
    print(model.summary())

    return model


# This function is used to balance the data
def balance_data():
    """
    Information is used to balance data
    left_right:   Counter({4.0: 43932, 3.0: 5586, 0.0: 5500, 2.0: 5465, 7.0: 5447, 1.0: 5436, 6.0: 5413, 5.0: 5406})
    up_down:      Counter({4.0: 43751, 3.0: 5541, 6.0: 5532, 0.0: 5500, 5.0: 5494, 2.0: 5478, 1.0: 5455, 7.0: 5434})
    for_back:     Counter({2.0: 65183, 1.0: 5973, 3.0: 5529, 0.0: 5500})
    """

    data_path = glob.glob('G:\\CODE\\Drone\\Data\\*.jpg')  # Get all the paths of the data
    path = []

    # This for loop go though each path of data and add more data is necessary
    for i in range(len(data_path)):
        labels = data_path[i].split('_')
        path.append(data_path[i])

        if labels[2] == '-1':
            path.append(data_path[i])

            if i % 5 == 0:
                path.append(data_path[i])

        if labels[4].split('.')[0] == '50' and labels[3] == '0' and labels[2] == '0':
            for j in range(62):
                path.append(data_path[i])

        if labels[4].split('.')[0] == '-50' and labels[3] == '0' and labels[2] == '0':
            for j in range(112):
                path.append(data_path[i])

        if labels[4].split('.')[0] == '-35' and labels[3] == '0' and labels[2] == '0':
            for j in range(7):
                path.append(data_path[i])

        if labels[4].split('.')[0] == '35' and labels[3] == '0' and labels[2] == '0':
            for j in range(9):
                path.append(data_path[i])

        if labels[4].split('.')[0] == '-20' and labels[3] == '0' and labels[2] == '0':
            for j in range(2):
                path.append(data_path[i])

        if labels[4].split('.')[0] == '20' and labels[3] == '0' and labels[2] == '0':
            for j in range(2):
                path.append(data_path[i])

        if labels[3] == '50' and labels[4].split('.')[0] == '0' and labels[2] == '0':
            for j in range(65):
                path.append(data_path[i])

        if labels[3] == '-50' and labels[4].split('.')[0] == '0' and labels[2] == '0':
            for j in range(91):
                path.append(data_path[i])

        if labels[3] == '35' and labels[4].split('.')[0] == '0' and labels[2] == '0':
            for j in range(29):
                path.append(data_path[i])

        if labels[3] == '-35' and labels[4].split('.')[0] == '0' and labels[2] == '0':
            for j in range(14):
                path.append(data_path[i])

        if labels[3] == '20' and labels[4].split('.')[0] == '0' and labels[2] == '0':
            for j in range(22):
                path.append(data_path[i])

        if labels[3] == '-20' and labels[4].split('.')[0] == '0' and labels[2] == '0':
            for j in range(10):
                path.append(data_path[i])

        if labels[2] == '35' and labels[4].split('.')[0] == '20':
            for j in range(5):
                path.append(data_path[i])

        if labels[2] == '35' and labels[4].split('.')[0] == '35':
            for j in range(8):
                path.append(data_path[i])

        if labels[2] == '-35' and labels[4].split('.')[0] == '-20':
            for j in range(3):
                path.append(data_path[i])

        if labels[2] == '-35' and labels[4].split('.')[0] == '-35':
            for j in range(3):
                path.append(data_path[i])

        if labels[2] == '35' and labels[4].split('.')[0] == '0' and labels[3] == '0':
            for j in range(4):
                path.append(data_path[i])

        if labels[2] == '-35' and labels[4].split('.')[0] == '0' and labels[3] == '0':
            for j in range(4):
                path.append(data_path[i])

    '''
    NOTE: this is used to debug and evauluate this function
    left_right = np.ones((len(path),))
    up_down = np.ones((len(path),))
    for_back = np.ones((len(path),))

    for i in range(len(path)):
        labels = path[i].split('_')

        left_right[i] = labels[4].split('.')[0]
        up_down[i] = labels[3]
        for_back[i] = labels[2]

    print(collections.Counter(left_right))
    print(collections.Counter(up_down))
    print(collections.Counter(for_back))
    print(len(path))
    '''

    random.shuffle(path)  # Shuffle the data path for better randomization in data
    return path


# This function loads data
def load_data(mode):
    # Mode 0 is for training and mode 1 is for evaluating
    if mode == 0:
        data_path = balance_data()
        validation_split = 0.2  # The ratio of total data that is reserved for validation. NOTE: The result have to be an integer
    else:
        data_path = glob.glob('G:\\CODE\\Drone\\Data\\*.jpg')
        validation_split = 0  # All of the data is used for validation function

    train_num = int(len(data_path) * (1 - validation_split))
    test_num = int(len(data_path) * validation_split)
    data_train = np.zeros((train_num, height, width, channel))
    data_test = np.zeros((test_num, height, width, channel))
    left_right_train = np.zeros((train_num, len(left_right_label)))
    left_right_test = np.zeros((test_num, len(left_right_label)))
    up_down_train = np.zeros((train_num, len(up_down_label)))
    up_down_test = np.zeros((test_num, len(up_down_label)))
    for_back_train = np.zeros((train_num, len(for_back_label)))
    for_back_test = np.zeros((test_num, len(for_back_label)))

    # Load data
    for i in range(len(data_path)):
        print(i)
        path = data_path[i]

        if i < train_num:
            data_train[i] = img_to_array(load_img(path=path, color_mode='grayscale', target_size=(height, width)))

            labels = path.split('_')

            for j in range(len(left_right_label)):
                if labels[4].split('.')[0] == left_right_label[j]:
                    left_right_train[i][j] = 1.0
                    break

            for j in range(len(up_down_label)):
                if labels[3] == up_down_label[j]:
                    up_down_train[i][j] = 1.0
                    break

            for j in range(len(for_back_label)):
                if labels[2] == for_back_label[j]:
                    for_back_train[i][j] = 1.0
                    break
        else:
            data_test[int(i - train_num)] = img_to_array(load_img(path=path, color_mode='grayscale', target_size=(height, width)))

            labels = path.split('_')

            for j in range(len(left_right_label)):
                if labels[4].split('.')[0] == left_right_label[j]:
                    left_right_test[int(i - train_num)][j] = 1.0
                    break

            for j in range(len(up_down_label)):
                if labels[3] == up_down_label[j]:
                    up_down_test[int(i - train_num)][j] = 1.0
                    break

            for j in range(len(for_back_label)):
                if labels[2] == for_back_label[j]:
                    for_back_test[int(i - train_num)][j] = 1.0
                    break

    # Convert all labels to float16
    left_right_train = left_right_train.astype('float16')
    left_right_test = left_right_test.astype('float16')
    up_down_train = up_down_train.astype('float16')
    up_down_test = up_down_test.astype('float16')
    for_back_train = for_back_train.astype('float16')
    for_back_test = for_back_test.astype('float16')

    # Convert all data to float16 and normalize it
    data_train = data_train.astype('float16')
    data_train /= 255.0
    data_test = data_test.astype('float16')
    data_test /= 255.0

    return data_train, data_test, left_right_train, left_right_test, up_down_train, up_down_test, for_back_train, for_back_test


# This function plot the history of loss and acc of the model during training
def plot_history(result):
    fig = plt.figure()

    loss = fig.add_subplot(2, 1, 1)
    loss.plot(result.history['loss'], label='loss')
    loss.plot(result.history['val_loss'], label='val_loss')

    acc = fig.add_subplot(2, 1, 2)
    acc.plot(result.history['left_right_accuracy'], label='left_right_accuracy')
    acc.plot(result.history['up_down_accuracy'], label='up_down_accuracy')
    acc.plot(result.history['for_back_accuracy'], label='for_back_accuracy')

    plt.savefig("G:\\CODE\\Drone\\Model\\Analysis.png")


# This function evaluates the model performance
def evaluating():
    model = load_model(model_path)  # Load model
    data_train, data_test, left_right_train, left_right_test, up_down_train, up_down_test, for_back_train, for_back_test = load_data(1)  # Load data
    score = 0  # Initialize the score
    
    # These values are used for evaluation. To interpret the result, the first index of the 2-dimensional array is the correct result from data and the second index of the array is the number of wrong predicted
    # result which index is corresponding to the index the labels
    left_right_evaluate = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0]]
    up_down_evaluate = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]
    for_back_evaluate = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]

    predict_left_right, predict_up_down, predict_for_back = model.predict(data_train)  # Let the model predict the result

    # Compare the predicted result to the known result
    for i in range(len(glob.glob('G:\\CODE\\Drone\\Data\\*.jpg'))):
        if left_right_train[i][np.argmax(predict_left_right[i])] == 1 and up_down_train[i][np.argmax(predict_up_down[i])] == 1 and for_back_train[i][np.argmax(predict_for_back[i])] == 1:
            score += 1
        else:
            if left_right_train[i][np.argmax(predict_left_right[i])] != 1:
                for j in range(len(left_right_label)):
                    if left_right_train[i][j] == 1:
                        left_right_evaluate[j][int(np.argmax(predict_left_right[i]))] += 1
                        break

            if up_down_train[i][np.argmax(predict_up_down[i])] != 1:
                for j in range(len(up_down_label)):
                    if up_down_train[i][j] == 1:
                        up_down_evaluate[j][int(np.argmax(predict_up_down[i]))] += 1
                        break

            if for_back_train[i][np.argmax(predict_for_back[i])] != 1:
                for j in range(len(for_back_label)):
                    if for_back_train[i][j] == 1:
                        for_back_evaluate[j][int(np.argmax(predict_for_back[i]))] += 1
                        break

    # Output the results for evaluation
    print(score)
    print('\nleft_right ')
    print(left_right_evaluate)
    print('\nup_down ')
    print(up_down_evaluate)
    print('\nfor_back ')
    print(for_back_evaluate)


# This function calculate the class weights for imbalanced data training
def compute_class_weights(left_right, up_down, for_back, num):
    left_right_weight = {}
    up_down_weight = {}
    for_back_weight = {}

    total_left_right = 0
    total_up_down = 0
    total_for_back = 0

    left_right_temp = []
    up_down_temp = []
    for_back_temp = []

    # Since all labels are one hot encoded, these for loop undo that
    for i in range(len(left_right)):
        for j in range(len(left_right[i])):
            if left_right[i][j] == 1:
                left_right_temp.append(j)

    for i in range(len(up_down)):
        for j in range(len(up_down[i])):
            if up_down[i][j] == 1:
                up_down_temp.append(j)

    for i in range(len(for_back)):
        for j in range(len(for_back[i])):
            if for_back[i][j] == 1:
                for_back_temp.append(j)

    # Count the repeated elements in the list
    left_right_count = collections.Counter(left_right_temp)
    up_down_count = collections.Counter(up_down_temp)
    for_back_count = collections.Counter(for_back_temp)

    # Find the total
    for i in range(len(left_right_label)):
        total_left_right += num - left_right_count[i]

    for i in range(len(up_down_label)):
        total_up_down += num - up_down_count[i]

    for i in range(len(for_back_label)):
        total_for_back += num - for_back_count[i]

    # Create class weights
    for i in range(len(left_right_label)):
        left_right_weight[i] = round((num - left_right_count[i]) / total_left_right, 5)

    for i in range(len(up_down_label)):
        up_down_weight[i] = round((num - up_down_count[i]) / total_up_down, 5)

    for i in range(len(for_back_label)):
        for_back_weight[i] = round((num - for_back_count[i]) / total_for_back, 5)

    return left_right_weight, up_down_weight, for_back_weight


# This function trains the model
def train():
    batch_size = 4  # Only multiple of 4 such as 4, 16, 64, 128, etc
    epoch = 6

    model = InceptionResnetV2()  # Build model
    data_train, data_test, left_right_train, left_right_test, up_down_train, up_down_test, for_back_train, for_back_test = load_data(0)  # Load data
    left_right_weight, up_down_weight, for_back_weight = compute_class_weights(left_right_train, up_down_train, for_back_train, len(data_train))  # Compute class weight

    # Train the model
    result = model.fit(x=data_train, y=[left_right_train, up_down_train, for_back_train], batch_size=batch_size, epochs=epoch, class_weight=[left_right_weight, up_down_weight, for_back_weight],
                       validation_data=(data_test, [left_right_test, up_down_test, for_back_test]))

    # Save the model and plot history
    model.save(filepath=model_path)
    print('model saved')
    plot_history(result)


if __name__ == '__main__':
    print('0 for training, 1 for evaluating')

    while True:
        try:
            user = int(input('Enter the mode: '))
            break
        except ValueError:
            print('Invalid mode')

    if user == 0:
        train()
    elif user == 1:
        evaluating()
