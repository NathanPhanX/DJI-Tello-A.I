# NOTE:
# TOTAL TRAINING LOOPS = EPOCH * SUB_EPOCH
# THE AMOUNT OF DATA IS TOO LARGE TO FIT THE MEMORY. THEREFORE, LOAD EACH PORTION OF DATA AND TRAIN SEPARATELY
# TRAINING_SPLIT PLUS TESTING_SPLIT MUST EQUAL 1

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import random
import time
import progressbar

from PIL import Image
from skimage.util import random_noise
from keras.layers import Conv2D, Input, UpSampling2D, Concatenate, LeakyReLU, Dropout
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import plot_model

model_path = "G:\\CODE\\Drone\\Model\\DeNoiseModel.h5"  # Path of the model
data_path = "G:\\CODE\\Drone\\DeNoiseData\\"  # Path of data
model_analysis_path = "G:\\CODE\\Drone\\Model\\DeNoiseModel.png"  # Path of the image of the model analysis
result_analysis_path = "G:\\CODE\\Drone\\Model\\DeNoiseAnalysis.png"  # Path of the image of the model analysis
noise_data_path = data_path + 'Noisy\\'  # Path of the data
original_data_path = data_path + 'Original\\'  # Path of the label
pre_process_video = False  # This variable decides whether or not to pre_process video or images
show_img = False  # This variable decides whether or not to show each image in load_data
display_train = False  # This variable decides whether or not to show images for training or testing in load_data. Only applies when show_img is True
predict_diff = False  # This variable decides whether or not load different images

width = 400  # Width of the image
height = 480  # Height of the image
channel = 3  # Channel of image. 3 for RGB and 1 for grayscale
epoch = 2  # Total number of epoch
sub_epoch = 2  # Total number of epoch used in the sub process
batch_size = 4  # Size of data in batches
learning_rate = 0.01  # Learning rate for the optimizer
decay = 0.001  # Decay rate for the optimizer
data_num = 50000  # Number of total data
fraction = 0.1  # Number of total data trained at a single time
training_split = 0.8  # The amount of data reserved for training
testing_split = 0.2  # The amount of data reserved for testing
patience = 10  # The number of epoch to wait before terminate training


# This function is used to pre_process data
def pre_process_data():
    video1 = cv2.VideoCapture(data_path + 'video1.mp4')
    video2 = cv2.VideoCapture(data_path + 'video2.mp4')
    path = glob.glob(data_path + 'Data\\*')
    random.shuffle(path)

    index = len(glob.glob(original_data_path + '*'))

    for i in range(index, data_num + index):
        if pre_process_video:
            video_choice = np.random.randint(0, 2)  # This variable decides which video to pick

            if video_choice == 0:
                video1.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, video1.get(cv2.CAP_PROP_FRAME_COUNT)))
                ret, frame = video1.read()
            else:
                video2.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, video2.get(cv2.CAP_PROP_FRAME_COUNT)))
                ret, frame = video2.read()
        else:
            frame = cv2.imread(path[i])

        num = np.random.randint(0, 6)  # This variable decides which noise method to add to image

        # Add noise to images
        if num == 0:
            img = random_noise(frame, mode='speckle', var=np.random.uniform(0.01, 0.2))
        elif num == 1:
            img = random_noise(frame, mode='s&p', amount=np.random.uniform(0.1, 0.3))
        elif num == 2:
            img = random_noise(frame, mode='gaussian', var=np.random.uniform(0.01, 0.2))
        elif num == 3:
            img = random_noise(frame, mode='pepper', amount=np.random.uniform(0.1, 0.3))
        else:
            img = cv2.blur(frame, (np.random.randint(5, 10), np.random.randint(5, 10)))
            img = np.array(img / 255.0)

        img = np.array(img * 255.0)
        img = cv2.resize(img, (width, height))
        cv2.imwrite(noise_data_path + str(i) + '.jpg', img)

        original = cv2.resize(frame, (width, height))
        cv2.imwrite(original_data_path + str(i) + '.jpg', original)


# This function loads data
def load_data(noise_path, label_path, is_predict=False):
    # Initialize variables
    x_train = np.zeros((int(len(noise_path) * training_split), width, height, channel), dtype='float16')
    y_train = np.zeros((int(len(label_path) * training_split), width, height, channel), dtype='float16')
    x_test = np.zeros((int(len(noise_path) * testing_split), width, height, channel), dtype='float16')
    y_test = np.zeros((int(len(label_path) * testing_split), width, height, channel), dtype='float16')

    # Load data and label for training
    for i in range(int(len(noise_path) * training_split)):
        x_train[i] = img_to_array(load_img(path=noise_path[i], color_mode='rgb', target_size=(width, height)))
        y_train[i] = img_to_array(load_img(path=label_path[i], color_mode='rgb', target_size=(width, height)))

    # Load data and label for validating
    for i in range(int(len(label_path) * (1 - training_split))):
        x_test[i] = img_to_array(load_img(path=noise_path[i + int(len(noise_path) * training_split)], color_mode='rgb', target_size=(width, height)))
        y_test[i] = img_to_array(load_img(path=label_path[i + int(len(label_path) * training_split)], color_mode='rgb', target_size=(width, height)))

    # Normalize training data
    x_train = x_train.astype('float16')
    x_train /= 255.0
    y_train = y_train.astype('float16')
    y_train /= 255.0

    # Normalize validating data
    x_test = x_test.astype('float16')
    x_test /= 255.0
    y_test = y_test.astype('float16')
    y_test /= 255.0

    if show_img:
        for i in range(100):
            if display_train:
                plt.imshow(np.concatenate(((x_train[i] * 255).astype('uint8'), (y_train[i] * 255).astype('uint8')), axis=1))
            else:
                plt.imshow(np.concatenate(((x_test[i] * 255).astype('uint8'), (y_test[i] * 255).astype('uint8')), axis=1))

            plt.show()

    if is_predict:
        return np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    else:
        return x_train, y_train, x_test, y_test


# This function builds layers in the denoise model
def block(input_layer):
    x1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
    pool1 = Conv2D(8, kernel_size=(5, 5), strides=(2, 2), padding='same')(x1)
    pool1 = LeakyReLU(0.2)(pool1)

    x2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool1)
    x2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x2)
    pool2 = Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same')(x2)
    pool2 = LeakyReLU(0.2)(pool2)

    x3 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool2)
    x3 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x3)
    pool3 = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same')(x3)
    pool3 = LeakyReLU(0.2)(pool3)

    x4 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool3)
    x4 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x4)
    pool4 = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x4)
    pool4 = LeakyReLU(0.2)(pool4)

    x5 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool4)

    x6 = UpSampling2D((2, 2))(x5)
    x6 = Concatenate()([x4, x6])
    x6 = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding='same')(x6)
    x6 = Conv2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x6)
    x6 = LeakyReLU(0.2)(x6)

    x7 = UpSampling2D((2, 2))(x6)
    x7 = Concatenate()([x3, x7])
    x7 = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding='same')(x7)
    x7 = Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding='same')(x7)
    x7 = LeakyReLU(0.2)(x7)

    x8 = UpSampling2D((2, 2))(x7)
    x8 = Concatenate()([x2, x8])
    x8 = Conv2D(16, kernel_size=(2, 2), strides=(1, 1), padding='same')(x8)
    x8 = Conv2D(16, kernel_size=(4, 4), strides=(1, 1), padding='same')(x8)
    x8 = LeakyReLU(0.2)(x8)

    x9 = UpSampling2D((2, 2))(x8)
    x9 = Concatenate()([x1, x9])
    x9 = Conv2D(8, kernel_size=(2, 2), strides=(1, 1), padding='same')(x9)
    x9 = Conv2D(8, kernel_size=(4, 4), strides=(1, 1), padding='same')(x9)
    x9 = LeakyReLU(0.2)(x9)

    x10 = Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(x9)
    x10 = Dropout(0.2)(x10)

    return x10


# This function builds the denoise model
def build_denoise_model():
    input_layer = Input((width, height, channel))
    layer = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    sub_layer_1a = block(layer)
    sub_layer_1b = block(layer)
    layer1 = Concatenate()([sub_layer_1a, sub_layer_1b])

    sub_layer_2a = block(layer)
    sub_layer_2b = block(layer)
    layer2 = Concatenate()([sub_layer_2a, sub_layer_2b])

    layer4 = Concatenate()([layer1, layer2])
    layer4 = UpSampling2D((2, 2))(layer4)
    layer4 = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same')(layer4)
    layer4 = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same')(layer4)

    layer4 = UpSampling2D((2, 2))(layer4)
    layer4 = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same')(layer4)
    layer4 = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same')(layer4)

    output_layer = Conv2D(channel, kernel_size=(2, 2), strides=(2, 2), padding='same')(layer4)
    output_layer = Conv2D(channel, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='sigmoid')(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    plot_model(model, model_analysis_path, show_shapes=True)
    return model


# This function plots model performance during training for further analysis
def plot_model_performance(result):
    fig = plt.figure()

    loss = fig.add_subplot(1, 1, 1)
    loss.plot(result.history['loss'], label='loss')
    loss.plot(result.history['val_loss'], label='val_loss')

    plt.savefig(result_analysis_path)


# This function calculates the learning rate
def learning_rate_calculation(lr, decay, epoch):
    result = '1: ' + str(lr) + '\n'

    for i in range(1, epoch):
        lr = lr * 1 / (1 + decay * i)
        result += str(i) + ': ' + str(lr) + '\n'

    print(result)


# This function trains the denoise model
def train():
    count = 0
    final_result = np.zeros((int(epoch / fraction), width, height, channel), dtype='uint8')

    model = build_denoise_model()  # Build model
    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=learning_rate, decay=decay, momentum=0.8, nesterov=True), metrics=['accuracy'])  # Compile model
    print(model.summary())

    # Get the path of noise and label data
    noise_path = glob.glob(noise_data_path + '*')
    label_path = glob.glob(original_data_path + '*')

    # Initialize progress bar
    bar = progressbar.ProgressBar(maxval=int(epoch / fraction), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    time.sleep(3)
    print('')

    # The reason for doing this is because all the data cannot be loaded at once due to memory constraint
    for training in range(epoch):
        print('\n===================================================================== EPOCH #' + str(training) + ' =====================================================================\n')

        # Shuffle data
        num = random.randint(0, 69420)
        random.Random(num).shuffle(noise_path)
        random.Random(num).shuffle(label_path)

        for i in range(int(1 / fraction)):
            x_train, y_train, x_test, y_test = load_data(noise_path[int(data_num * fraction * i): int(data_num * fraction * (i+1))], label_path[int(data_num * fraction * i): int(data_num * fraction * (i+1))])  # Load data

            # Train model
            result = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=sub_epoch,
                               callbacks=[ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'), EarlyStopping(monitor='val_loss', mode='min', patience=patience)])

            # Analyze result
            plot_model_performance(result)
            final_result[count] = img_to_array(load_img(path=result_analysis_path, color_mode='rgb', target_size=(width, height)))
            count += 1

            # Update progress
            bar.update(count)

    # Process and save the training result
    final_result = final_result.astype('uint8')
    analysis_img = final_result[0]

    for i in range(1, int(epoch / fraction)):
        analysis_img = np.concatenate((analysis_img, final_result[i]), axis=1)

    Image.fromarray(analysis_img, 'RGB').save(result_analysis_path)
    print('\nTRAINING FINISHED')


# This function predicts the result
def predict():
    random_num = []

    if predict_diff:
        num = 1
    else:
        num = 10

    # Get the path of noise and label data
    noise_path = glob.glob(noise_data_path + '*')
    label_path = glob.glob(original_data_path + '*')

    # Shuffle data
    num_rand = random.randint(0, 69420)
    random.Random(num_rand).shuffle(noise_path)
    random.Random(num_rand).shuffle(label_path)

    noise_path = noise_path[:num]
    label_path = label_path[:num]

    model = load_model(model_path)  # Load model
    data, label = load_data(noise_path, label_path, True)  # Load data
    print(np.shape(data))

    for i in range(num):
        random_num.append(random.randint(0, num))

    if predict_diff:
        data[0] = img_to_array(load_img(path="G:\\CODE\\Drone\Data\\0_0_-35_0_0.jpg", color_mode='rgb', target_size=(width, height)))
        data[0] = data[0].astype('float16')
        data[0] /= 255.0

    start = time.time()
    denoise = model.predict(data, batch_size=1)  # Denoise image
    stop = time.time()
    print('Average time per image: ' + str((stop - start) / num) + ' seconds')

    # Denormalize data
    noise = np.array(data * 255.0, dtype='uint8')
    denoise = np.array(denoise * 255.0, dtype='uint8')

    '''
    from PIL import ImageOps, Image
    for i in range(num):
        b = Image.fromarray(denoise[i])
        a = ImageOps.equalize(b)
        test = np.hstack((noise[i], denoise[i], a))
        cv2.imshow('', test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''

    # SHow result
    for i in range(num):
        if predict_diff:
            diff = noise[i] - denoise[i]
            display = np.hstack((denoise[i], diff))
        else:
            display = np.hstack((noise[i], denoise[i]))

        cv2.imshow('', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    print("Enter 1 to train model, 2 to denoise image, 3 to pre_process data, 4 to calculate learning rate decay after each epoch")

    while True:
        user_input = input('Enter: ')
        flag = True

        if user_input == '1':
            train()
        elif user_input == '2':
            predict()
        elif user_input == '3':
            pre_process_data()
        elif user_input == '4':
            learning_rate_calculation(learning_rate, decay, epoch)
        else:
            print('Invalid choice')
            flag = False

        if flag:
            break

    print('done')
