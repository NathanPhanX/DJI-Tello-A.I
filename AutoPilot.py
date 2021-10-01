import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import glob
import h5py
import cv2

from keras.layers import Conv2D, Input, UpSampling2D, Concatenate, LeakyReLU, Dropout
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model

path_to_depth = "NavigationData/nyu_depth_v2_labeled.mat"  # Original data path
data_path = "NavigationData/Data/"  # Data path
label_path = "NavigationData/Label/"  # Label path
model_path = "Model/AutoPilot.h5"  # Model path
result_analysis_path = "Model/AutoPilotAnalysis.png"  # Path for the analyzed result
predict_path = "NavigationData/predict.jpg"  # Path for the predicted image

width = 400  # Width of the image
height = 480  # Height of the image
channel = [3, 1]  # Channel of image. 3 for RGB and 1 for grayscale. Data will be RGB and label will be grayscale

training_split = 0.8  # The amount of data reserved for training
batch_size = 20  # Size of data in batches
epoch = 100  # Total number of training
patience = 20  # This value is used in early stopping. It means the number of epoch to wait after there is no improvement in the model
debug = False


# This function preprocess the original data from NYU Depth V2 dataset
def preprocess_original_data():
    # read mat file
    file = h5py.File(path_to_depth, mode='r')

    for i in range(len(file['images'])):
        # read image. original format is [3 x 640 x 480], uint8
        img = file['images'][i]

        # reshape data and convert its data_type
        data = np.empty([480, 640, 3])
        data[:, :, 0] = img[0, :, :].T
        data[:, :, 1] = img[1, :, :].T
        data[:, :, 2] = img[2, :, :].T
        data = data.astype('uint8')

        # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
        depth = file['depths'][i]

        # reshape label and convert its data_type
        label = np.empty([480, 640, 3])
        label[:, :, 0] = depth[:, :].T
        label[:, :, 1] = depth[:, :].T
        label[:, :, 2] = depth[:, :].T
        label = (label / int(label[:, :, :].max() + 1)) * 255.0
        label = label.astype('uint8')

        # Resize data and label
        data = cv2.resize(data, (width, height))
        label = cv2.resize(label, (width, height))

        # Save data and label
        io.imsave(data_path + str(i) + '.jpg', data)
        io.imsave(label_path + str(i) + '.jpg', label)


# This function loads data
def load_data(is_predict=False):
    # Get path
    data_path = glob.glob('NavigationData/Data/*')
    label_path = glob.glob('NavigationData/Label/*')

    # Initialize variables
    testing_split = 1 - training_split
    x_train = np.zeros((int(len(data_path) * training_split), width, height, channel[0]), dtype='float16')
    y_train = np.zeros((int(len(label_path) * training_split), width, height, channel[1]), dtype='float16')
    x_test = np.zeros((int(len(data_path) * testing_split), width, height, channel[0]), dtype='float16')
    y_test = np.zeros((int(len(label_path) * testing_split), width, height, channel[1]), dtype='float16')

    # Load data and label for training
    for i in range(int(len(data_path) * training_split)):
        x_train[i] = img_to_array(load_img(path=data_path[i], color_mode='rgb', target_size=(width, height)))
        y_train[i] = img_to_array(load_img(path=label_path[i], color_mode='grayscale', target_size=(width, height)))

    # Load data and label for validating
    for i in range(int(len(label_path) * (1 - training_split))):
        x_test[i] = img_to_array(load_img(path=data_path[i + int(len(data_path) * training_split)], color_mode='rgb', target_size=(width, height)))
        y_test[i] = img_to_array(load_img(path=label_path[i + int(len(label_path) * training_split)], color_mode='grayscale', target_size=(width, height)))

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

    if debug:
        print('\n')
        # print(str(x_train[:, :, :, :].max()) + ' ' + str(x_test[:, :, :, :].max()) + ' ' + str(y_train[:, :, :, :].max()) + ' ' + str(y_test[:, :, :, :].max()))
        # print(str(x_train[:, :, :, :].min()) + ' ' + str(x_test[:, :, :, :].min()) + ' ' + str(y_train[:, :, :, :].min()) + ' ' + str(y_test[:, :, :, :].min()))
        print('x_train shape: ' + str(np.shape(x_train)))
        print('x_test shape: ' + str(np.shape(x_test)))
        print('y_train shape: ' + str(np.shape(y_train)))
        print('y_test shape: ' + str(np.shape(y_test)))

    if is_predict:
        return np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    else:
        return x_train, y_train, x_test, y_test


# This function builds model
def build_model():
    input_layer = Input((width, height, channel[0]))  # Input layer

    # The architecture of model is similar to the U-Net Model
    x1 = Conv2D(8, kernel_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    x1 = Conv2D(8, kernel_size=(2, 2), strides=(1, 1), padding='same')(x1)
    pool1 = Conv2D(8, kernel_size=(4, 4), strides=(2, 2), padding='same')(x1)
    pool1 = LeakyReLU(0.2)(pool1)

    x2 = Conv2D(16, kernel_size=(2, 2), strides=(1, 1), padding='same')(pool1)
    x2 = Conv2D(16, kernel_size=(2, 2), strides=(1, 1), padding='same')(x2)
    pool2 = Conv2D(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(x2)
    pool2 = LeakyReLU(0.2)(pool2)

    x3 = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding='same')(pool2)
    x3 = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding='same')(x3)
    pool3 = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(x3)
    pool3 = LeakyReLU(0.2)(pool3)

    x4 = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding='same')(pool3)
    x4 = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding='same')(x4)
    pool4 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x4)
    pool4 = LeakyReLU(0.2)(pool4)

    x5 = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding='same')(pool4)
    x5 = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding='same')(x5)
    x5 = LeakyReLU(0.2)(x5)
    x5 = Dropout(0.2)(x5)

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

    x10 = UpSampling2D((2, 2))(x9)
    x10 = Conv2D(4, kernel_size=(2, 2), strides=(1, 1), padding='same')(x10)
    x10 = Conv2D(4, kernel_size=(2, 2), strides=(1, 1), padding='same')(x10)
    x10 = Conv2D(4, kernel_size=(4, 4), strides=(2, 2), padding='same')(x10)

    output_layer = Conv2D(channel[1], kernel_size=(1, 1), strides=(1, 1), padding='same')(x10)
    output_layer = Conv2D(channel[1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(output_layer)

    # Create and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', metrics=['accuracy', 'mse'], optimizer='adam')  # SGD(learning_rate=learning_rate, decay=decay, momentum=0.8, nesterov=True))

    if debug:
        print(model.summary())

    return model


# This function plot the performance of model during training for further analysis
def plot_model_performance(result):
    fig = plt.figure()

    loss = fig.add_subplot(2, 1, 1)
    loss.plot(result.history['loss'], label='loss')
    loss.plot(result.history['val_loss'], label='val_loss')

    acc = fig.add_subplot(2, 1, 2)
    acc.plot(result.history['accuracy'], label='accuracy')
    acc.plot(result.history['val_accuracy'], label='val_accuracy')

    plt.savefig(result_analysis_path)


# This function trains data
def train():
    model = build_model()  # Build model
    x_train, y_train, x_test, y_test = load_data()  # Load data

    # Train the model
    result = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epoch, verbose=2,
                       callbacks=[ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'), EarlyStopping(monitor='val_loss', mode='min', patience=patience)])

    # Analyze result
    plot_model_performance(result)


# This function predict result
def predict():
    data, label = load_data(is_predict=True)  # Load data
    model = load_model(model_path)  # Load model
    result = model.predict(data[:10])  # Model prediction

    # Display predicted image with original data and label
    for i in range(len(result)):
        sub_label = cv2.cvtColor((label[i] * 255.0).astype('uint8'), cv2.COLOR_GRAY2RGB).astype('float32') / 255.0
        sub_result = cv2.cvtColor((result[i] * 255.0).astype('uint8'), cv2.COLOR_GRAY2RGB).astype('float32') / 255.0
        display = np.concatenate((data[i], sub_label, sub_result), axis=1)
        display = (display * 255.0).astype('uint8')
        plt.imshow(display)
        plt.show()


# Main Function
if __name__ == '__main__':
    print("Enter 1 to train model, 2 to predict, 3 to pre_process data")

    while True:
        user_input = input('Enter: ')
        flag = True

        if user_input == '1':
            train()
        elif user_input == '2':
            predict()
        elif user_input == '3':
            preprocess_original_data()
        else:
            print('Invalid choice')
            flag = False

        if flag:
            break

    print('done')
