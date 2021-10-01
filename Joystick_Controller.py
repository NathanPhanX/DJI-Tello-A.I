import cv2
import librosa
import pygame
import sounddevice
from pygame.locals import *
import numpy as np
import time
import pygame.locals
import pygame.joystick as js
import pandas as pd
import matplotlib.image as save

from PIL import Image
from python_speech_features import logfbank, mfcc
from scipy.io.wavfile import write
from djitellopy import Tello
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class FrontEnd(object):

    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
    """

    def __init__(self):
        # Model for voice control
        self.model = load_model("G:\\CODE\\Drone\\Model\\VoiceControlModel.h5")

        # Info for voice control
        self.voice_height = 199
        self.voice_width = 199
        self.sample_rate = 44100
        self.second = 2
        self.move_distance = 20  # The unit is cm

        # Init pygame
        pygame.init()

        # Height and width of display
        self.display_width = 1920
        self.display_height = 1080

        # This variable enables VR mode
        self.vr = False

        # Init joystick
        js.init()
        self.device = js.Joystick(0)
        self.device.init()

        # This variable determines the total joystick velocity
        self.factor = 0.5

        # This variable determines the threshold for joystick
        self.thres = 0.1

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([self.display_width, self.display_height])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone speed
        self.S = 30

        # Frame per second
        self.FPS = 30

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # Image height and width and channel for the model prediction
        self.img_height = 299
        self.img_width = 299
        self.channel = 1

        # Information for the model prediction
        self.left_right_label = [-1, -30, -20, -10, 0, 10, 20, 30]
        self.up_down_label = [-1, -30, -20, -10, 0, 10, 20, 30]
        self.for_back_label = [-1, -20, 0, 20]
        self.ai = False

        # Path of the data for deep learning
        self.path = "/Users/nhanphan/Desktop/Code/DroneLearning/Data1/"

        self.index = -50
        self.barometer = ''
        self.altitude = ''
        self.height = ''
        self.distance = ''

        # Check if the video is recorded
        self.record = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

    # This function aims to reduce the 'dead' sound which means the sound under the threshold
    @staticmethod
    def envelop(y, rate, threshold):
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()

        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)

        return mask

    def run(self):
        predict_data = np.zeros((1, self.voice_height, self.voice_width, 3))
        predict_signal = np.zeros((1, self.voice_height, self.voice_width))

        # Determine the speed mode of the drone
        while True:
            user_input = input('Set mode (1 for beginner, 2 for expert): ')

            if user_input == '1':
                self.S = 50
            elif user_input == '2':
                self.S = 100
                self.factor = 0.9

            if user_input == '1' or user_input == '2':
                break

        vr_enable = input('Enter 1 to enable VR mode: ')

        if vr_enable == '1':
            self.vr = True

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        print('Battery: ' + str(self.tello.get_battery()))
        input('Start?')
        frame_read = self.tello.get_frame_read()
        model = load_model("G:\\CODE\\Drone\\Model\\DroneAI.h5")

        should_stop = False
        while not should_stop:
            if self.ai:
                image = frame_read.frame

                cv2.imwrite("G:\\CODE\\Drone\\test.jpg", image)
                image = load_img("G:\\CODE\\Drone\\test.jpg", color_mode="grayscale", target_size=(self.img_height, self.img_width))

                # Convert the image into array
                image = img_to_array(image)

                # Reshape the image
                image = image.reshape(1, self.img_height, self.img_width, self.channel)

                # Prepare the data
                image = image.astype('float16')
                image = image / 255.0
                predict_left_right, predict_up_down, predict_for_back = model.predict(image)

                self.left_right_velocity = self.left_right_label[int(np.argmax(predict_left_right))]
                self.up_down_velocity = self.up_down_label[int(np.argmax(predict_up_down))]
                self.for_back_velocity = self.for_back_label[int(np.argmax(predict_for_back))]

                if self.left_right_velocity == -1:
                    self.left_right_velocity = 0

                if self.up_down_velocity == -1:
                    self.up_down_velocity = 0

                if self.for_back_velocity == -1:
                    self.for_back_velocity = 0

                self.update()
                self.left_right_velocity = 0
                self.up_down_velocity = 0
                self.for_back_velocity = 0

            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_stop = True
                    elif event.key == K_v:
                        self.record = True
                    elif event.key == K_g and self.record:
                        self.record = False
                        self.tello.stop_video_capture()
                    elif event.key == K_f:
                        if self.ai:
                            self.ai = False
                        else:
                            self.ai = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)
                elif event.type == pygame.JOYBUTTONDOWN:
                    record_audio = sounddevice.rec(int(self.second * self.sample_rate), samplerate=self.sample_rate, channels=2)
                    sounddevice.wait()
                    write('G:\\CODE\\Drone\\predict.wav', rate=self.sample_rate, data=record_audio)

                    signal, rate = librosa.load(path='G:\\CODE\\Drone\\predict.wav', sr=self.sample_rate)
                    mask = self.envelop(signal, rate, 0.0001)
                    signal = signal[mask]

                    # Determine the filter bank and mel frequency
                    bank = logfbank(signal, self.sample_rate, nfilt=52, nfft=1103)
                    mel = mfcc(signal, self.sample_rate, numcep=52, nfilt=52, nfft=1103)

                    # Get signal data
                    data = np.concatenate((bank, mel), axis=1)
                    data = data.flatten()

                    if len(data) > self.voice_height * self.voice_width:
                        new_data = data[:self.voice_height * self.voice_width]
                    else:
                        extra_data = np.zeros((self.voice_height * self.voice_width - len(data)))
                        new_data = np.concatenate((data, extra_data))

                    new_data = new_data.astype('float32')
                    max_data = max(new_data)
                    min_data = min(new_data)

                    new_data = (new_data - min_data) / (max_data - min_data)  # Normalize data
                    predict_signal[0] = new_data.reshape((self.voice_height, self.voice_width))

                    # Save the processed data
                    save.imsave('G:\\CODE\\Drone\\predict_a.png', bank, cmap='hot')
                    img1 = Image.open('G:\\CODE\\Drone\\predict_a.png')
                    save.imsave('G:\\CODE\\Drone\\predict_b.png', mel, cmap='hot')
                    img2 = Image.open('G:\\CODE\\Drone\\predict_b.png')
                    img3 = np.concatenate((img1, img2), axis=1)
                    save.imsave('G:\\CODE\\Drone\\predict.png', img3, cmap='hot')

                    # Load, resize, and save the final image
                    img = Image.open('G:\\CODE\\Drone\\predict.png')
                    img = img.resize((self.voice_height, self.voice_width), Image.ANTIALIAS)
                    img.save('G:\\CODE\\Drone\\predict.png', cmap='hot')

                    # Load and prepare data
                    predict_data[0] = img_to_array(load_img(path='G:\\CODE\\Drone\\predict.png', color_mode='rgb', target_size=(self.voice_height, self.voice_width)))

                    predict_data = predict_data.astype('float32')
                    predict_data /= 255.0

                    # Make prediction
                    model_prediction = model.predict([predict_data, predict_signal], batch_size=1)
                    result = np.argmax(model_prediction[0])

                    if result == 1:
                        self.tello.takeoff()
                    elif result == 2:
                        self.tello.move_back(self.move_distance)
                    elif result == 3:
                        self.tello.move_forward(self.move_distance)
                    elif result == 4:
                        self.tello.move_left(self.move_distance)
                    elif result == 5:
                        self.tello.move_right(self.move_distance)
                    elif result == 6:
                        self.tello.land()
                    elif result == 7:
                        self.tello.rotate_clockwise(360)
                    elif result == 8:
                        self.tello.rotate_counter_clockwise(90)
                    elif result == 9:
                        self.tello.rotate_clockwise(90)
                    elif result == 10:
                        self.factor = 0.9
                    elif result == 11:
                        self.factor = 0.5
                    else:
                        pass

                elif event.type == pygame.JOYAXISMOTION:
                    self.up_down_velocity = -self.device.get_axis(3) * self.factor
                    self.yaw_velocity = self.device.get_axis(2) * self.factor
                    self.for_back_velocity = -self.device.get_axis(1) * self.factor
                    self.left_right_velocity = self.device.get_axis(0) * self.factor

                    if 0 < self.up_down_velocity < self.thres or 0 > self.up_down_velocity > -self.thres:
                        self.up_down_velocity = 0
                    else:
                        self.up_down_velocity *= 100

                    if 0 < self.yaw_velocity < self.thres or 0 > self.yaw_velocity > -self.thres:
                        self.yaw_velocity = 0
                    else:
                        self.yaw_velocity *= 100

                    if 0 < self.for_back_velocity < self.thres or 0 > self.for_back_velocity > -self.thres:
                        self.for_back_velocity = 0
                    else:
                        self.for_back_velocity *= 100

                    if 0 < self.left_right_velocity < self.thres or 0 > self.left_right_velocity > -self.thres:
                        self.left_right_velocity = 0
                    else:
                        self.left_right_velocity *= 100

                    self.left_right_velocity = int(self.left_right_velocity)
                    self.for_back_velocity = int(self.for_back_velocity)
                    self.up_down_velocity = int(self.up_down_velocity)
                    self.yaw_velocity = int(self.yaw_velocity)

                    self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)

            if self.index > 1000:
                self.tello.stop_video_capture()
                self.tello.land()
                break

            if frame_read.stopped:
                frame_read.stop()
                break

            if self.record:
                self.index += 1

            if self.index > 0:
                cv2.imwrite(self.path + "unknown" + str(self.index) + ".png", frame_read.frame)

            self.screen.fill([0, 0, 0])
            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.display_width, self.display_height))
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))

            if self.vr:
                self.screen.blit(frame, (int(-self.display_width / 2), 0))

            pygame.display.update()

            time.sleep(1 / self.FPS)

        # Call it always before finishing. I deallocate resources.
        print(self.barometer)
        print(self.altitude)
        print(self.distance)
        print(self.height)
        self.record = False
        self.index = 0
        self.tello.end()
        return

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_w:  # set forward velocity
            self.for_back_velocity = self.S
        elif key == pygame.K_s:  # set backward velocity
            self.for_back_velocity = -self.S
        elif key == pygame.K_a:  # set left velocity
            self.left_right_velocity = -self.S
        elif key == pygame.K_d:  # set right velocity
            self.left_right_velocity = self.S
        elif key == pygame.K_UP:  # set up velocity
            self.up_down_velocity = self.S
        elif key == pygame.K_DOWN:  # set down velocity
            self.up_down_velocity = -self.S
        elif key == pygame.K_LEFT:  # set yaw counter clockwise velocity
            self.yaw_velocity = -self.S
        elif key == pygame.K_RIGHT:  # set yaw clockwise velocity
            self.yaw_velocity = self.S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_w or key == pygame.K_s:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_UP or key == pygame.K_DOWN:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


# Throttle is axis 3
# Turn is axis 2
# Forward_Backward is axis 1
# Left_Right is axis 0
def test():
    pygame.init()
    js.init()

    device = js.Joystick(0)
    device.init()

    print('Initialization Completed')
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                # print(event.axis)

                for i in range(4):
                    value = device.get_axis(i)

                    if 0 < value < 0.1 or 0 > value > -0.1:
                        value = 0

                    print('Axis ' + str(i) + ': ' + str(value))


if __name__ == '__main__':
    main()
