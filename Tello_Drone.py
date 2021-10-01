import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import pygame.locals

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
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

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

    def run(self):
        # Determine the speed mode of the drone
        while True:
            user_input = input('Set mode (1 for beginner, 2 for expert): ')

            if user_input == '1':
                self.S = 50
            else:
                if user_input == '2':
                    self.S = 100

            if user_input == '1' or user_input == '2':
                break

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

        print(str(self.tello.get_battery()))
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
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
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
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
