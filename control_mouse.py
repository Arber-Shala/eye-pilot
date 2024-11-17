# This file allows the user to control the mouse using eye movements and input from the Muse headset.
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np
import dlib
import mouse
import time
from PyQt6 import QtCore, QtWidgets
from pynput.mouse import Button, Controller

from real_time_learning import hecatron
from real_time_learning.rlplot_live import MainLiveRL, MainLiveRLWindow

pynput_mouse = Controller()

def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def movementV2(middle, new_position, dt):
    deadzone = [] # square in the callibrated center where the mouse does not move
    speed = 4

    x1, y1 = middle
    x2, y2 = new_position

    x_movement = x1 - x2
    y_movement = y2 - y1

    print("x_movement", x_movement)
    print("y_movement", y_movement)
    if abs(x_movement) >= 20 or abs(y_movement) >= 20:
        pynput_mouse.move(x_movement * speed * dt, y_movement * speed * dt)
        # time.sleep(0.2)



def func():
    pass

def click():
    pynput_mouse.click(Button.left)


response = input("Do you want to use a connected board? (y/n)")
if response.lower() == 'y':
    response = input("Search for board? (y/n)")
    if response.lower() == 'y':
        board_id, serial_port = hecatron.find_port_and_id()
    else:
        serial_port = input("Serial Port: ")
        board_id = int(input("Board ID: "))
else:
    board_id = None
    serial_port = None
print("board_id:", board_id)
print("serial_port:", serial_port)

board = hecatron.init_board(serial_port, board_id)

if __name__ == "__main__":
    
    print("Pick a device id from the list")
    print("device id: device name")
    choice = ""
    for devid, devname in get_available_cameras().items():
        print(f"{devid}: {devname}")

    while not choice.isdigit() or choice == "":
        choice = input("choose video device id: ")
    choice = int(choice)
    cap = cv2.VideoCapture(choice, cv2.CAP_DSHOW)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # load in dataset that contains important locations on a face

    # eeg board and model

    eeg_model = MainLiveRLWindow(board, [func, click], "real_time_learning/models/best_model_yay") # model to make predictions
    app = QtWidgets.QApplication([])
    eeg_model.show()
    app.exec()

    count = 0
    dt = 0.06
    action_cooldown = 0.2 # cooldown between action predictions from RL model in seconds
    last_action_time = time.time()
    last_action = None

    while True:
        prev = time.time()
        
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn video to grayscale to save computation
        faces = detector(gray)

        for face in faces:
            # face detection
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)

            # nose detection
            landmarks = predictor(gray, face)


            left_point = (landmarks.part(31).x, landmarks.part(31).y)
            right_point = (landmarks.part(35).x, landmarks.part(35).y)
            center_top = midpoint(landmarks.part(30), landmarks.part(30))
            center_bottom = midpoint(landmarks.part(33), landmarks.part(33))

            avg_point_x = int((left_point[0] + right_point[0] + center_top[0] + center_bottom[0]) / 4)
            avg_point_y = int((left_point[1] + right_point[1] + center_top[1] + center_bottom[1]) / 4)
            #print("avg_point_x", avg_point_x)
            avg_point = (avg_point_x, avg_point_y)

            horizontal_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2) # create a horizontal line across the nose
            vertical_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
            # make the starting position of the nose as the reference for all mouse movements
            if(count == 0):
                middle = avg_point #center_top
                count += 1
            #https://stackoverflow.com/questions/9734821/how-to-find-the-center-coordinate-of-rectangle

            # middle_x = int((x1 + x2) / 2)
            # middle_y = int((y1 + y2) / 2)
            # middle = (middle_x, middle_y)


            print("avg_point", avg_point)
            print("middle:", middle)
            # draw a circle where the center point is
            # https://stackoverflow.com/questions/49799057/how-to-draw-a-point-in-an-image-using-given-co-ordinate-with-python-opencv
            cv2.circle(frame, (middle[0],middle[1]), radius=0, color=(0, 0, 255), thickness=5)


            movementV2(middle, avg_point, dt)  # center_top

        current_time = time.time()
        dt = current_time-prev
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27: # if the esc key is pressed
            break

        if (current_time - last_action_time > action_cooldown):
            selected_action = eeg_model.last_action

            print(selected_action)
            print("action selected")
            last_action_time = current_time
            last_action = selected_action

    cap.release()
    cv2.destroyAllWindows()
