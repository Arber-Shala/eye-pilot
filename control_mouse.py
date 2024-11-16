# This file allows the user to control the mouse using eye movements and input from the Muse headset.
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np
import dlib
from pynput.mouse import Button, Controller
import time


def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def movementV2(middle, new_position, dt):
    speed = 3
    deadzone = 17  # deadzone amt

    x1, y1 = middle
    x2, y2 = new_position

    x_movement = x1 - x2
    y_movement = y2 - y1
    mouse = Controller()

    if abs(x_movement) >= deadzone or abs(y_movement) >= deadzone:
        mouse.move(x_movement * speed * dt, y_movement * speed * dt)



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

    count = 0
    dt = 0.06
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
            avg_point = (avg_point_x, avg_point_y)

            horizontal_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2) # create a horizontal line across the nose
            vertical_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

            # make the starting position of the nose as the reference for all mouse movements
            if(count == 0):
                middle = avg_point #center_top
                count += 1

            
            #https://stackoverflow.com/questions/9734821/how-to-find-the-center-coordinate-of-rectangle
            frame = cv2.putText(frame, str(avg_point), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)

            # https://stackoverflow.com/questions/49799057/how-to-draw-a-point-in-an-image-using-given-co-ordinate-with-python-opencv
            cv2.circle(frame, (middle[0],middle[1]), radius=0, color=(0, 0, 255), thickness=5)


            movementV2(middle, avg_point, dt)  # center_top

        dt = time.time()-prev
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27: # if the esc key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
