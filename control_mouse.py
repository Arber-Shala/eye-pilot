# This file allows the user to control the mouse using eye movements and input from the Muse headset.
import subprocess
import sys

# # https://stackoverflow.com/questions/12332975/how-can-i-install-a-python-module-within-code
# def install(package):
#     '''
#     This function installs required libraries in function runtime so judges don't have to install packages to run code
#     input: package
#     output: installs the package to local PC
#     '''
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# # install the mouse package
# install("mouse")
# install("opencv-python")
# install("numpy")
# install("cmake")
# install("dlib")
# install("pyglet")

# import packages
import mouse
import cv2
import numpy as np
import dlib 
from math import hypot
import pyglet 


# click left button of mouse
mouse.click('left')
print(mouse.get_position())

# https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv-p-1/
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # load in dataset that contains important locations on a face, such as location of eyes

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn video to grayscale to save computation
    faces = detector(gray)
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)
        
        # eye detection
        landmarks = predictor(gray, face)
        #print(landmarks.part(39)) # print coordinate of inner eye
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        horizontal_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2) # create a horizontal line across the eye
        vertical_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27: # if the esc key is pressed
        break
cap.release()
cv2.destroyAllWindows()

