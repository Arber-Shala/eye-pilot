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

# https://www.youtube.com/watch?v=kbdbZFT9NQI
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # load in dataset that contains important locations on a face, such as location of eyes

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    ret, frame = cap.read()
    roi = frame[269: 795, 537:1416]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 0)

    _, threshold = cv2.threshold(gray_roi, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27: # if the esc key is pressed
        break

cap.release()
cv2.destroyAllWindows()

