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

# https://www.youtube.com/watch?v=VWUgkcX_KoY
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn video to grayscale to save computation
    faces = detector(gray)
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)
        print(face)
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27: # if the esc key is pressed
        break
cap.release()
cv2.destroyAllWindows()

