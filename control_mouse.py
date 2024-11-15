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

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27: # if the esc key is pressed
        break
cap.release()
cv2.destroyAllWindows()

