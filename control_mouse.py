# This file allows the user to control the mouse using eye movements and input from the Muse headset.
import subprocess
import sys

# # https://stackoverflow.com/questions/12332975/how-can-i-install-a-python-module-within-code
def install(package):
    '''
    This function installs required libraries in function runtime so judges don't have to install packages to run code
    input: package
    output: installs the package to local PC
    '''
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install the mouse package
# install("mouse")
# install("opencv-python==4.10.0.84")
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
import time

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def movementV2(middle, new_position):
    x1, y1 = middle
    x2, y2 = new_position
    x_movement = x1 - x2
    y_movement = y2 - y1
    if abs(x_movement) >= 20 or abs(y_movement) >= 20:
        mouse.move(x_movement * 1.2, y_movement * 1.2, False, 0.2)
        # time.sleep(0.2)



if __name__ == "__main__":
    choice = ""
    choice = input("choose camera, front or back: ")
    if(choice == "front"):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    elif(choice == "back"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # load in dataset that contains important locations on a face

    count = 0
    while True:
        
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

            avg_point_x = int(left_point[0] + right_point[0] + center_top[0] + center_bottom[0] / 4)
            avg_point_y = int(left_point[1] + right_point[1] + center_top[1] + center_bottom[1] / 4)
            #print("avg_point_x", avg_point_x)
            avg_point = (avg_point_x, avg_point_y)

            horizontal_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2) # create a horizontal line across the nose
            vertical_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
            # make the starting position of the nose as the reference for all mouse movements
            if(count == 0):
                middle = avg_point #center_top
                count += 1
            print(center_top)
            print(middle)
            movementV2(middle, avg_point)  # center_top


        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27: # if the esc key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()


# while True:
#     ret, frame = cap.read()
#     roi = frame[269: 795, 537:1416]
#     rows, cols, _ = roi.shape
#     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 0)

#     _, threshold = cv2.threshold(gray_roi, 5, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
#     for contour in contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
#         cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
#         break
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1)
#     if key == 27: # if the esc key is pressed
#         break

# cap.release()
# cv2.destroyAllWindows()


# cascPath = sys.argv[1]
# noseCascade = cv2.CascadeClassifier()

# video_capture = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     nose = noseCascade.detectMultiScale(gray, 1.3, 5)
#     # Draw a rectangle around the nose
#     for (x, y, w, h) in nose:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()