# This file allows the user to control the mouse using eye movements and input from the Muse headset.
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np
import dlib
from pynput.mouse import Button, Controller
import time
import math

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


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

lerp = lambda p1, p2, t: p1 + t * (p2 - p1)
inv_lerp = lambda x, a, b : (x - a) / (b - a)
lndmark_get = lambda landmarks, idx : [landmarks.part(idx).x, landmarks.part(idx).y]

def movement_nose(landmarks, screen_pos):
    nose_points =  [30,31,51,35]
    nose_top, nose_left, nose_bottom, nose_right = [lndmark_get(landmarks, idx)  for idx in nose_points]
    nose_center = line_intersection([nose_top, nose_bottom], [nose_left, nose_right])
    
    hor_x = inv_lerp(nose_center[0], nose_right[0], nose_left[0])
    ver_y = inv_lerp(nose_center[1], nose_bottom[1], nose_top[1])

    cv2.putText(frame, f"({hor_x*100}%, {ver_y*100}%)", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 0), 2, cv2.LINE_AA)
    

    norm_x = inv_lerp(hor_x, 0.25, 0.65)
    norm_y = inv_lerp(ver_y, 0.3, 0.7)

    x = norm_x*screen_pos[0]
    y = norm_y*screen_pos[1]


    mouse.move(x, y, True, 0.1)


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
    count_chin = 0
    dt = 0.06
    intial_face_coor = None
    set_face_coor = False
    _, frame = cap.read()
    screen_width, screen_height, _ = frame.shape
    face_outline_points = list(range(17))
    face_outline_points = [30,31,51,35]
    # print(face_outline_points)
    z = 0
    face = None
    while True:
        prev = time.time()
        
        _, frame = cap.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn video to grayscale to save computation
        faces = detector(gray)

        if faces:
            face = faces[0]
            # face detection
        if face:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            face_coor =  (x1,y1,x2,y2)

            if x2 > x1 and y2 > y1 and not set_face_coor and face_coor:
                intial_face_coor = face_coor
                set_face_coor = True
                # print("Set face coordinates location:", intial_face_coor)
            
            cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)

            # nose detection
            landmarks = predictor(gray, face)
            poly_points = np.array([ [int(landmarks.part(idx).x), int(landmarks.part(idx).y)] for idx in face_outline_points])
            # poly_points = np.array([[100,100], [100,200], [200,200],[200,100]], np.int32)
            # print(poly_points)
            # print("landmarks.part(2)", landmarks.part(2))
            # print("landmarks.part(14)", landmarks.part(14))
            # print("landmarks.part(8): CHIN", landmarks.part(8))
            poly_points = poly_points.reshape(-1,1,2)
            cv2.fillPoly(frame, [poly_points], (255,255,255)) 
            # CHANGES****************************
            if(count_chin == 0): # calibrate where the chin is orginally so we can use how the chin moves to get a z-coordinate to calculate tilt
                start_point = landmarks.part(8)
                count_chin += 1
            z = landmarks.part(8) - start_point
            # CHANGES****************************

            left_point = (landmarks.part(31).x, landmarks.part(31).y)
            rotZ = math.atan(landmarks.part(31).y/landmarks.part(31).x)

            left_point_3D = (landmarks.part(31).x, landmarks.part(31).y, rotZ)

            right_point = (landmarks.part(35).x, landmarks.part(35).y)

            center_top = midpoint(landmarks.part(30), landmarks.part(30))
            center_bottom = midpoint(landmarks.part(33), landmarks.part(33))

            # test rotations
            rotZ = math.atan(landmarks.part(31).y/landmarks.part(31).x)
            # print("rotZ", rotZ)

            avg_point_x = int((left_point[0] + right_point[0] + center_top[0] + center_bottom[0]) / 4)
            avg_point_y = int((left_point[1] + right_point[1] + center_top[1] + center_bottom[1]) / 4)
            avg_point = (avg_point_x, avg_point_y)

            horizontal_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2) # create a horizontal line across the nose
            vertical_line = cv2.line(frame, center_top, (landmarks.part(51).x, landmarks.part(51).y), (0, 255, 0), 2)

            # make the starting position of the nose as the reference for all mouse movements
            if(count == 0):
                middle = avg_point #center_top
                count += 1

            
            # cv2.putText(frame, str((round(nose_pos[0],2), round(nose_pos[1],2))), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 
            #         1, (255, 255, 255), 2, cv2.LINE_AA)

            # https://stackoverflow.com/questions/49799057/how-to-draw-a-point-in-an-image-using-given-co-ordinate-with-python-opencv
            cv2.circle(frame, (middle[0],middle[1]), radius=0, color=(0, 0, 255), thickness=5)

            # movementV2(middle, avg_point, dt)  # center_top
            movement_nose(landmarks, (1535, 863))
            # movementV2(middle, avg_point, dt)  # center_top
        
        dt = time.time()-prev
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27: # if the esc key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
