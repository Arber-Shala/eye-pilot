# This file allows the user to control the mouse using eye movements and input from the Muse headset.
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np
import dlib
import mouse
import time
import math
import os

def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def movementV2(middle, new_position, dt):
    speed = 5
    deadzone = 25  # deadzone amt

    x1, y1 = middle
    x2, y2 = new_position

    x_movement = x1 - x2
    y_movement = y2 - y1


    if abs(x_movement) >= deadzone or abs(y_movement) >= deadzone:
        mouse.move(x_movement * speed * dt, y_movement * speed * dt, False, 0.2)


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

def average_position(positions):
    if not positions:
        raise ValueError("The list of positions is empty.")
    
    # Convert to a NumPy array for easier calculations
    positions_array = np.array(positions)
    
    # Calculate the mean along the first axis (for each coordinate)
    average = positions_array.mean(axis=0)
    
    # Convert back to a tuple
    return tuple(average)

lerp = lambda p1, p2, t: p1 + t * (p2 - p1)
inv_lerp = lambda x, a, b : (x - a) / (b - a)
lndmark_get = lambda landmarks, idx : [landmarks.part(idx).x, landmarks.part(idx).y]

def movement_nose(landmarks, screen_pos):
    key_points =  [30,31,51,35]
    top, left, bottom, right = [lndmark_get(landmarks, idx)  for idx in key_points]
    
    nose_center = line_intersection([top, bottom], [left, right])
    

    hor_x = inv_lerp(nose_center[0], right[0], left[0])
    ver_y = inv_lerp(nose_center[1], bottom[1], top[1])

    #calibration of the values
    norm_x = inv_lerp(hor_x, 0.25, 0.65)
    norm_y = inv_lerp(ver_y, 0.3, 0.7)

    cv2.putText(frame, f"({round(min(max(norm_x*100,0), 100))}%, {round(min(max(norm_y*100, 0), 100))}%)", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 0), 2, cv2.LINE_AA)
    
    poly_points = np.array([top, left, bottom, right]).reshape(-1,1,2)
    cv2.fillPoly(frame, [poly_points], (255,255,255))

    # Draw the cross to show
    cv2.line(frame, left, right, (0, 255, 0), 2)
    cv2.line(frame, top, bottom, (0, 255, 0), 2)

    x = norm_x*screen_pos[0]
    y = norm_y*screen_pos[1]


    return (x,y)


def exponential_moving_average(new_pos, prev_avg_pos=None, alpha=0.3):
    """
    Compute the exponential moving average for smoothing mouse movements.
    
    Parameters:
        new_pos (tuple): The current position (x, y).
        prev_avg_pos (tuple): The previous EMA position (x, y). If None, start with the current position.
        alpha (float): Smoothing factor (0 < alpha <= 1). Higher values are more responsive.
    
    Returns:
        tuple: The updated EMA position (x, y).
    """
    if prev_avg_pos is None:
        return new_pos  # Start with the first position if no previous average exists
    
    x_new, y_new = new_pos
    x_prev, y_prev = prev_avg_pos
    
    # Apply EMA formula
    x_avg = alpha * x_new + (1 - alpha) * x_prev
    y_avg = alpha * y_new + (1 - alpha) * y_prev
    
    return x_avg, y_avg

if __name__ == "__main__":    
    print("Pick a device id from the list")
    print("device id: device name")
    choice = ""
    for devid, devname in get_available_cameras().items():
        print(f"{devid}: {devname}")

    while not choice.isdigit() or choice == "":
        choice = input("choose video device id: ")
    choice = int(choice)
        
    if os.name == "nt":
        cap = cv2.VideoCapture(choice, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(choice)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # load in dataset that contains important locations on a face

    
    count = 0
    count_chin = 0
    dt = 0.06
    intial_face_coor = None
    set_face_coor = False
    _, frame = cap.read()
    screen_width, screen_height, _ = frame.shape
    z = 0
    face = None
    i = 0
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
            
            # z = landmarks.part(8) - start_point

            # left_point = (landmarks.part(31).x, landmarks.part(31).y)
            # rotZ = math.atan(landmarks.part(31).y/landmarks.part(31).x)

            # left_point_3D = (landmarks.part(31).x, landmarks.part(31).y, rotZ)

            # right_point = (landmarks.part(35).x, landmarks.part(35).y)

            # center_top = midpoint(landmarks.part(30), landmarks.part(30))
            # center_bottom = midpoint(landmarks.part(33), landmarks.part(33))

            # test rotations
            # rotZ = math.atan(landmarks.part(31).y/landmarks.part(31).x)
            # # print("rotZ", rotZ)

            # avg_point_x = int((left_point[0] + right_point[0] + center_top[0] + center_bottom[0]) / 4)
            # avg_point_y = int((left_point[1] + right_point[1] + center_top[1] + center_bottom[1]) / 4)
            # avg_point = (avg_point_x, avg_point_y)

            # make the starting position of the nose as the reference for all mouse movements
            # if(count == 0):
            #     middle = avg_point #center_top

            # # https://stackoverflow.com/questions/49799057/how-to-draw-a-point-in-an-image-using-given-co-ordinate-with-python-opencv
            # cv2.circle(frame, (middle[0],middle[1]), radius=0, color=(0, 0, 255), thickness=5)

            # movementV2(middle, avg_point, dt)  # center_top


            x,y = movement_nose(landmarks, (1535, 863))
            if (count == 0):
                prev_avg_pos = (x,y)

            prev_avg_pos = exponential_moving_average((x, y), prev_avg_pos, 0.25)

            mouse.move(*prev_avg_pos, True, 0.1)

            # movementV2(middle, avg_point, dt)  # center_top
        
        dt = time.time()-prev
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27: # if the esc key is pressed
            break
        count += 1

    cap.release()
    cv2.destroyAllWindows()
