# This file allows the user to control the mouse using eye movements and input from the Muse headset.
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np
import dlib
import mouse
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
    speed = 5
    deadzone = 25  # deadzone amt

    x1, y1 = middle
    x2, y2 = new_position

    x_movement = x1 - x2
    y_movement = y2 - y1


    if abs(x_movement) >= deadzone or abs(y_movement) >= deadzone:
        mouse.move(x_movement * speed * dt, y_movement * speed * dt, False, 0.2)


# def movement_direct(middle, new_pos, init_face_rect,screen_width, screen_height, dt):
#     x1,y1,x2, y2 = init_face_rect
#     per_x, per_y = (new_pos[0]/abs(x2-x1), new_pos[1]/abs(y2-y1))
    
#     mouse.move(per_x*screen_width, per_y*screen_height, True, 0.1)


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
    intial_face_coor = None
    set_face_coor = False
    _, frame = cap.read()
    screen_width, screen_height, _ = frame.shape
    face_outline_points = list(range(17))
    print(face_outline_points)
    prev_face = None
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
                print("Set face coordinates location:", intial_face_coor)
            
            cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)

            # nose detection
            landmarks = predictor(gray, face)
            poly_points = np.array([ [int(landmarks.part(idx).x), int(landmarks.part(idx).y)] for idx in face_outline_points])
            # poly_points = np.array([[100,100], [100,200], [200,200],[200,100]], np.int32)
            # print(poly_points)
            poly_points = poly_points.reshape(-1,1,2)
            cv2.fillPoly(frame, [poly_points], (255,255,255)) 


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
            # movementV2(middle, avg_point, dt)  # center_top
        
        dt = time.time()-prev
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27: # if the esc key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
