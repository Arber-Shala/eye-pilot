# This file allows the user to control the mouse using eye movements and input from the Muse headset.
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np
import dlib
import mouse
import time
import math
import pyautogui 
from tkinter import *
from tkinter import colorchooser, ttk

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

def paint():
    # importing pyautogui 
    screenWidth, screenHeight = pyautogui.size() 
    print(screenWidth)
    
class paint:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'Black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.pen_width = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def paint(self, e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width = self.pen_width, fill = self.color_fg, capstyle='round', smooth = True)
        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):
        self.old_x = None
        self.old_y = None
    
    def changedW(self, width):
        self.pen_width = width
    
    def clearcanvas(self):
        self.c.delete(ALL)
    
    def change_fg(self):
        self.color_fg = colorchooser.askcolor(color=self.color_fg)[1]
    
    def change_bg(self):
        self.color_bg = colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg

    def drawWidgets(self):
        self.controls = Frame(self.master, padx=5, pady=5)
        textpw = Label(self.controls, text='Pen Width', font='Georgia 16')
        textpw.grid(row=0, column=0)
        self.slider = ttk.Scale(self.controls, from_=5, to=100, command=self.changedW, orient='vertical')
        self.slider.set(self.pen_width)
        self.slider.grid(row=0, column=1)
        self.controls.pack(side="left")
        self.c = Canvas(self.master, width=500, height=400, bg=self.color_bg)
        self.c.pack(fill=BOTH, expand=True)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Menu', menu=optionmenu)
        optionmenu.add_command(label='Brush Color', command=self.change_fg)
        optionmenu.add_command(label='Background Color', command=self.change_bg)
        optionmenu.add_command(label='Clear Canvas', command=self.clearcanvas)
        optionmenu.add_command(label='Exit', command=self.master.destroy)    






if __name__ == "__main__": 
    # output paint application
    win = Tk()
    win.title("Paint App")
    paint(win)
    win.mainloop()

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
    face_outline_points = [30,31,33,35]
    # print(face_outline_points)
    prev_face = None
    z = 0
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
            # rotZ = math.atan(landmarks.part(31).y/landmarks.part(31).x)
            # print("rotZ", rotZ)

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

            # **************************************************************************************************************
            # GET ROTATION OF FACE
            # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
            size = frame.shape
 
            #2D image points. If you change the image, you need to change vector
            image_points = np.array([
                            (landmarks.part(33).x, landmarks.part(33).y),     # Nose tip
                            (landmarks.part(8).x, landmarks.part(8).y),     # Chin
                            (landmarks.part(45).x, landmarks.part(45).y),     # Left eye left corner
                            (landmarks.part(36).x, landmarks.part(36).y),     # Right eye right corne
                            (landmarks.part(54).x, landmarks.part(54).y),     # Left Mouth corner
                            (landmarks.part(48).x, landmarks.part(48).y)      # Right mouth corner
                        ], dtype="double")
 
            # 3D model points.
            model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
 
                        ])
            # Camera internals
 
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                                    [[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype = "double"
                                    )

            
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs) #, flags=cv2.SOLVEPNP_ITERATIVE)
            print("rotation_vector", rotation_vector) # euler vector

            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            
            cv2.line(frame, p1, p2, (255,0,0), 2)
            # ***************************************************************************************************************

            movementV2(middle, avg_point, dt)  # movementV2(middle, avg_point, dt) # center_top
            # movementV2(middle, avg_point, dt)  # center_top
        
        dt = time.time()-prev
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27: # if the esc key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
