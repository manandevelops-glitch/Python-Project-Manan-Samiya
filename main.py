import cv2 as cv
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

 
cam = cv.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_face = mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, refine_landmarks=True)

left_eye_landmark_list = [133, 468, 33]
right_eye_landmark_list = [263, 473, 362]

right_inner_x = 0
right_inner_y = 0
right_iris_x = 0
right_iris_y = 0
right_outer_x = 0
right_outer_y = 0
left_inner_x = 0
left_inner_y = 0
left_iris_x = 0
left_iris_y = 0
left_outer_x = 0
left_outer_y = 0

right_eye_ratio = 0
left_eye_ratio = 0

avg_ratio = 0

def no_of_People(frame):
    model = YOLO("YOLOv8n Face.pt") 
    results = model.predict(source=frame, conf=0.5)
    for result in results:
        face_count = len(result.boxes)
    if face_count>1:
        alert_Management(f"Number of faces detected: {face_count}")
    
def unwanted_objects(frame):
    model2 = YOLO("YOLOv26 COCO Dataset.pt") 
    results2 = model2.predict(source=frame,conf=0.5,classes=[67])
    for result in results2:
        phones = len(result.boxes)
    if phones>0:
        alert_Management("Phone detected")

try:
    os.mkdir("Proof_images")
except FileExistsError:
    pass

def alert_Management(message, code):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    '''
    0 - Looking away
    1 - person
    2 - object
    '''
    if code == 0:
        cv.putText(frame, message, (100,90), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2,cv.LINE_AA)
    elif code == 1:
        cv.putText(frame, message, (100,160), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2,cv.LINE_AA)
    elif code == 2:
        cv.putText(frame, message, (100,240), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2,cv.LINE_AA)
    
    cv.imwrite(f"Proof_images/{message} - {current_time}.png",frame)

    

while True:
    ret, frame = cam.read()
    frame = frame[: , ::-1]                     #Mirroring the image
    height, width, _ = frame.shape
    frame.flags.writeable = False

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = mp_face.process(rgb_frame)

    frame.flags.writeable = True
    frame = np.ascontiguousarray(frame)                          #Read about this!!!
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        
        for mark in enumerate(landmarks.landmark):          #landmarks start from 0 and the use of indexing
            try:
                if mark[0] in right_eye_landmark_list:                                  #--------------Right eye--------------
                    if mark[0] == right_eye_landmark_list[0]:            
                        right_inner_x = int(mark[1].x * width)                                 #right innerouter x
                        right_inner_y = int(mark[1].y * height)                                #right innerouter y
                        cv.circle(frame, (right_inner_x,right_inner_y), 5, (0, 0, 255), -1)
                        
                    if mark[0] == right_eye_landmark_list[1]:                       
                        right_iris_x = int(mark[1].x * width)                                   #right iris x
                        right_iris_y = int(mark[1].y * height)                                  #right iris y
                        cv.circle(frame, (right_iris_x,right_iris_y), 5, (0, 0, 255), -1)

                    if mark[0] == right_eye_landmark_list[2]:
                        right_outer_x = int(mark[1].x * width)                                  #right outer x
                        right_outer_y = int(mark[1].y * height)                                 #right outer y
                        cv.circle(frame, (right_outer_x,right_outer_y), 5, (0, 0, 255), -1)
                    
                    right_eye_ratio = (right_iris_x-right_inner_x)/(right_outer_x-right_inner_x)        # outer-inner
                    
        
                if mark[0] in left_eye_landmark_list:                                      #--------------left eye--------------
                    if mark[0] == left_eye_landmark_list[0]:            
                        left_inner_x = int(mark[1].x * width)                                   #left inner x
                        left_inner_y = int(mark[1].y * height)                                  #left inner y
                        cv.circle(frame, (left_inner_x,left_inner_y), 5, (0, 0, 255), -1)
                        
                    elif mark[0] == left_eye_landmark_list[1]:                       
                        left_iris_x = int(mark[1].x * width)                                    #left iris x
                        left_iris_y = int(mark[1].y * height)                                   #left iris y
                        cv.circle(frame, (left_iris_x,left_iris_y), 5, (0, 0, 255), -1)

                    elif mark[0] == left_eye_landmark_list[2]:
                        left_outer_x = int(mark[1].x * width)                                   #left outerinnerouter x
                        left_outer_y = int(mark[1].y * height)                                  #left outerinnerouter y
                        cv.circle(frame, (left_outer_x,left_outer_y), 5, (0, 0, 255), -1)

                    # inner - outer
                    left_eye_ratio = (left_inner_x-left_iris_x)/(left_inner_x-left_outer_x)
            
            except ZeroDivisionError:
                pass
            
            avg_ratio = (right_eye_ratio+left_eye_ratio)/2

    if(avg_ratio<0.4 or avg_ratio>0.6):
        alert_Management("Looking away!",0)

    no_of_People(frame,1)
    unwanted_objects(frame,2)


    cv.imshow("Webcam", frame)
    if cv.waitKey(1) == ord(" "):
        break
cam.release()
cv.destroyAllWindows()