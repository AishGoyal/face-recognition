import face_recognition
import cv2
import numpy as np 
import csv
import os
import glob
from datetime import datetime
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("imgs/")
#Load Camera
cap = cv2.VideoCapture(0)

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f= open(current_date+'.csv','w+')
lnwriter = csv.writer(f)
students = sfr.known_face_names.copy()

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        if name in students:
            print(name)
            current_time= now.strftime("%H-%M-%S")
            lnwriter.writerow([name,current_time])
            students.remove(name)

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(5000) == ord("a"):
        break
        
cap.release()
cv2.destroyAllWindows()