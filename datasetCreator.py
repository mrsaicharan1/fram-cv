import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('face-rec/data/haarcascades/haarcascade_frontalface_default.xml')

name=input("Enter Name")
number = 0
os.mkdir("images"+"/"+name)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for (x, y, w, h) in faces:
        number+=1
        cv2.imwrite("images/"+name+"/"+str(number)+".jpg",gray[y:y+h,x:x+h])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow("Face",frame)
    cv2.waitKey(1)
    if (number>50):
        break

cap.release()
cap.destroyAllWindows()
