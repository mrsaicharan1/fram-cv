import numpy as np
import cv2
import pickle

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('face-rec/data/haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+h]
        roi_color = frame[y:y+h,x:x+h]

        id_,conf = recognizer.predict(roi_gray)
        if conf>=95:
            print(id_)
            print(labels[id_])
        img_item = "my-image.png"
        cv2.imwrite(img_item,roi_gray)

        color = (0,255,0)
        stroke = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        stroke = 2
        name = labels[id_]
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)
        cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)


    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
