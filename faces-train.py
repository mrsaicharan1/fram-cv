import os
import numpy as np
from PIL import Image
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('face-rec/data/haarcascades/haarcascade_frontalface_alt2.xml')

x_train = []
y_labels = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") & file.endswith("jpg"):
            path = os.path.join(root,path)
            label = os.path.basename(root).replace(" ","-").lower()
            print(label,path)

            pil_image = Image.open(path).convert("L") #convert to grayscale
            image_array = np.array(pil_image,"uint8") #convert into numpy array
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5) # detect x,y,w,h of faces in image

            for (x,y,w,h) in faces: # obtain training data (append ROIs to x_train)
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
