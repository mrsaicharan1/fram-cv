import os
import numpy as np
from PIL import Image

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

            pil_image = Image.open(path).convert("L")#convert to grayscale
            image_data = np.array("",)#convert into numpy array
