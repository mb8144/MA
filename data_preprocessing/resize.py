import cv2
import glob
import numpy as np

input_path = r'<path>/*.jpg'
out_path = '<path>/'

image_paths = list(glob.glob(input_path))

for i, img in enumerate(image_paths):
    try:
        image = cv2.imread(img)
        image = cv2.resize(image, (224, 224))
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        continue
