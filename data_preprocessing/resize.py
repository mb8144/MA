"""
Skript wurde erstellt, um alle Bilder
des finalen Datensets(ca. 19'000) mit der Grösse (224, 224)zu
komprimierern.
"""
import cv2
import glob
import numpy as np

input_path = r'<path>/*.jpg'
out_path = '<path>/'

image_paths = list(glob.glob(input_path))

for i, img in enumerate(image_paths):
    img_size = 224
    try:
        image = cv2.imread(img)
        image = cv2.resize(image, (img_size, img_size))
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        continue
