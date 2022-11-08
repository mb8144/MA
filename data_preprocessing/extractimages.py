"""
Skript für das Extrahieren der einzelnen Frames aus
Videos.
"""

import cv2
import os

count = 0
pathofinputdir = "<path>"
outputimages = "<path>/"

# Jedes Video im Ordner wird gelesen
for files in os.listdir(pathofinputdir):
    vidcap = cv2.VideoCapture(os.path.join(pathofinputdir, files))
    success, image = vidcap.read()

    # 'success' erkennt, ob frames erstellbar sind und image
    # ist ein einzelner Frame.
    print(success)
    while success:
        success, image = vidcap.read()

        if not success:
            break

        cv2.imwrite( outputimages + str(count) + ".jpg", image)

        if cv2.waitKey(10) == 27:
            break

        count += 1
