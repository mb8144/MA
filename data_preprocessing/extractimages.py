import cv2
import os

#TODO Hochformatbilder stehen auf dem Kopf, im Moment jedoch nur Querformat
count = 0
pathofinputdir = ""
outputimages = ""

# / bei outputimages war zu wenig, deshalb hat es zuerst nicht in richtigen Ordner eingefügt

for files in os.listdir(pathofinputdir):
    vidcap = cv2.VideoCapture(os.path.join(pathofinputdir, files))
    success, image = vidcap.read()

    #  success erkennt, ob frames erstellbar sind und 2. Argument (image) extracts image
    print(success)
    while success:
        success, image = vidcap.read()

        if not success:
            break

        cv2.imwrite( outputimages + str(count) + ".jpg", image)

        if cv2.waitKey(10) == 27:
            break

        count += 1
