"""
Input Bilder prüfen und schauen ob ResNet-50
Input-Matrix funktioniert.
"""
import cv2
import os

# Ordner wird erstellt, falls er noch nicht
# existiert
try:
    os.mkdir('resized')
except:
    folder = 'resized'
    
# Beispielbild wird gelesen
image = cv2.imread("make8.jpg")

# Testgrössen werden festgelegt
img_small = 224
img_test = 400
smaller = cv2.resize(image, (img_small,img_small))
test = cv2.resize(image, (img_test,img_test))

# Zeigen der unterschiedlichen Grössen
cv2.imshow('original_image', image)
cv2.imshow('smaller', smaller)
cv2.imshow('test', test)

# Speichern eines Bildes
cv2.imwrite('resized/smaller.jpg', smaller)
cv2.waitKey(0)
