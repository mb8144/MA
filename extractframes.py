import cv2
import os

count = 0
for files in os.listdir("/Users/maurusbrunnschweiler/Downloads/OneDrive_1_11.9.2022"):
    vidcap = cv2.VideoCapture(os.path.join("/Users/maurusbrunnschweiler/Downloads/OneDrive_1_11.9.2022", files))
    success, image = vidcap.read()

# todo geht aus unerklärlichen Gründen nicht!
    # if not os.path.exists("/Users/maurusbrunnschweiler/Downloads/OneDrive_1_11.9.2022/frames"):
    # os.mkdir("./frames/")
    #  success erkennt, ob frames erstellbar sind und 2. Argument(image) extracts image
    print(success)
    while success:
        success, image = vidcap.read()

        if not success:
            break

        cv2.imwrite("/Users/maurusbrunnschweiler/Pictures/framesofvideo1/" + str(count) + ".jpg", image)

        if cv2.waitKey(10) == 27:
            break

        count += 1
