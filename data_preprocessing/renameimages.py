import os

inputdirpath = ""
path = os.chdir(inputdirpath)
class1 = "miss"
class2 = "make"

i = 0
for file in os.listdir(path):
    new_file_name = class1 + "{}.jpg".format(i)
    os.rename(file, new_file_name)

    i += 1