"""
Skript für die strukturierte Umbenennung 
aller Bilder
"""
import os

# Ordner, in welchem die Bilder liegen
# werden festgelegt
inputdirpath = ""
path = os.chdir(inputdirpath)

# Erstellen der beiden "Klassen"
class1 = "miss"
class2 = "make"

# Bilder werden je nach Klasse untetschiedlich
# umbenannt und nummeriert
i = 0
for file in os.listdir(path):
    new_file_name = class1 + "{}.jpg".format(i)
    os.rename(file, new_file_name)

    i += 1
