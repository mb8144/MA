'''
Beschreibung: basketball classification (MA)
Author:	Maurus Brunnschweiler
'''

# matplotlib importieren für backend Aktionen
import matplotlib
print(matplotlib.__version__)
matplotlib.use("Agg")

# Alle nötigen Libraries werden importiert
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

print("[INFO] Test: The program is running...")

# Argument-Parser wird initialisiert
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading images...")

# alle Dateipfade werden in einer Liste gespeichert
image_paths = list(paths.list_images(args["dataset"]))
# Test ob beide Klassen in Liste sind
randommiss = image_paths[-100]
print(randommiss)

data = []
labels = []

for image_path in image_paths:
  print(image_path)

# Label wird aus Dateiname entnommen
  label = image_path.split(os.path.sep)[-2]

# Bild wird gelesen und in (224, 224) umgeformt
# auf aspect ratio wird vorest verzichtet
  try:
   img_size = 224
   image = cv2.imread(image_path)
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image = cv2.resize(image, (img_size, img_size))

# Hinzufügen des Bildes in entsprechende Liste
   data.append(image)
   labels.append(label)
    
  except:
    print(f"{image_path} is not working.")
                

# Konvertieren der Listen in NumPy Arrays
data = np.array(data)
labels = np.array(labels)

# one-hot encoding wird durchgeführt
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Aufteilung in Trainingsdatenstz(75%) und Testdatensatz(25%)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

# Initialisierung des 'ImageDataGenerator'
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Inititalisierung des 'testing data augmentation' Objekts
# zudem 'mean-substraction'
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the
# mean subtraction value for each of the data augmentation
# objects

# Festlegung der 'mean subtraction' von ImageNet für data augmentation
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# ResNet-50 network wird geladen(Transfer Learning)
# FC Layer wird bewusst weggelassen
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(img_size, img_size, 3)))

# Festlegung des 'headModel', welches auf 'baseModel'(ResNet-50) gesetzt wird
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

# FC Modell wird auf 'baseModel' gesetzt
# finales Modell entsteht
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# !not! be updated during the training process (no backpropagation)
for layer in baseModel.layers:
	layer.trainable = False

# compile model(erstellen des Modells)
# Da nur 2 Klassen, wird binary_crossentropy verwendet

print("[INFO] compiling model...")
opt = SGD(learning_rate=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Trainieren des FC Modells für einige Epochen, alle anderen Layers sind
# "eingefroren", dies ermöglicht dem FC Modell, mit richtigen 
# "gelernten" Werten initialisiert zu werden im Vergleich zu random Werten
print("[INFO] training head...")
H = model.fit(
	x=trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])

# Evaluieren des Netzes
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# Darstellen des Fehlers und der Wahrscheindlichkeit
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss und Accuracy auf den Datensätzen")
plt.xlabel("Epochen #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Serialisieren des Modells auf Colab
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")

# Serialisieren des Label-Binarizers auf Colab
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
