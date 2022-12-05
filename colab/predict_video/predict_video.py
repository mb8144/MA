'''
basketball classification (MA)
pyimagesearch Tutorial
Code wurde noch nicht genauer betrachtet, da
Netz nicht genug genau ist 
'''

# Alle nötigen Libraries werden importiert
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from google.colab.patches import cv2_imshow

# Argument-Parser wird initialisiert
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())


# trainiertes Modell und Label-Binarizer werden hochgeladen
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
# 'image mean' für 'mean subtraction' wird definiert und
# Queue(deutsch: "Reihe") wird initialisiert
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# Videostream wird initalisiert und Frames gelesen
# Dimensionen werden mit 'None' initialisiert
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
# Aktionen werden für jeden Frame ausgeführt
while True:
	# nächster Frame wird abgelesen
	(grabbed, frame) = vs.read()
	# Schluss, wenn kei Frame mehr erkennt wird
	if not grabbed:
		break
	
	# wenn nötig werden Dimensionen festgelegt
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# Frame wird kopiert und bearbeitet inklusive 'mean
	# subtraction'
	img_size = 224
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (img_size, img_size)).astype("float32")
	frame -= mean

	# Vorhersage für Klassifizierung wird gemacht für den
	# einzelnen Frame und die Queue aktualisiert
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
	# 'Prediction Averaging' wird auf aktueller Queue durchgeführt
	# und Klasse zugeordnet
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	# Klasse wird graphisch auf Video ausgegeben
	text = "result: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
				1.25, (0, 255, 0), 5)
	# Prüfen, ob writer None ist
	if writer is None:
		# 'Video Writer' wird initialisiert
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
								 (W, H), True)
	# Output-Frame wird gesichert
	writer.write(output)
	# Output-Frame wird gezeigt
	cv2_imshow( output)
	key = cv2.waitKey(1) & 0xFF
	# wenn 'q' gedrückt wird: Abbruch
	if key == ord("q"):
		break

print("[INFO] cleaning up...")

writer.release()
vs.release()
print("[INFO] The program has reached its end...")
