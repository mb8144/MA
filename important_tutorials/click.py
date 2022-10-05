"""Dieses Skript wird benötigt, damit Colab während dem Trainieren des
Models nicht disconnected. Es führt einen simplen Linksclick alle 3min durch.
Der Crusor wird eine Zeile unterhalb platziert."""

from pynput.mouse import Controller, Button
import time
mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    print("clicked")

    time.sleep(180)
