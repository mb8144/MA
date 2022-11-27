""" Dieses Skript wurde erstellt, um
die Funktion der Library argparse verstehen
und anwenden zu können.
""" 
import argparse

# einfaches Beispiel, wie argparse funktioniert
parser = argparse.ArgumentParser(description= "calculates number")
parser.add_argument("number", metavar = "number", type = int , help = "Enter you number" )
args = parser.parse_args()

number = args.number
number_final = number *4

print(number_final)
