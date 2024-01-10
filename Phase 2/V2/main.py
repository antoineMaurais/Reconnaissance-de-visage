import json
import cv2 as cv

####################################
# Fonctions
####################################
from vectoriser_oeil import vectoriser_oeil
from classifier import classifier
from pick_random_img_in_label import pick_random_img_in_label
from calculer_la_moyenne_du_vecteur_de_chaque_label import calculer_la_moyenne_du_vecteur_de_chaque_label

####################################
# Paramètres
####################################

face_cascade = cv.CascadeClassifier('../../Phase 1/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier("../../Phase 1/haarcascade_eye.xml")

# Image d'entrée
img = cv.imread('../../Image/img_label_2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Dimension de l'image capturée pour avoir des données uniforme.
desired_width = 24
desired_height = 24

scale_factor = 1.3 # Un facteur plus petit augmente la chance de détecter des visages plus petits, mais peut aussi augmenter les faux positifs.
min_neighbors = 5 # Augmenter cette valeur réduit les faux positifs. Cela définit le nombre de voisins qu'un rectangle doit avoir pour être retenu.

nb_label = 2 # Nombre de personnes accessible par le model. 1 label = Liste des photos d'une personne

####################################
# Création des données nécessaires
####################################
# (Décommenter si vous avec besoin de créer les données (cela peut prendre du temps)
# calculer_la_moyenne_du_vecteur_de_chaque_label(2)

####################################
# Prédiction de la classe de l'image
####################################


# Supposons que 'nouveau_vecteur' est le vecteur de la nouvelle image
# et que 'vecteurs_moyens' est le dictionnaire de vos vecteurs moyens

# Chemin vers votre fichier JSON
fichier_json = '../../Data/mean_vectors.json'

# Charger les vecteurs moyens à partir du fichier JSON
with open(fichier_json, 'r') as file:
    vecteurs_moyens = json.load(file)

# Vectoriser l'image d'entrée de l'utilisateur
nouveau_vecteur = vectoriser_oeil(img, show_results=True, scale_factor=1.2)

if nouveau_vecteur is None or len(nouveau_vecteur) == 0:
    print("Aucun œil détecté ou erreur dans la vectorisation de l'image. Len = ", len(nouveau_vecteur))
else:
    classe_predite = classifier(nouveau_vecteur, vecteurs_moyens)
    print(f"La classe prédite pour la nouvelle image est : {classe_predite}")
    pick_random_img_in_label(classe_predite)