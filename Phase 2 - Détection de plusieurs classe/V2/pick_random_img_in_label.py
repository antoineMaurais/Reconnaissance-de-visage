# Affichier une image aléatoire du label prédit.

import matplotlib.pyplot as plt
import cv2 as cv
import random
import json

# Chemin vers votre fichier JSON contenant les images par label
fichier_images_json = '../../Data/sorted_by_label.json'

# Charger les noms des images à partir du fichier JSON
with open(fichier_images_json, 'r') as file:
    images_par_label = json.load(file)['data']


def pick_random_img_in_label(classe_predite):
    if classe_predite in images_par_label:
        # Choisit une image au hasard dans la liste pour ce label
        nom_image = random.choice(images_par_label[classe_predite])
        print(f"On charge l'image {nom_image}")

        chemin_image = f'../../Image/img_align_celeba/img_align_celeba/{nom_image}'  # Modifier selon l'organisation de vos données

        # Charger et afficher l'image
        img_label = cv.imread(chemin_image)
        if img_label is not None:
            # plt.imshow(cv.cvtColor(img_label, cv.COLOR_BGR2RGB))
            plt.imshow(img_label)
            plt.title(f"Image du label {classe_predite}")
            plt.show()

            cv.imshow('img_label', img_label)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print(f"Impossible de charger l'image {nom_image}.")
