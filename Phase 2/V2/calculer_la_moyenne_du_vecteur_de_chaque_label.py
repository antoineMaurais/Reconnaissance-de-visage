import json
import cv2 as cv
import numpy as np
from vectoriser_oeil import vectoriser_oeil


# Créer le fichier "mean_vectors.json"
def calculer_la_moyenne_du_vecteur_de_chaque_label(nb_label=2):
    # Étape 1 : Lire le Fichier JSON

    # Charger le fichier JSON
    with open('../../Data/sorted_by_label.json', 'r') as file:
        data = json.load(file)

    # Utiliser seulement les 30 premiers labels
    selected_labels = list(data['data'].keys())[:nb_label]

    # Étape 2 : Calculer les Vecteurs pour Chaque Image

    vectors_by_class = {label: [] for label in selected_labels}

    for label in selected_labels:
        for image_name in data['data'][label]:
            img_path = f'../../Image/img_align_celeba/img_align_celeba/{image_name}'
            img = cv.imread(img_path)
            # Assurez-vous que l'image a été chargée correctement
            if img is not None:
                vecteurs = vectoriser_oeil(img, show_results=False)
                if vecteurs is not None:
                    vectors_by_class[label].extend(vecteurs)
                else:
                    print(f"Aucun œil détecté pour l'image {image_name}")

    # Étape 3 : Calculer le Vecteur Moyen pour Chaque Label

    mean_vectors = {}
    for label, vectors in vectors_by_class.items():
        # Vérifier que tous les vecteurs ont la même longueur
        if all(len(vec) == len(vectors[0]) for vec in vectors):
            mean_vectors[label] = np.mean(vectors, axis=0).tolist()
            print(f"Success : Les vecteurs pour le label {label} ont la même longueur.")
        else:
            print(f"Erreur : Les vecteurs pour le label {label} n'ont pas la même longueur.")

    # Étape 4 : Enregistrer en Format JSON

    with open('../../Data/mean_vectors.json', 'w') as json_file:
        json.dump(mean_vectors, json_file,
                  indent=4)  # Indent = Nombre d'espaces à chaque colonne du fichier JSON (améliore la lecture du fichier)

    # Ici, on obtient le vecteur moyen des yeux observé pour chaque label de personnes.
