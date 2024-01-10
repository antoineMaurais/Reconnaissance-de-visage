import numpy as np


# Étape 5 : Classification

def classifier(nouveau_vecteur, vecteurs_moyens):
    classe_plus_proche = None
    distance_minimale = float('inf')

    for label, vecteur_moyen in vecteurs_moyens.items():
        distance = np.linalg.norm(np.array(nouveau_vecteur) - np.array(vecteur_moyen))
        if distance < distance_minimale:
            distance_minimale = distance
            classe_plus_proche = label

    return classe_plus_proche
