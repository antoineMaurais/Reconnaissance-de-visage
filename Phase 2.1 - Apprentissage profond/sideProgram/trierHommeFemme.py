import json
import os
import shutil

# Chemin du fichier JSON
chemin_fichier_json = 'classification_result-trainSet.json'

# Chemins des dossiers source et cible
chemin_dossier_source = '../Image/img_align_celeba/img_align_celeba/'
chemin_dossier_hommes = '../Image/hommes/'
chemin_dossier_femmes = '../Image/femmes/'

# Créer les dossiers cibles s'ils n'existent pas
os.makedirs(chemin_dossier_hommes, exist_ok=True)
os.makedirs(chemin_dossier_femmes, exist_ok=True)

# Charger les données JSON
with open(chemin_fichier_json, 'r') as fichier:
    data = json.load(fichier)

# Déplacer les images dans les dossiers correspondants
for homme in data['men']:
    chemin_source = os.path.join(chemin_dossier_source, homme)
    chemin_destination = os.path.join(chemin_dossier_hommes, homme)
    shutil.move(chemin_source, chemin_destination)

for femme in data['women']:
    chemin_source = os.path.join(chemin_dossier_source, femme)
    chemin_destination = os.path.join(chemin_dossier_femmes, femme)
    shutil.move(chemin_source, chemin_destination)
