import json
import os
from pathlib import Path
from PIL import Image
import cv2 as cv


# Chemin vers votre fichier JSON
fichier_json = '../Data/sorted_by_label.json'

# Charger les vecteurs moyens à partir du fichier JSON
with open(fichier_json, 'r') as file:
    json_content = json.load(file)

# Path to the directory where images are stored
image_directory_path = Path('../Image/img_align_celeba/img_align_celeba')

# Function to display an image by filename
def display_image(filename):
    try:
        image_path = image_directory_path / filename
        if not image_path.exists():
            print(f"Image file does not exist: {image_path}")
            return
        image = Image.open(image_path)
        image.show()
    except IOError as e:
        print(f"Error opening image {filename}: {e}")


# Function to classify images based on user input and create a JSON structure
def classify_images(json_data, start_label_index=0, end_label_index=100):
    classification_result = {'men': [], 'women': []}
    label_count = 0

    for label, images in json_data['data'].items():
        if label_count < start_label_index:
            label_count += 1
            continue  # Ignorer les labels avant l'indice de début
        if label_count >= end_label_index:
            break  # Arrêter le traitement après l'indice de fin

        if images:  # If there are images for the label
            # Display the first image of the label
            print(images[0])
            display_image(images[0])
            # Ask the user to classify the label
            classification = input("Classify the person as a man (m) or a woman (w): ").strip().lower()
            if classification == 'm':
                classification_result['men'].extend(images)
            elif classification == 'w':
                classification_result['women'].extend(images)
            else:
                print("Invalid input, skipping this label.")

        label_count += 1

    # Write the classification result to a JSON file
    with open('classification_result.json', 'w') as outfile:
        json.dump(classification_result, outfile, indent=4)

    print("Classification complete! Results saved to classification_result.json.")


# Exemple d'utilisation : traiter seulement les labels de 0 à 100
classify_images(json_content, start_label_index=200, end_label_index=300)

