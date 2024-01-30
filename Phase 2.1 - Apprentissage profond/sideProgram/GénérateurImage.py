from keras.preprocessing.image import ImageDataGenerator

# Configuration de l'augmentation des données
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalisation des valeurs de pixels
    rotation_range=40,      # Rotation aléatoire de l'image (degrés, 0-180)
    width_shift_range=0.2,  # Décalage horizontal aléatoire (fraction de la largeur totale)
    height_shift_range=0.2, # Décalage vertical aléatoire (fraction de la hauteur totale)
    shear_range=0.2,        # Cisaillement aléatoire
    zoom_range=0.2,         # Zoom aléatoire à l'intérieur des images
    horizontal_flip=True,   # Retournement aléatoire des images horizontalement
    fill_mode='nearest'     # Stratégie pour remplir les pixels nouvellement créés
)

# Chemin vers le dossier d'entraînement
train_dir = '../Image/trainSet/'  # Mettez à jour avec le chemin approprié

# Générateur de données d'entraînement
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(178, 218),  # Taille des images après redimensionnement
    batch_size=32,
    class_mode='binary'  # 'binary' pour classification binaire, 'categorical' pour multiclasse
)
