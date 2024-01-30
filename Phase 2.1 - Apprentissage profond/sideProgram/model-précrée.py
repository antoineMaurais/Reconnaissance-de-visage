# model-précrée.py

# Utilisation d'un model déjà existant, on vient juste rajouter des couches

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# Charger VGG16 sans la partie supérieure (top) - sans les couches de classification
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Ajouter vos propres couches pour la classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # 1 pour classification binaire

model = Model(inputs=base_model.input, outputs=predictions)

# Compiler le modèle
for layer in base_model.layers:
    layer.trainable = False  # Geler les couches du modèle pré-entraîné

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
