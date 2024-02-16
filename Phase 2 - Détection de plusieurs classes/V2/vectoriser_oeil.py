import cv2 as cv
from matplotlib import pyplot as plt

face_cascade = cv.CascadeClassifier('../Phase 1/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('../Phase 1/haarcascade_eye.xml')

# Détecte les yeux d'un visage et créer un vecteur à partir de chaque œil
def vectoriser_oeil(photo_personnalite, face_cascade=face_cascade, eye_cascade=eye_cascade, desired_width=24,
                    desired_height=24, scale_factor=1.3, min_neighbors=5, show_results=False):
    gray = cv.cvtColor(photo_personnalite, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

    vecteurs_oeils = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            # Aucun œil détecté
            return None
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey: ey + eh, ex: ex + ew]
            resized_eye = cv.resize(eye_roi, (desired_width, desired_height))
            eye_vector_normalized = resized_eye.flatten() / 255.0
            vecteurs_oeils.append(eye_vector_normalized)

            if show_results:
                plt.imshow(photo_personnalite)
                plt.imshow(cv.cvtColor(photo_personnalite, cv.COLOR_BGR2RGB))
                plt.title("Image d'entré")
                plt.show()

                plt.imshow(cv.resize(eye_roi, (desired_width, desired_height)), cmap='gray')
                plt.title("Œil Détecté")
                plt.show()

                plt.plot(eye_vector_normalized)
                plt.title("Vecteur de l'Œil Normalisé")
                plt.show()

    return vecteurs_oeils

# vectoriser_oeil(img, show_results=True, scale_factor=1.2)
# Maintenant, eye_vector_normalized peut être utilisé pour l'apprentissage
# eye_vector_normalized