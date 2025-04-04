import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuración inicial (ajustar con tus datos)

IMAGE_DIR = r"C:\Users\tguev\Documents\Fing\Polytech\para2100"
TRAIN_MODE = True  # Cambiar a False después de entrenar


def extract_features(image):
    # Preprocesamiento
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


    # 1. Máscara de nieve (combinando HSV y LAB)
    white_mask_hsv = cv2.inRange(hsv, (150, 0, 30), (255, 50, 50))
    white_mask_lab = cv2.inRange(lab, (100, 120, 120), (255, 135, 135))
    combined_mask = cv2.bitwise_or(white_mask_hsv, white_mask_lab)

    # 2. Mejorar máscara con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Porcentaje inicial de píxeles blancos
    white_perc = np.sum(cleaned_mask == 255) / cleaned_mask.size

    return [white_perc, hsv, cleaned_mask]

def fuzzy_classifier(mean_h, cant_snow, std_h):
    p1 = 0.5
    p2 = 0.5
    p3 = 0.5
    if mean_h <= 50:
        p1 = 0
    elif 50 < mean_h < 100:
        p1 = (mean_h - 50) / 50
    else:
        p1 = 1

    if cant_snow <= 0.05:
        p2 = 0
    elif 0.05 < cant_snow < 0.5:
        p2 = (cant_snow - 0.05) / 0.45
    else:
        p2 = 1

    if std_h <= 0.1:
        p3 = 1
    elif 0.1 < std_h < 0.5:
        p3 = 1-((std_h - 0.1) / 0.4)
    else:
        p3 = 0
    if p3 == 0:
        return p2
    else:    # Definir la función de pertenencia difusa
        fuzzy_membership = (p1*p3 + p2) / 2
    return fuzzy_membership


def classify_snow(data):
    #claclar la media y desviación estándar de la h de la columan 1 de features
    mean_h = np.mean(data[1][:, :, 0])
    std_h = np.std(data[1][:, :, 0])
    # Aplicar el clasificador difuso

    return fuzzy_classifier(mean_h, data[0], std_h)



lib = os.listdir(IMAGE_DIR)
result = []
foto = r"C:\Users\tguev\Documents\Fing\Polytech\para2100\para2100__2019-03-10__16-00-00(1).JPG"

image = cv2.imread(foto)


if image is None:
    print("Error al cargar la imagen.")
else:
    # Redimensionar si es necesario
    if image.shape[:2] != (1512, 2688):
        image = cv2.resize(image, (2688, 1512))

    features = extract_features(image)
    snow = classify_snow(features)
    print("Probabilité de neige: " + str(int(snow*100)) + "%")
    print("Cantité de neige: " + str(int(features[0]*100)) + "%")

#plotear el histograma de h
h, s, v = cv2.split(features[1])
plt.hist(h.ravel(), 256, [0, 256])
plt.show()

