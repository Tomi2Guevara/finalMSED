import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
# Configuración inicial (ajustar con tus datos)

IMAGE_DIR = r"C:\Users\tguev\Documents\Fing\Polytech\para2100"
TRAIN_MODE = True  # Cambiar a False después de entrenar


def extract_features(img):
    # Preprocesamiento
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 1. Máscara de nieve (combinando HSV y LAB)
    white_mask_hsv = cv2.inRange(hsv, (150, 0, 30), (255, 50, 50))
    white_mask_lab = cv2.inRange(lab, (100, 120, 120), (255, 135, 135))
    combined_mask = cv2.bitwise_or(white_mask_hsv, white_mask_lab)

    # 2. Mejorar máscara con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Porcentaje inicial de píxeles blancos
    white_perc = np.sum(cleaned_mask == 255) / cleaned_mask.size

    # 4. Entropía en áreas "no nieve"
    non_snow = cv2.bitwise_not(cleaned_mask)
    texture = entropy(gray, disk(5), mask=non_snow)
    texture_mean = np.mean(texture) if np.any(non_snow) else 0

    # 5. Densidad de bordes en áreas no cubiertas
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(cv2.bitwise_and(edges, edges, mask=non_snow)) / 255.0

    return [white_perc, hsv, cleaned_mask, edge_density]

def fuzzy_classifier(mean_h, cant_snow, std_h):
    str = std_h/mean_h
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

    if str <= 0.3:
        p3 = 1
    elif 0.3 < str < 0.6:
        p3 = 1-((str - 0.1) / 0.4)
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

def detectObst(edge_density):
    # Definir los límites para la detección de obstáculos
    if edge_density > 0.01:
        return True
    else:
        return False



lib = os.listdir(IMAGE_DIR)
result = []
foto = r"C:\Users\tguev\Documents\Fing\Polytech\para2100\para2100__2019-05-30__16-00-01(1).JPG"

image = cv2.imread(foto)


if image is None:
    print("Error al cargar la imagen.")
else:
    # Redimensionar si es necesario
    if image.shape[:2] != (1512, 2688):
        image = cv2.resize(image, (2688, 1512))

    features = extract_features(image)
    print(features[-1])
    if detectObst(features[-1]):
        snow = classify_snow(features)
        print("Probabilité de neige: " + str(int(snow * 100)) + "%")
        print("Cantité de neige: " + str(int(features[0] * 100)) + "%")
    else:
        print("obstrucción detectada")


# mostrar los 3 canales r g b como histograma sobre el mismo grafico
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# r, g, b = cv2.split(rgb)
# plt.hist(r.ravel(), 256, [0, 256], color='red', alpha=0.5)
# plt.hist(g.ravel(), 256, [0, 256], color='green', alpha=0.5)
# plt.hist(b.ravel(), 256, [0, 256], color='blue', alpha=0.5)
# plt.title('Histogramas RGB')
# plt.xlabel('Intensidad de píxel')
# plt.ylabel('Número de píxeles')
# plt.legend(['Rojo', 'Verde', 'Azul'])
# plt.show()

#plotear hsv
# h, s, v = cv2.split(features[1])
# plt.hist(h.ravel(), 256, [0, 256], color='red', alpha=0.5)
# plt.hist(s.ravel(), 256, [0, 256], color='green', alpha=0.5)
# plt.hist(v.ravel(), 256, [0, 256], color='blue', alpha=0.5)
# plt.title('Histogramas HSV')
# plt.xlabel('Intensidad de píxel')
# plt.ylabel('Número de píxeles')
# plt.legend(['H', 'S', 'V'])
# plt.show()


# #plotear el histograma de h
# h, s, v = cv2.split(features[1])
# plt.hist(h.ravel(), 256, [0, 256])
# plt.show()

