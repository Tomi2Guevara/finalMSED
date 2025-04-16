import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
import re
import json

# Configuración inicial (ajustar con tus datos)

IMAGE_DIR = r"C:\Users\tguev\Documents\Fing\Polytech\para2100"
TRAIN_MODE = True  # Cambiar a False después de entrenar


def extract_features(img):
    # Preprocesamiento
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 1. Máscara de nieve (combinando HSV y LAB)
    white_mask_hsv = cv2.inRange(hsv, (100, 40, 40), (255, 255, 255))
    white_mask_lab = cv2.inRange(lab, (75, 70, 70), (255, 135, 135))
    combined_mask = cv2.bitwise_or(white_mask_hsv, white_mask_lab)

    # 2. Mejorar máscara con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Porcentaje inicial de píxeles blancos
    white_perc = np.sum(cleaned_mask == 255) / cleaned_mask.size

    # 4. Entropía en áreas "no nieve"
    # non_snow = cv2.bitwise_not(cleaned_mask)
    # texture = entropy(gray, disk(5), mask=non_snow)
    # texture_mean = np.mean(texture) if np.any(non_snow) else 0
    #
    # 5. Densidad de bordes en áreas no cubiertas
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(cv2.bitwise_and(edges, edges)) / 255.0

    return [white_perc, hsv, cleaned_mask, edge_density]

def fuzzy_classifier(mean_h, cant_snow, std_h):
    CVD = std_h / mean_h
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

    if CVD <= 0.5:
        p3 = 1
    elif 0.5 < CVD < 0.7:
        p3 = 1 - ((CVD - 0.5) / 0.2)
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

    if data[0] > 0.5:
        return True
    else:
        result = fuzzy_classifier(mean_h, data[0], std_h)
        # Clasificación
        if (result > 0.10) or ((result+data[0]) > 0.15):
            return True
        else:
            return False

def process_images(image_dir):
    results = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".JPG"):
            # Extraer la fecha del nombre del archivo
            match = re.search(r'para2100__(\d{4}-\d{2}-\d{2})__', filename)
            if match:
                date = match.group(1)
            else:
                continue

            # Leer la imagen
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error al cargar la imagen: {filename}")
                continue

            # Redimensionar si es necesario
            if image.shape[:2] != (1512, 2688):
                image = cv2.resize(image, (2688, 1512))

            # Extraer características
            features = extract_features(image)
            snow = classify_snow(features)
            cant = features[0]
            obs = detectObst(features[3])

            # Guardar los resultados en el diccionario
            results[date] = {
                'snow': snow,
                'cant': cant,
                'obs': obs
            }

    return results

def detectObst(edge_density):
    # Definir los límites para la detección de obstáculos
    if edge_density >= 0.003:
        return False
    else:
        return True


data = process_images(IMAGE_DIR)
for i, (date, result) in enumerate(data.items()):
    if result['snow']:
        # Obtener las claves ordenadas del diccionario
        dates = list(data.keys())

        # Verificar si hay un día anterior y posterior
        if i > 0 and i < len(dates) - 1:
            prev_day = data[dates[i - 1]]
            next_day = data[dates[i + 1]]

            # Evaluar si el día anterior y posterior no tienen nieve
            if not prev_day['snow'] and not next_day['snow']:
                result['obs'] = True


# Guardar el diccionario en un archivo JSON
with open('data.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Diccionario guardado en 'data.json'")

# #crear un gráfico de barras de la cantidad de nieve
# snow_counts = [result['cant'] for result in data.values()]
# dates = list(data.keys())
# plt.bar(dates, snow_counts)
# plt.xlabel('Fecha')
# plt.ylabel('Cantidad de nieve')
# plt.title('Cantidad de nieve por fecha')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



