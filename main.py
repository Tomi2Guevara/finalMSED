import cv2
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from skimage.filters.rank import entropy
from skimage.morphology import disk
import joblib

# Configuración inicial (ajustar con tus datos)
MODEL_PATH = "snow_model.pkl"
IMAGE_DIR = r"C:\Users\tguev\Documents\Fing\Polytech\para2100"
TRAIN_MODE = True  # Cambiar a False después de entrenar


def extract_features(image):
    # Preprocesamiento
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Máscara de nieve (combinando HSV y LAB)
    white_mask_hsv = cv2.inRange(hsv, (190, 0, 30), (250, 50, 50))
    white_mask_lab = cv2.inRange(lab, (140, 120, 120), (255, 135, 135))
    combined_mask = cv2.bitwise_or(white_mask_hsv, white_mask_lab)

    # 2. Mejorar máscara con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Porcentaje inicial de píxeles blancos
    white_perc = np.mean(cleaned_mask) / 255.0

    # 4. Entropía en áreas "no nieve"
    non_snow = cv2.bitwise_not(cleaned_mask)
    texture = entropy(gray, disk(5), mask=non_snow)
    texture_mean = np.mean(texture) if np.any(non_snow) else 0

    # 5. Densidad de bordes en áreas no cubiertas
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(cv2.bitwise_and(edges, edges, mask=non_snow)) / 255.0

    # 6. Medida de contraste local
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    return [white_perc, texture_mean, edge_density, blur]


def train_model(image_dir, labels):
    # Cargar y procesar imágenes etiquetadas
    features = []
    for img_path, obstruction in labels.items():
        image = cv2.imread(os.path.join(image_dir, img_path))
        if image is not None:
            features.append(extract_features(image))

    # Entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(features, labels.values(), test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Guardar modelo
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo entrenado. Precisión: {model.score(X_test, y_test):.2%}")


def predict_obstruction(image_dir):
    model = joblib.load(MODEL_PATH)
    files = sorted(os.listdir(image_dir), key=lambda x: datetime.strptime(x.split('.')[0], "%Y-%m-%d"))

    results = []
    for file in files:
        image = cv2.imread(os.path.join(image_dir, file))
        if image is None:
            continue

        # Redimensionar si es necesario
        if image.shape[:2] != (1512, 2688):
            image = cv2.resize(image, (2688, 1512))

        features = extract_features(image)
        obstruction = model.predict([features])[0]
        obstruction = max(0, min(100, obstruction))  # Asegurar rango 0-100%

        results.append({
            'date': file,
            'obstruction_percent': round(obstruction, 1),
            'features': features
        })

    return results


# Ejecución (usar tus datos reales)
if TRAIN_MODE:
    # Formato: {'imagen1.jpg': 30.5, 'imagen2.jpg': 0.0, ...}
    labeled_data = {...}  # Cargar tus etiquetas aquí
    train_model(IMAGE_DIR, labeled_data)
else:
    results = predict_obstruction(IMAGE_DIR)
    for res in results:
        print(f"{res['date']}: {res['obstruction_percent']}% obstrucción")