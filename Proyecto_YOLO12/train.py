import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO

def preparar_dataset():
    # 1. Si la carpeta data no existe o está vacía, descargar con Roboflow
    if not os.path.exists("data") or len(os.listdir("data")) == 0:
        print("Descargando dataset desde Roboflow...")
        rf = Roboflow(api_key="gJpbJCuwKOZNsMJHhgAX")
        project = rf.workspace("prutengiz").project("geometric-shape-recognition")
        project.version(1).download("yolov8", location="data")

    # 2. Corregir si Roboflow creó una subcarpeta anidada dentro de 'data'
    subcarpetas = [f for f in os.listdir("data") if os.path.isdir(os.path.join("data", f))]
    for sub in subcarpetas:
        if sub.lower().startswith("geometric-shape"):
            ruta_anidada = os.path.join("data", sub)
            for item in os.listdir(ruta_anidada):
                shutil.move(os.path.join(ruta_anidada, item), "data")
            os.rmdir(ruta_anidada)

    # 3. Renombrar 'val' a 'valid' si existe
    if os.path.exists("data/val") and not os.path.exists("data/valid"):
        os.rename("data/val", "data/valid")

    # 4. Verificar estructura requerida por YOLO
    for split in ["train", "valid"]:
        ruta_images = os.path.join("data", split, "images")
        if not os.path.exists(ruta_images):
            print(f"Advertencia: No se encontró la carpeta {ruta_images}")

def entrenar_modelo():
    preparar_dataset()

    # Cargar y entrenar modelo YOLOv12 Nano en GPU GTX 1650
    modelo = YOLO("yolo12n.pt")

    modelo.train(
        data="data.yaml",
        epochs=30,
        imgsz=416,
        batch=16,
        device=0,  # GPU GTX 1650
        project="runs/detect/models",
        name="yolo12_figuras",
        patience=8
    )

if __name__ == "__main__":
    entrenar_modelo()