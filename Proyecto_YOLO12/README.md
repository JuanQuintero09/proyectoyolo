# Detección de Formas Geométricas en Tiempo Real con YOLOv12

Este proyecto implementa un sistema de visión por computador basado en **YOLOv12 (versión Nano `yolo12n.pt`)** para la detección y clasificación en tiempo real de tres clases de figuras geométricas: **Círculo**, **Cuadrado** y **Triángulo**, desplegado mediante cámara web con aceleración por GPU.

---

## 💻 Requisitos y Entorno de Ejecución

* **Sistema Operativo**: Windows 10 / 11
* **Lenguaje**: Python 3.12
* **Aceleración Hardware**: NVIDIA GeForce GTX 1650 (CUDA 12.1)
* **Librerías necesarias**:
  ```bash
  pip install ultralytics opencv-python torch torchvision numpy roboflow


1. Preparacion de datos

El dataset utilizado es "Geometric Shape Recognition" de Roboflow Universe (autor: prutengiz), el cual contiene 450 imágenes etiquetadas para las clases circle, square y triangle.

from roboflow import Roboflow
rf = Roboflow(api_key="gJpbJCuwKOZNsMJHhgAX")
project = rf.workspace("prutengiz").project("geometric-shape-recognition")
dataset = project.version(1).download("yolov8", location="data")

Proyecto_YOLO12/
├── data/
│   ├── train/       # Imágenes y etiquetas de entrenamiento
│   ├── valid/       # Imágenes y etiquetas de validación
│   └── test/        # Imágenes y etiquetas de prueba
└── data.yaml        # Archivo de configuración del dataset

path: C:/Users/camil/Documents/Proyecto_YOLO12/data
train: train/images
val: valid/images
test: test/images

names:
  0: circle
  1: square
  2: triangle

2. Cómo Entrenar el Modelo

python train.py

3. Cómo Ejecutar la Demo en Tiempo Real

python detect.py

4. Cómo Reproducir las Métricas de Validación

python -c "import glob; from ultralytics import YOLO; pesos = max(glob.glob('**/best.pt', recursive=True)); model = YOLO(pesos); metrics = model.val(data='data.yaml'); print('\n--- RESULTADOS REPRODUCIDOS ---'); print(f'mAP@0.5 (Accuracy_val): {metrics.box.map50 * 100:.2f}%'); print(f'Precisión (Precision):   {metrics.box.mp * 100:.2f}%'); print(f'Exhaustividad (Recall): {metrics.box.mr * 100:.2f}%')"


Conclusiones
- Para evitar ruido (detecciones fantasma/de fondo) se debe hacer un entrenamiento con otro dataset de caras y fondos para que el modelo los detecte y los distinga de las figuras
- El ambiente en el que se haga la detección se debe tener una iluminación decente si no puede causar ruido o incluso fallos en la detección 
- A la hora de realizar el entrenamiento se requiere tener un early stopping para evitar el sobreentrenamiento del modelo
- A la hora de descarga de datos se debe tener en cuenta no tener archivos con nombres similares a los del dataset para evitar sobreescrituras y fallos de compilación
- El uso de YOLO12n facilita la detección en tiempo real sobre el YOLO12s debido a su menor consumo de hardware y su mayor velocidad de detección.