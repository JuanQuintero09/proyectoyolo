import os
from roboflow import Roboflow

def descargar_dataset():
    if not os.path.exists("data"):
        print("Descargando dataset desde Roboflow Universe...")
        rf = Roboflow(api_key="gJpbJCuwKOZNsMJHhgAX")
        
        # Workspace: prutengiz | Proyecto: geometric-shape-recognition
        project = rf.workspace("prutengiz").project("geometric-shape-recognition")
        dataset = project.version(1).download("yolov8", location="data")
        print("¡Dataset descargado exitosamente en la carpeta 'data'!")

if __name__ == "__main__":
    descargar_dataset()