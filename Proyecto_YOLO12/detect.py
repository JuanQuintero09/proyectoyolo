import cv2
import numpy as np
from ultralytics import YOLO

def ejecucion_tiempo_real():
    ruta_pesos = r"C:\Users\camil\Documents\Proyecto_YOLO12\runs\detect\runs\detect\models\yolo12_figuras-4\weights\best.pt"
    modelo = YOLO(ruta_pesos)

    accuracy_val_global = 83.3  # mAP@0.5 real sobre validación

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara web.")
        return

    print("Detección multicolor con etiquetas dinámicas activa. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dibujo = frame.copy()
        
        resultados = modelo(
            frame, 
            stream=True, 
            conf=0.20, 
            iou=0.45, 
            imgsz=640
        )

        # Paleta de colores distintos (BGR)
        COLOR_CIRCULO = (0, 255, 255)   # Amarillo
        COLOR_CUADRADO = (0, 255, 0)    # Verde
        COLOR_TRIANGULO = (255, 255, 0) # Cian / Azul Claro
        COLOR_DEFECTO = (255, 0, 255)   # Magenta

        # Lista para almacenar los textos dibujados y evitar colisiones
        etiquetas_dibujadas = []

        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                nombre_clase = modelo.names[cls_id].lower()
                confianza = float(box.conf[0])

                ancho = x2 - x1
                alto = y2 - y1

                if ancho < 25 or alto < 25:
                    continue

                grosor = 2

                # Análisis de contornos para conteo de vértices
                roi = frame[y1:y2, x1:x2]
                num_vertices = 0
                
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c_max = max(cnts, key=cv2.contourArea)
                        epsilon = 0.04 * cv2.arcLength(c_max, True)
                        approx = cv2.approxPolyDP(c_max, epsilon, True)
                        num_vertices = len(approx)

                # Variables para la figura actual
                color_actual = COLOR_DEFECTO
                texto_etiqueta = ""
                tipo_figura = ""

                # 1. CÍRCULO
                if any(k in nombre_clase for k in ["circle", "circulo"]):
                    aspect_ratio = ancho / float(alto)
                    if confianza >= 0.48 and (0.75 <= aspect_ratio <= 1.30):
                        centro_x = (x1 + x2) // 2
                        centro_y = (y1 + y2) // 2
                        radio = (ancho + alto) // 4
                        cv2.circle(frame_dibujo, (centro_x, centro_y), radio, COLOR_CIRCULO, grosor)
                        color_actual = COLOR_CIRCULO
                        texto_etiqueta = f"circulo {confianza:.2f}"
                        tipo_figura = "circulo"
                    else:
                        continue

                # 2. CUADRADO / RECTÁNGULO
                elif any(k in nombre_clase for k in ["square", "cuadrado", "rectangle", "rectangulo", "box"]) or num_vertices == 4:
                    if confianza >= 0.20:
                        cv2.rectangle(frame_dibujo, (x1, y1), (x2, y2), COLOR_CUADRADO, grosor)
                        color_actual = COLOR_CUADRADO
                        texto_etiqueta = f"cuadrado {confianza:.2f}"
                        tipo_figura = "cuadrado"
                    else:
                        continue

                # 3. TRIÁNGULO
                elif any(k in nombre_clase for k in ["triangle", "triangulo"]):
                    if num_vertices == 4:
                        cv2.rectangle(frame_dibujo, (x1, y1), (x2, y2), COLOR_CUADRADO, grosor)
                        color_actual = COLOR_CUADRADO
                        texto_etiqueta = f"cuadrado (corr) {confianza:.2f}"
                        tipo_figura = "cuadrado"
                    elif confianza >= 0.32:
                        punto_superior = ((x1 + x2) // 2, y1)
                        punto_inf_izq = (x1, y2)
                        punto_inf_der = (x2, y2)
                        puntos = np.array([punto_superior, punto_inf_izq, punto_inf_der], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame_dibujo, [puntos], isClosed=True, color=COLOR_TRIANGULO, thickness=grosor)
                        color_actual = COLOR_TRIANGULO
                        texto_etiqueta = f"triangulo {confianza:.2f}"
                        tipo_figura = "triangulo"
                    else:
                        continue

                else:
                    if confianza >= 0.35:
                        cv2.rectangle(frame_dibujo, (x1, y1), (x2, y2), COLOR_DEFECTO, grosor)
                        color_actual = COLOR_DEFECTO
                        texto_etiqueta = f"{nombre_clase} {confianza:.2f}"
                    else:
                        continue

                # --- CONTROL DE POSICIÓN DE TEXTO PARA EVITAR TRASLAPE ---
                pos_y = max(y1 - 10, 25)
                pos_x = x1

                # Verificar si ya existe una etiqueta muy cerca verticalmente
                for prev_x, prev_y in etiquetas_dibujadas:
                    if abs(pos_x - prev_x) < 80 and abs(pos_y - prev_y) < 25:
                        pos_y = prev_y + 25  # Desplazar hacia abajo si hay colisión

                etiquetas_dibujadas.append((pos_x, pos_y))

                # Dibujar fondo negro pequeño detrás del texto para mejorar la legibilidad
                (w_txt, h_txt), _ = cv2.getTextSize(texto_etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_dibujo, (pos_x, pos_y - h_txt - 4), (pos_x + w_txt + 4, pos_y + 4), (0, 0, 0), -1)

                # Dibujar el texto con su color correspondiente
                cv2.putText(
                    frame_dibujo, 
                    texto_etiqueta, 
                    (pos_x + 2, pos_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    color_actual, 
                    2, 
                    cv2.LINE_AA
                )

        # OVERLAY FIJO CON ACCURACY DE VALIDACIÓN (83.3%)
        cv2.rectangle(frame_dibujo, (10, 10), (250, 50), (0, 0, 0), -1)
        texto_overlay = f"Acc_val: {accuracy_val_global:.1f}%"
        cv2.putText(
            frame_dibujo, 
            texto_overlay, 
            (20, 38), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )

        cv2.imshow("Detección de Formas Geométricas - YOLOv12", frame_dibujo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ejecucion_tiempo_real()