import cv2
import torch
import os

# Cargar el modelo de YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./modelos/best.pt', force_reload=True) 

# Abrir el video
video_path = 'video_test/personas.mp4'
video_name = 'personas.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# Preparar el guardado del video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Definir el codec
out = cv2.VideoWriter('result_video_test/' + video_name + '_test.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el formato de imagen de BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar detección
    results = model(frame_rgb)

    # Dibujar cajas delimitadoras y etiquetas en el frame
    results.render()

    # Convertir de vuelta a BGR para mostrar y guardar con OpenCV
    frame_out = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

    # Escribir el frame en el archivo de salida
    out.write(frame_out)

    # Opcional: Mostrar el frame en una ventana
    cv2.imshow('Video', frame_out)
    if cv2.waitKey(1) == ord('q'):
        break

# Limpieza
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video procesado guardado como 'output_video.mp4'")
