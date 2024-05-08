import cv2
import torch

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True) 
model.conf = 0.25  # Umbral de confianza para las detecciones

# Abrir el vídeo de entrada
video_path = 'video_test/store.mp4'
video_name = 'store2'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Configurar el VideoWriter para guardar el video
output_path = 'result_video_test/' + video_name + '_test.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Definir los puntos de la línea de entrada y salida
entry_line_y = int(frame_height * 0.3)  # Ajustar según necesidad
exit_line_y = int(frame_height * 0.7)   # Ajustar según necesidad

entering_count = 0
exiting_count = 0

# Procesar el video frame por frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección con YOLOv5 cada 10 frames
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
        results = model(frame)
        for det in results.xyxy[0]:
            if results.names[int(det[5])] == 'person' and det[4] >= model.conf:
                x, y, w, h = int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])
                center_x = x + w // 2
                center_y = y + h // 2

                # Comprobar si cruza la línea de entrada
                if center_y <= entry_line_y:
                    entering_count += 1
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Punto rojo

                # Comprobar si cruza la línea de salida
                if center_y >= exit_line_y:
                    exiting_count += 1
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Punto verde

    # Dibujar las líneas de entrada y salida
    cv2.line(frame, (0, entry_line_y), (frame_width, entry_line_y), (0, 0, 255), 2)  # Línea roja
    cv2.line(frame, (0, exit_line_y), (frame_width, exit_line_y), (0, 255, 0), 2)  # Línea verde

    # Mostrar contadores
    cv2.putText(frame, f'Entrada: {entering_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Salida: {exiting_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Escribir el frame en el archivo de salida y mostrar el frame en tiempo real
    out.write(frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video procesado guardado como '{output_path}'")
print(f"Total personas entrando: {entering_count}")
print(f"Total personas saliendo: {exiting_count}")
