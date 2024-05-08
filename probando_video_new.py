import cv2
import torch

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True) 
model.conf = 0.25  # Umbral de confianza para las detecciones

# Abrir el vídeo de entrada
video_path = 'video_test/count_person.mp4'
video_name = 'count_person1'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Configurar el VideoWriter para guardar el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result_video_test/' + video_name + '_test.mp4', fourcc, fps, (frame_width, frame_height))

# Posición de la línea central
line_position = frame_height // 2
entering_count = 0
exiting_count = 0

# Procesar el video frame por frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección con YOLOv5
    results = model(frame)

    # Dibujar la línea central
    cv2.line(frame, (0, line_position), (frame_width, line_position), (255, 0, 0), 2)

    # Rastrear personas y contar
    for det in results.xyxy[0]:
        if results.names[int(det[5])] == 'person' and det[4] >= model.conf:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            center_y = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if center_y < line_position:  # Persona cruzando hacia arriba (entrando)
                entering_count += 1
            else:  # Persona cruzando hacia abajo (saliendo)
                exiting_count += 1

    # Mostrar contadores
    cv2.putText(frame, f'Entering: {entering_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Exiting: {exiting_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Escribir el frame en el archivo de salida
    out.write(frame)

    # Mostrar el frame en tiempo real
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video procesado guardado como '{output_path}'")
print(f"Total personas entrando: {entering_count}")
print(f"Total personas saliendo: {exiting_count}")
