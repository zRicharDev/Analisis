import cv2
import torch

# Cargar el modelo YOLOv5 para detección inicial (asumiendo que ya está cargado y configurado)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True) 
model.conf = 0.25  # Configura un umbral de confianza

# Abrir video de entrada
video_path = 'video_test/personas.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Inicializar el objeto MultiTracker
trackers = cv2.legacy.MultiTracker_create()

# Contador de personas detectadas
person_count = 0

# Procesar video frame por frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Cada N frames realizamos una detección y reinicializamos los trackers
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:  # Ejemplo: cada 10 frames
        trackers = cv2.legacy.MultiTracker_create()  # Reiniciar trackers
        results = model(frame)  # Realizar detección con YOLOv5
        for det in results.xyxy[0]:  # Asumiendo que det es [x1, y1, x2, y2, conf, cls]
            if det[4] >= model.conf and results.names[int(det[5])] == 'person':
                x, y, w, h = int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])
                tracker = cv2.legacy.TrackerKCF_create()
                trackers.add(tracker, frame, (x, y, w, h))
                person_count += 1  # Incrementar contador de personas detectadas

    # Actualizar los trackers
    success, boxes = trackers.update(frame)
    for box in boxes:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibujar rectángulo alrededor de la persona

    # Mostrar el contador de personas en el frame
    cv2.putText(frame, f'Total Persons Detected: {person_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el resultado en una ventana
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Salir con la tecla 'q'
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
