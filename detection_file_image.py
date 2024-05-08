import cv2
import torch
import os

# Cargar el modelo entrenado
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./modelos/best.pt', force_reload=True)  # Asegúrate de que la ruta al modelo es correcta

# Cargar una imagen
image_path = 'image_test/image_test.jpg'
name_image = 'image_test.jpg'
img = cv2.imread(image_path)

# Verificar que la imagen fue cargada correctamente
if img is None:
    print(f"No se pudo cargar la imagen en {image_path}")
else:
    # Convertir el formato de imagen de BGR (OpenCV) a RGB para YOLOv5
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Realizar detección
    results = model(img_rgb)

    # Dibujar cajas delimitadoras y etiquetas en la imagen
    results.render()

    # Obtener la imagen procesada (la primera en el lote)
    img_out = results.ims[0]

    # Convertir de vuelta a BGR para mostrar con OpenCV
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    # Mostrar la imagen
    cv2.imshow('YOLOv5 Detection', img_out)
    cv2.waitKey(0)  # Espera hasta que se presione una tecla para cerrar la ventana
    cv2.destroyAllWindows()

    # Guardar la imagen
    save_path = 'result_image_test'
    filename = name_image
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Crear la carpeta si no existe
    cv2.imwrite(os.path.join(save_path, filename), img_out)
    print(f"Imagen guardada en {os.path.join(save_path, filename)}")