from controller import Robot, Camera, Display, Lidar
from vehicle import Driver
import numpy as np
import tensorflow as tf
import time
import csv
import os
import cv2
import ctypes

# Definir el tamaño de las imágenes
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 128, 1

# Función para obtener la imagen de la cámara
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    return image[:, :, :3]

# Función para convertir a escala de grises
def greyscale_cv2(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Función para mostrar la imagen en el display
def display_image(display, image):
    image_flattened = image.flatten()
    image_ctypes = (ctypes.c_ubyte * len(image_flattened))(*image_flattened)
    display_image = display.imageNew(image_ctypes, Display.RGB, image.shape[1], image.shape[0])
    display.imagePaste(display_image, 0, 0, False)
    display.imageDelete(display_image)

# Función principal
def main():
    robot = Robot()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())
    
    camera_center = robot.getDevice("camera")
    camera_center.enable(timestep)
    
    lidar = robot.getDevice("lidar")
    lidar.enable(timestep)
    lidar.enablePointCloud()
    
    display = robot.getDevice("display")
    
    driver.setCruisingSpeed(30.0)  # Establecer velocidad de 30 km/h
    
    # Cargar el modelo de Keras
    final_model_path = 'final_model_savedmodel'
    model = tf.keras.models.load_model(final_model_path)

    print("Iniciando conducción automática...")
    
    image_save_interval = 1.0  # Intervalo de tiempo en segundos para guardar imágenes
    last_image_save_time = time.time()

    image_count = 0

    # Crear directorio para guardar las imágenes
    image_dir = 'saved_images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Abrir el archivo CSV para escritura
    with open('steering_angles.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'steering_angle', 'image_path'])
        
        last_obstacle_time = None
        stop_count = 0
        waiting_until = None
        angle_adjustment_until = None

        while robot.step(timestep) != -1:
            current_time = time.time()
            
            # Leer datos del Lidar
            lidar_values = lidar.getRangeImage()
            
            if angle_adjustment_until:
                if current_time >= angle_adjustment_until:
                    driver.setSteeringAngle(0.0)
                    angle_adjustment_until = None

            if waiting_until:
                if current_time >= waiting_until:
                    driver.setCruisingSpeed(30.0)
                    waiting_until = None
                else:
                    continue
            
            # Comprobar si hay un obstáculo dentro de un rango de 3 metros
            if any(distance < 5.0 for distance in lidar_values):
                print("Obstáculo detectado. Deteniendo el vehículo...")
                driver.setCruisingSpeed(0.0)
                
                if last_obstacle_time and (current_time - last_obstacle_time <= 10):
                    stop_count += 1
                    if stop_count >= 2:
                        print("Segunda detención dentro de 10 segundos. Ajustando la dirección y omitiendo detención.")
                        driver.setSteeringAngle(0.04)
                        angle_adjustment_until = current_time + 2  # Ajustar la dirección durante 2 segundos
                        driver.setCruisingSpeed(30.0)  # Reanudar la velocidad de crucero inmediatamente
                        stop_count = 0
                        last_obstacle_time = None  # Resetear el tiempo de última detección para evitar repetición de la lógica
                        continue
                else:
                    stop_count = 1
                
                last_obstacle_time = current_time
                waiting_until = current_time + 3  # Detener el vehículo durante 3 segundos si no es la segunda detención en 10 segundos
                continue
            
            # Guardar la imagen capturada si ha pasado el intervalo de tiempo
            if current_time - last_image_save_time >= image_save_interval:
                image_center = get_image(camera_center)
                
                # Convertir la imagen a escala de grises
                greyscale_image = greyscale_cv2(image_center)
                
                # Mostrar la imagen en el display
                display_image(display, greyscale_image)
                
                # Guardar la imagen capturada
                image_path = os.path.join(image_dir, f'image_{image_count:04d}.jpg')
                cv2.imwrite(image_path, greyscale_image)
                image_count += 1
                last_image_save_time = current_time
                
                # Preparar la imagen para la predicción
                img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_array = img_array / 255.0  # Normalizar los valores de los píxeles
                img_array = img_array.astype(np.float32)
                img_array = np.expand_dims(img_array, axis=-1)  # Añadir el canal de color (resultando en (64, 128, 1))
                img_array = np.expand_dims(img_array, axis=0)   # Añadir el batch dimension
                
                # Realizar predicciones
                predicted_angle = model.predict(img_array, verbose=0)[0][0]
                
                print("steering_angle:", predicted_angle)
                
                # Guardar en el archivo CSV
                writer.writerow([current_time, predicted_angle, image_path])
                
                # Configurar el ángulo de dirección del driver
                driver.setSteeringAngle(predicted_angle)
                driver.setCruisingSpeed(30.0)

if __name__ == "__main__":
    main()
