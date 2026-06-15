import cv2
import os
import numpy as np
import time
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "facerecognition/doorlock/open"

DATA_DIR = "data/known_faces_opencv"  # Struktur: data/known_faces_opencv/<nama>/*.jpg

def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists(data_folder_path):
        print(f"Folder data '{data_folder_path}' empty. Please create one and add a face image.")
        return faces, labels, label_map

    for name in os.listdir(data_folder_path):
        person_dir = os.path.join(data_folder_path, name)
        if not os.path.isdir(person_dir):
            continue
        label_map[current_label] = name
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        current_label += 1
    return faces, labels, label_map

def print_instructions():
    """Instructions"""
    print("\n=== Program Face Recognition using OpenCV ===")
    print("How to use:")
    print("1.  Add your face to folder data/known_faces_opencv/<nama>/")
    print("2. Program recognize face that alredy registed")
    print("3. Press 'q' to quit program")
    print("================================================\n")

def main():
    print_instructions()
    # Pastikan folder data ada
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Folder {DATA_DIR} telah dibuat.")
    
    # Load training data
    faces, labels, label_map = prepare_training_data(DATA_DIR)
    if len(faces) == 0:
        print("No data please add to data/known_faces_opencv/<nama>/*.jpg")
        print("Use script 'add_face_opencv.py' to add data from webcam.")
        return

    # Train LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Setup MQTT client
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"Terhubung ke MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"Error connect MQTT broker: {e}")
        return

    # Coba buka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera. Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
        return
        
    print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")
    
    prev_time = time.time()
    fps = 0
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_rects:
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (faces[0].shape[1], faces[0].shape[0]))  # Resize ke ukuran training
            label, confidence = recognizer.predict(face_roi_resized)
            # Jika confidence < 50, tampilkan "Tidak dikenali" (merah), jika tidak, tampilkan nama (hijau)
            if confidence < 50:
                name = "Unknown"
                color = (0, 0, 255)  # Merah (BGR)
            else:
                name = label_map.get(label, "Unknown")
                color = (0, 255, 0)  # Hijau (BGR)
                # Publish MQTT message when face is recognized
                mqtt_client.publish(MQTT_TOPIC, payload=name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Hitung FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow("Face Recognition (OpenCV)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

if __name__ == "__main__":
    main()
