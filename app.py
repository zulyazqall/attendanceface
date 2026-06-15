import os
import cv2
import numpy as np
import time
import json
from datetime import datetime, timedelta
from io import BytesIO
import paho.mqtt.client as mqtt
from flask import Flask, render_template, request, redirect, url_for, Response, flash, session, send_file

# --- Konfigurasi Aplikasi ---
DATA_DIR = "data/known_faces_opencv"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = "logdata.json"  # File untuk log absensi
CONFIG_FILE = "config.json"  # File untuk konfigurasi aplikasi

app = Flask(__name__)
app.secret_key = "supersecretkey_for_face_recognition_app"

# Konfigurasi MQTT
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "" #"facerecognition/doorlock/open"

REGISTER_STATE = {}

last_detection_status = {
    "recognized": False,
    "name": "Unknown",
    "accuracy": 0.0,
    "clocked_in_today": False,
    "clocked_out_today": False,
    "pending_clock_out_message": ""
}

last_automatic_attendance_time = {}
ATTENDANCE_COOLDOWN_SECONDS = 10

# --- Variabel baru untuk pesan error stream ---
video_stream_error_message = "" # Variabel global untuk menyimpan pesan error stream

# --- Fungsi Callback MQTT (untuk CallbackAPIVersion.VERSION2) ---
def on_mqtt_connect(client, userdata, flags, rc, properties):
    """Callback saat berhasil terhubung ke broker MQTT."""
    print(f"Connected to MQTT Broker with result code {rc}")


def on_mqtt_disconnect(client, userdata, flags, rc, properties): # Perbaikan: Tambahkan 'flags'
    """Callback saat terputus dari broker MQTT."""
    print(f"Disconnected from MQTT Broker with result code {rc}")


# --- Fungsi Utilitas ---

def _setup_mqtt_client():
    """Menyiapkan koneksi ke broker MQTT."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_mqtt_connect
    client.on_disconnect = on_mqtt_disconnect
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()  # Jalankan loop di background
        return client
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        return None


def load_config():
    """Memuat konfigurasi dari file JSON."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {CONFIG_FILE}, creating default config.")
                config = {}
    else:
        config = {}
    # Menggunakan 'min_hours_between_clock_in_out' sebagai kunci utama
    config.setdefault('min_hours_between_clock_in_out', 8.0) # Default 8 jam
    return config


def save_config(config):
    """Menyimpan konfigurasi ke file JSON."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_today_attendance_status(name: str):
    """
    Membaca logdata.json dan mengembalikan status absensi untuk nama tertentu
    pada hari ini.
    Mengembalikan dictionary dengan 'clocked_in_time' (datetime obj or None)
    dan 'clocked_out' (bool).
    """
    clocked_in_time = None
    clocked_out = False

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

        today_date_str = datetime.now().strftime("%Y-%m-%d")

        for entry in reversed(data):
            if "time" not in entry or not isinstance(entry["time"], str):
                continue

            try:
                entry_date_dt = datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S")
                entry_date_str = entry_date_dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

            if entry_date_str == today_date_str and entry["name"] == name:
                if entry.get("clock_in"):
                    clocked_in_time_str = f"{entry_date_str} {entry['clock_in']}"
                    try:
                        clocked_in_time = datetime.strptime(clocked_in_time_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            clock_in_time_only = datetime.strptime(entry['clock_in'], "%H:%M:%S").time()
                            clocked_in_time = datetime.combine(entry_date_dt.date(), clock_in_time_only)
                        except ValueError:
                            pass
                if entry.get("clock_out"):
                    clocked_out = True
                if clocked_in_time and clocked_out:
                    break
    return {"clocked_in_time": clocked_in_time, "clocked_out": clocked_out}


def log_attendance(name: str, confidence: float, accuracy: float, action: str):
    """
    Menambahkan catatan absensi (masuk/pulang) ke file JSON.
    Mengupdate entri yang sudah ada jika action adalah "clock_out" untuk hari yang sama.
    """
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_hour_minute_second = datetime.now().strftime("%H:%M:%S")
    today_date_str = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    updated = False
    if action == "clock_out":
        for i in range(len(data) - 1, -1, -1):
            log_entry = data[i]
            if "time" not in log_entry or "name" not in log_entry:
                continue

            try:
                log_date_dt = datetime.strptime(log_entry["time"], "%Y-%m-%d %H:%M:%S")
                log_date_str = log_date_dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

            if (log_date_str == today_date_str and
                    log_entry["name"] == name and
                    log_entry.get("clock_in") and
                    not log_entry.get("clock_out")):
                data[i]["clock_out"] = current_hour_minute_second
                data[i]["time"] = current_time_str
                updated = True
                print(f"Updated Clock Out for {name} at {current_time_str}")
                break

    if not updated:
        entry = {
            "time": current_time_str,
            "name": name,
            "confidence": round(confidence, 2),
            "accuracy_percent": round(accuracy, 2),
            "clock_in": "",
            "clock_out": ""
        }
        if action == "clock_in":
            entry["clock_in"] = current_hour_minute_second
            print(f"Logged new Clock In for {name} at {current_time_str}")
        elif action == "clock_out":
            entry["clock_out"] = current_hour_minute_second
            print(f"Logged new Clock Out (without prior Clock In) for {name} at {current_time_str}")
        data.append(entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def gen_register(name, num_samples=50, delay=0.7):
    """
    Menghasilkan frame video untuk proses registrasi wajah.
    Mengambil 'num_samples' gambar wajah untuk setiap nama.
    """
    save_dir = os.path.join(DATA_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream for registration.")
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, "Error: Kamera tidak dapat diakses.", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                    2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Haar cascade XML file not loaded.")
        cap.release()
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, "Error: File deteksi wajah tidak ditemukan.", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    count = 0
    last_capture = time.time()

    try:
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera during registration. Stream might have ended.")
                frame = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(frame, "no connection, camera off", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            2)
                cv2.putText(frame, "Coba refresh halaman.", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Sample {count + 1}/{num_samples}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

                if time.time() - last_capture > delay:
                    face_img = gray[y:y + h, x:x + w]
                    face_img = cv2.resize(face_img, (200, 200))

                    face_img = cv2.equalizeHist(face_img)

                    img_path = os.path.join(save_dir, f"{name}_{count + 1}.jpg")
                    cv2.imwrite(img_path, face_img)
                    count += 1
                    last_capture = time.time()
                    if count >= num_samples:
                        break

            cv2.putText(frame, f"Capturing face: {count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    finally:
        cap.release()
        REGISTER_STATE[name] = True


def prepare_training_data(data_folder_path):
    """Mempersiapkan data gambar wajah untuk pelatihan model."""
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for name in os.listdir(data_folder_path):
        person_dir = os.path.join(data_folder_path, name)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_label] = name

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            if img.size == 0:
                print(f"Warning: Empty image file {img_path}")
                continue

            faces.append(img)
            labels.append(current_label)
        current_label += 1
    return faces, labels, label_map


def gen_recognition():
    """
    Menghasilkan frame video untuk proses pengenalan wajah.
    Mengenali wajah, menampilkan nama, Jarak Wajah (Confidence),
    dan Akurasi Kemiripan (persentase).
    """
    global last_detection_status, last_automatic_attendance_time, video_stream_error_message # Tambahkan video_stream_error_message

    config = load_config()
    # Ambil nilai dalam jam dan konversi ke detik
    min_hours_between_clock_in_out = float(config.get('min_hours_between_clock_in_out', 8.0))
    min_seconds_between_clock_in_out = min_hours_between_clock_in_out * 3600 # Konversi jam ke detik

    total_registered_images = 0
    for name in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, name)
        if os.path.isdir(person_dir):
            total_registered_images += len(os.listdir(person_dir))

    MIN_SAMPLES_FOR_RECOGNITION = 50
    if total_registered_images < MIN_SAMPLES_FOR_RECOGNITION:
        # Set error message for detection status
        video_stream_error_message = f"Not enough data ({total_registered_images}/{MIN_SAMPLES_FOR_RECOGNITION} sample). please register"
        last_detection_status = {
            "recognized": False,
            "name": "N/A", # Atau bisa juga "Belum Cukup Data"
            "accuracy": 0.0,
            "clocked_in_today": False,
            "clocked_out_today": False,
            "pending_clock_out_message": "",
            "video_error": video_stream_error_message # Tambahkan ini
        }
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, video_stream_error_message, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    faces, labels, label_map = prepare_training_data(DATA_DIR)

    if len(faces) == 0:
        video_stream_error_message = "Please register, system cannot found any data"
        last_detection_status = {
            "recognized": False,
            "name": "N/A", # Atau bisa juga "Belum Ada Data"
            "accuracy": 0.0,
            "clocked_in_today": False,
            "clocked_out_today": False,
            "pending_clock_out_message": "",
            "video_error": video_stream_error_message # Tambahkan ini
        }
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, video_stream_error_message, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    mqtt_client = _setup_mqtt_client()
    if mqtt_client is None:
        print("MQTT client is not available. Skipping MQTT operations.")
        # Ini bukan error video stream, jadi tidak perlu mengatur video_stream_error_message

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.train(faces, np.array(labels))
    except cv2.error as e:
        video_stream_error_message = f"Error: Gagal melatih model pengenalan wajah. ({e})"
        last_detection_status = {
            "recognized": False,
            "name": "N/A",
            "accuracy": 0.0,
            "clocked_in_today": False,
            "clocked_out_today": False,
            "pending_clock_out_message": "",
            "video_error": video_stream_error_message # Tambahkan ini
        }
        print(video_stream_error_message)
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, video_stream_error_message, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        video_stream_error_message = "Error: File deteksi wajah tidak ditemukan."
        last_detection_status = {
            "recognized": False,
            "name": "N/A",
            "accuracy": 0.0,
            "clocked_in_today": False,
            "clocked_out_today": False,
            "pending_clock_out_message": "",
            "video_error": video_stream_error_message # Tambahkan ini
        }
        print("Error: Haar cascade XML file not loaded for recognition.")
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, video_stream_error_message, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        video_stream_error_message = "Error: Kamera tidak dapat diakses."
        last_detection_status = {
            "recognized": False,
            "name": "N/A", # Atau bisa juga "Kamera Tidak Aktif"
            "accuracy": 0.0,
            "clocked_in_today": False,
            "clocked_out_today": False,
            "pending_clock_out_message": "",
            "video_error": video_stream_error_message # Tambahkan ini
        }
        print("Error: Could not open video stream for recognition.")
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(frame, video_stream_error_message, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        return

    mqtt_sent = False
    CONFIDENCE_THRESHOLD = 50

    # Reset error message when camera starts successfully
    video_stream_error_message = "" # Penting: reset jika berhasil memulai stream

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Update status for camera disconnection during runtime
                video_stream_error_message = "camera off, try to refresh."
                last_detection_status = {
                    "recognized": False,
                    "name": "N/A",
                    "accuracy": 0.0,
                    "clocked_in_today": False,
                    "clocked_out_today": False,
                    "pending_clock_out_message": "",
                    "video_error": video_stream_error_message # Tambahkan ini
                }
                print(
                    f"Warning: Failed to read frame from camera. Stream might have ended or camera is busy. Releasing camera.")
                frame_for_display = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(frame_for_display, "not connect to camera", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                cv2.putText(frame_for_display, "try to refresh.", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                _, jpeg = cv2.imencode('.jpg', frame_for_display)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                break # Keluar dari loop

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

            recognized_this_frame = False
            recognized_name = "Unknown"
            current_confidence = 0.0
            current_accuracy = 0.0
            attendance_status = {"clocked_in_time": None, "clocked_out": False}
            current_pending_message = ""

            if len(faces_rects) > 0:
                (x, y, w, h) = faces_rects[0]

                face_roi = gray[y:y + h, x:x + w]
                face_roi_equalized = cv2.equalizeHist(face_roi)

                if faces:
                    face_roi_resized = cv2.resize(face_roi_equalized, (faces[0].shape[1], faces[0].shape[0]))
                else:
                    face_roi_resized = cv2.resize(face_roi_equalized, (200, 200))

                label, confidence = recognizer.predict(face_roi_resized)

                accuracy_percent = max(0, 100 - confidence)

                if confidence < CONFIDENCE_THRESHOLD:
                    name_display = label_map.get(label, "Unknown")
                    status_text = f"Name: {name_display}"
                    confidence_text = f"Face Distance (Confidence): {confidence:.0f} (10-50)"
                    accuracy_text = f"Accuracy: {accuracy_percent:.0f}% (di atas 50%)"
                    color = (0, 255, 0)

                    recognized_this_frame = True
                    recognized_name = name_display
                    current_confidence = confidence
                    current_accuracy = accuracy_percent

                    attendance_status = get_today_attendance_status(recognized_name)

                    if current_accuracy >= 50:
                        last_action_timestamp = last_automatic_attendance_time.get(recognized_name, 0)
                        if (time.time() - last_action_timestamp) > ATTENDANCE_COOLDOWN_SECONDS:
                            if not attendance_status["clocked_in_time"]:
                                log_attendance(recognized_name, current_confidence, current_accuracy, "clock_in")
                                last_automatic_attendance_time[recognized_name] = time.time()
                                print(f"[ABSENSI] Clock In otomatis untuk {recognized_name} berhasil.")
                                current_pending_message = ""
                            elif attendance_status["clocked_in_time"] and not attendance_status["clocked_out"]:
                                time_since_clock_in = datetime.now() - attendance_status["clocked_in_time"]
                                remaining_seconds = min_seconds_between_clock_in_out - time_since_clock_in.total_seconds()

                                if remaining_seconds <= 0:
                                    log_attendance(recognized_name, current_confidence, current_accuracy, "clock_out")
                                    last_automatic_attendance_time[recognized_name] = time.time()
                                    print(f"[Attendance] Clock Out Otomatic for {recognized_name} Successfully.")
                                    current_pending_message = ""
                                else:
                                    remaining_minutes = remaining_seconds / 60
                                    current_pending_message = (
                                        f"Waiting {min_hours_between_clock_in_out:.1f} hours to Clock Out. " # Sesuaikan pesan
                                        f"You Have {remaining_minutes:.1f} minutes."
                                    )
                        else:
                            if attendance_status["clocked_in_time"] and not attendance_status["clocked_out"]:
                                time_since_clock_in = datetime.now() - attendance_status["clocked_in_time"]
                                remaining_seconds = min_seconds_between_clock_in_out - time_since_clock_in.total_seconds()
                                if remaining_seconds > 0:
                                    remaining_minutes = remaining_seconds / 60
                                    current_pending_message = (
                                        f"Wait {min_hours_between_clock_in_out:.1f} hours to Clock Out. " # Sesuaikan pesan
                                        f"you have {remaining_minutes:.1f} minutes."
                                    )
                                else:
                                    current_pending_message = f"Clock Out ready for {recognized_name}."
                            elif attendance_status["clocked_in_time"] and attendance_status["clocked_out"]:
                                current_pending_message = f"You alredy Clock In and Clock Out today."
                            else:
                                current_pending_message = f"Clock In otomatic for {recognized_name} successful. Cooldown..."
                    else:
                        current_pending_message = ""
                        status_text = f"Name: {name_display} (Low Accuracy)"

                else:
                    name_display = "Unknown"
                    status_text = f"Name: {name_display}"
                    confidence_text = f"Confidence: {confidence:.0f} (50-100+)"
                    accuracy_text = f"Accuracy: {accuracy_percent:.0f}% (under 50%)"
                    color = (0, 0, 255)
                    current_pending_message = ""

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, status_text, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, confidence_text, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, accuracy_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                last_detection_status["recognized"] = recognized_this_frame
                last_detection_status["name"] = recognized_name
                last_detection_status["accuracy"] = current_accuracy
                last_detection_status["clocked_in_today"] = attendance_status["clocked_in_time"] is not None
                last_detection_status["clocked_out_today"] = attendance_status["clocked_out"]
                last_detection_status["pending_clock_out_message"] = current_pending_message

                if recognized_this_frame and mqtt_client:
                    mqtt_client.publish(MQTT_TOPIC, payload=recognized_name)
                    mqtt_sent = True
                elif not recognized_this_frame:
                    mqtt_sent = False
            else:
                last_detection_status["recognized"] = False
                last_detection_status["name"] = "No Face Detection"
                last_detection_status["accuracy"] = 0.0
                last_detection_status["clocked_in_today"] = False
                last_detection_status["clocked_out_today"] = False
                last_detection_status["pending_clock_out_message"] = ""
                mqtt_sent = False

                cv2.putText(frame, "Cannot Recognize Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            2)

            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    finally:
        cap.release()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


# --- Routing Flask ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Nama tidak boleh kosong.", "danger")
        return redirect(url_for('index'))
    REGISTER_STATE[name] = False
    session['register_name'] = name
    return redirect(url_for('register_live'))


@app.route("/register_live")
def register_live():
    name = session.get('register_name')
    if not name:
        flash("Mohon masukkan nama terlebih dahulu.", "warning")
        return redirect(url_for('index'))
    return render_template("register_live.html", name=name)


@app.route("/register_feed")
def register_feed():
    name = session.get('register_name')
    if not name:
        return "Name not registed", 400
    if REGISTER_STATE.get(name):
        return Response(status=204)
    return Response(gen_register(name), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/register_done")
def register_done():
    name = session.get('register_name')
    if name:
        flash(f"Registrasi untuk {name} selesai dan berhasil disimpan.", "success")
        session.pop('register_name', None)
    return redirect(url_for('index'))


@app.route("/recognize")
def recognize():
    return render_template("recognize.html")



@app.route("/detection_status")
def detection_status():
    global last_detection_status, video_stream_error_message # Tambahkan video_stream_error_message
    # Buat salinan untuk ditambahkan/dimodifikasi tanpa mengubah objek global secara langsung
    status_to_send = last_detection_status.copy()
    status_to_send["video_error"] = video_stream_error_message # Sertakan pesan error video
    return status_to_send


@app.route("/video_feed")
def video_feed():
    return Response(gen_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/clock_in", methods=["POST"])
def clock_in_manual():
    flash("Absensi masuk sekarang otomatis.", "info")
    return redirect(url_for('recognize'))


@app.route("/clock_out", methods=["POST"])
def clock_out_manual():
    flash("Absensi pulang sekarang otomatis.", "info")
    return redirect(url_for('recognize'))


@app.route('/records')
def records():
    """Menampilkan catatan absensi."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    attendance_records = [
        entry for entry in data
        if entry.get("clock_in") or entry.get("clock_out")
    ]

    attendance_records.sort(key=lambda x: datetime.strptime(x["time"], "%Y-%m-%d %H:%M:%S"), reverse=True)

    return render_template('records.html', records=attendance_records)


@app.route('/clear_records')
def clear_records():
    """Menghapus seluruh file log deteksi wajah."""
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            flash("All attendance history already deleted.", "success")
        except Exception as e:
            flash(f"Fail to delete: {e}", "danger")
    else:
        flash("Empety Data History.", "info")
    return redirect(url_for('records'))


@app.route('/export')
def export():
    """Mengizinkan ekspor data deteksi wajah ke file Excel."""
    try:
        import pandas as pd
    except ImportError:
        flash(
            "Modul 'pandas' atau 'openpyxl' tidak ditemukan. Mohon instal terlebih dahulu: pip install pandas openpyxl",
            "danger")
        return redirect(url_for('index'))

    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_json(LOG_FILE)
        except Exception as e:
            print(f"Error reading JSON for export: {e}")
            df = pd.DataFrame(columns=["time", "name", "confidence", "accuracy_percent", "clock_in", "clock_out"])
    else:
        df = pd.DataFrame(columns=["time", "name", "confidence", "accuracy_percent", "clock_in", "clock_out"])

    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
        output.seek(0)
    except Exception as e:
        flash(f"Gagal membuat file Excel: {e}", "danger")
        return redirect(url_for('records'))

    return send_file(
        output,
        as_attachment=True,
        download_name="log_absensi.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    config = load_config()
    if request.method == 'POST':
        try:
            # Mengambil nilai dalam jam dari form
            min_hours = float(request.form.get('min_hours_between_clock_in_out', ''))
            if min_hours <= 0: # Pastikan lebih besar dari 0
                flash("Jumlah jam minimal harus lebih besar dari 0.", "danger")
            else:
                # Simpan nilai jam langsung ke konfigurasi
                config['min_hours_between_clock_in_out'] = min_hours
                save_config(config)
                flash(
                    f"Settings saved successfully: Minimum hours between Clock In and Clock Out is {min_hours:.1f} hour.",
                    "success")
        except ValueError:
            flash("Invalid input. Enter a number for the minimum hours..", "danger")
        return redirect(url_for('settings'))
    else:
        return render_template('settings.html', config=config)


if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        # Inisialisasi default dalam jam
        initial_config = {'min_hours_between_clock_in_out': 8.0}
        save_config(initial_config)

    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)