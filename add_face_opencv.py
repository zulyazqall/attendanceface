import cv2
import os
import time

DATA_DIR = "data/known_faces_opencv"

def capture_faces(name, num_samples=20, delay=0.7):
    save_dir = os.path.join(DATA_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    print(f"Face the camera. Press 'q' to exit early.")
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            img_path = os.path.join(save_dir, f"{name}_{count+1}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample {count}/{num_samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Capture Face", frame)
            cv2.waitKey(1)
            time.sleep(delay)  # Tambahkan delay agar user bisa berpindah posisi
            if count >= num_samples:
                break
        else:
            cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data retrieval complete. {count} image saved in {save_dir}")

if __name__ == "__main__":
    name = input("Enter the name of the person you want to register: ").strip()
    if name:
        capture_faces(name)
    else:
        print("Name cannot be blank.")
