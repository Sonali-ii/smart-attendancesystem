import cv2
import os
import numpy as np
import sqlite3
from datetime import datetime
import csv
from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)

ADMIN_PASSWORD = "admin123"

# ---------------- DATABASE SETUP ----------------

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS staff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    staff_id TEXT,
    department TEXT
)
""")

conn.commit()
conn.close()

# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")

# ---------------- ADD STAFF ----------------

@app.route("/add", methods=["GET", "POST"])
def add_staff():

    if request.method == "POST":

        password = request.form["password"]
        if password != ADMIN_PASSWORD:
            return "Unauthorized Access"

        name = request.form["name"]
        staff_id = request.form["staff_id"]
        department = request.form["department"]

        # Save to database
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO staff (name, staff_id, department) VALUES (?, ?, ?)",
                       (name, staff_id, department))
        conn.commit()
        conn.close()

        # Create folder for person
        person_path = os.path.join("known_faces", name)
        os.makedirs(person_path, exist_ok=True)

        # Handle uploaded images
        uploaded_files = request.files.getlist("images")

        if uploaded_files and uploaded_files[0].filename != "":
            for file in uploaded_files:
                file_path = os.path.join(person_path, file.filename)
                file.save(file_path)
        else:
            capture_face(name)

        return redirect("/")

    return render_template("add_staff.html")

# ---------------- SCAN PAGE ----------------

@app.route("/scan")
def scan():
    return render_template("scan.html")

# ---------------- VIDEO STREAM ----------------

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- FACE CAPTURE ----------------

def capture_face(name):

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    person_path = os.path.join("known_faces", name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(person_path, f"{count}.jpg"), face)
            count += 1

            if count >= 20:
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow("Capturing Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- TRAIN MODEL ----------------

def train_model():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    images = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir("known_faces"):

        person_path = os.path.join("known_faces", person_name)

        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            img = cv2.imread(image_path, 0)
            if img is None:
                continue

            images.append(img)
            labels.append(current_label)

        current_label += 1

    if len(images) == 0:
        return None, None

    recognizer.train(images, np.array(labels))

    return recognizer, label_map

# ---------------- VIDEO FRAME GENERATOR ----------------

def generate_frames():

    recognizer, label_map = train_model()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            if recognizer:
                label, confidence = recognizer.predict(face)

                if confidence < 80:
                    name = label_map[label]
                    mark_attendance(name)

                    cv2.putText(frame, name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y),
                          (x+w, y+h),
                          (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ATTENDANCE ----------------

def mark_attendance(name):

    file_name = "attendance.csv"

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            if name in line and date in line:
                return

    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time])

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)
