import datetime
import sqlite3
import cv2
import dlib
import numpy as np
import pickle
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import os

# Initialize the dlib face detector and face recognition models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Replace with your path
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Replace with your path

def log_attendance(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp))
    # conn.commit()
    # conn.close()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("SELECT * FROM attendance WHERE name=? AND DATE(timestamp)=?", (name, current_date))
    if c.fetchone() is None:
        c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp))
        conn.commit()
        messagebox.showinfo("Success", f"Attendance logged for {name}")

    conn.close()
    return True

def recognize_faces_ui(label):
    #attendance_status = False
    cap = cv2.VideoCapture(0)
    with open('face_encodings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = shape_predictor(gray, face)
            face_descriptor = face_recognizer.compute_face_descriptor(rgb_frame, landmarks)

            matches = []
            for encoding in known_face_encodings:
                distance = np.linalg.norm(np.array(encoding) - np.array(face_descriptor))
                if distance < 0.6:  # Adjust threshold as needed
                    matches.append(True)
                else:
                    matches.append(False)

            if any(matches):
                attendance_status = False
                name = known_face_names[matches.index(True)]
                if attendance_status == False:
                    attendance_status=log_attendance(name)
                    label.config(text=f"Welcome, {name}! to the Travelers bay!")
                    

            (x, y, w, h) = (face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Recognize Faces - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def capture_images(name):
    cap = cv2.VideoCapture(0)
    count = 0
    current_dir = os.getcwd()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Images - Press 'c' to capture, 'q' to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            count += 1
            img_name = f"{current_dir}/{name}_{count}.jpg"  # Save image in current directory
            cv2.imwrite(img_name, frame)
            print(f"Image {img_name} saved!")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_model(images_path=None):
    if images_path is None:
        images_path = os.getcwd()  # Use current directory as default images path

    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(images_path):
        if image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png'):
            image_path = os.path.join(images_path, image_name)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for dlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) != 1:
                print(f"Image {image_name} does not contain exactly one face. Skipping.")
                continue

            face = faces[0]
            landmarks = shape_predictor(gray, face)
            face_descriptor = face_recognizer.compute_face_descriptor(rgb_image, landmarks)

            name = image_name.split('_')[0]
            known_face_encodings.append(face_descriptor)
            known_face_names.append(name)

    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

def generate_report():
    conn = sqlite3.connect('attendance.db')
    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    conn.close()
    
    current_dir = os.getcwd()
    report_path = os.path.join(current_dir, 'attendance_report.xlsx')
    df.to_excel(report_path, index=False)
    messagebox.showinfo("Success", f"Attendance report generated and saved at {report_path}")

# Create the Tkinter UI
root = tk.Tk()
root.title("Face Recognition Attendance System")

tk.Label(root, text="Name:").pack()
name_entry = tk.Entry(root)
name_entry.pack()

def capture_images_ui():
    name = name_entry.get()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return
    capture_images(name)
    messagebox.showinfo("Success", f"Images captured for {name}.")

def train_model_ui():
    train_model()
    messagebox.showinfo("Success", "Model trained successfully.")

welcome_label = tk.Label(root, text="")
welcome_label.pack()

def recognize_faces_button():
    recognize_faces_ui(welcome_label)
    messagebox.showinfo("Success", "Attendance recorded successfully.")

tk.Button(root, text="Capture Images", command=capture_images_ui).pack()
tk.Button(root, text="Train Model", command=train_model_ui).pack()
tk.Button(root, text="Recognize Faces", command=recognize_faces_button).pack()
tk.Button(root, text="Generate Report", command=generate_report).pack()

root.mainloop()
