from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import os
import datetime
import csv
import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl import Workbook

app = Flask(__name__)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


reference_images = {}
reference_dir = r"C:\Users\hp\Desktop\Face\people" 

for filename in os.listdir(reference_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(reference_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            name = os.path.splitext(filename)[0]
            reference_images[name] = img

if not reference_images:
    print("Error: No reference images found. Check the directory.")
    exit()

attendance = {}
frame_count = 0
attendance_threshold = 20
matched_name = None


csv_file = "attendance.csv"
csv_columns = ['Name', 'Time']


def write_attendance_to_csv(name, time):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writerow({'Name': name, 'Time': time})


def check_face(frame):
    global matched_name, frame_count
    for name, ref_img in reference_images.items():
        try:
            result = DeepFace.verify(frame, ref_img.copy())
            if result['verified']:
                if name not in attendance:
                    attendance[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    write_attendance_to_csv(name, attendance[name])  
                    print(f"Attendance marked for {name} at {attendance[name]}")
                frame_count = 0  
                matched_name = name  
                return True
        except Exception as e:
            print(f"Error in face verification for {name}: {e}")
    return False


def gen_frames():
    global frame_count
    while True:
        success, frame = cap.read()  
        if not success:
            break
        else:
            frame_count += 1
            if frame_count % attendance_threshold == 0:
                try:
                    if check_face(frame.copy()):
                        cv2.putText(frame, f"MATCH: {matched_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "NO MATCH", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except Exception as e:
                    print("Error in face verification:", e)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def export_csv_to_excel(csv_file, excel_file):
    df = pd.read_csv(csv_file)
    df.to_excel(excel_file, index=False)
    print(f"Attendance data exported to {excel_file}")

@app.route('/export')
def export():
    export_csv_to_excel(csv_file, "attendance.xlsx")
    return "Attendance data exported to attendance.xlsx"

@app.route('/attendance_list')
def attendance_list():
    attendance_list_str = "\nAttendance List:\n"
    for name, time in attendance.items():
        attendance_list_str += f"{name}: {time}\n"
    return attendance_list_str

if __name__ == '__main__':
    app.run(debug=True)
