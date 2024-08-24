from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import os
import csv
import time
from datetime import datetime
import numpy as np
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

# Define a distance threshold for recognizing a face
distance_threshold = 2600.0

# List to store recognized names
recognized_names = set()

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Get the nearest neighbors and their distances
        distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
        mean_distance = distances[0][0]

        if mean_distance < distance_threshold:
            output = knn.predict(resized_img)
            print(f"Recognized: {output[0]} with mean distance: {mean_distance}")
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
            rectangle_color = (0, 255, 0)  # Green for recognized faces
            text_color = (0, 255, 0)
            cv2.putText(frame, "Face Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display message

            # Add recognized name to the list
            recognized_names.add(output[0])
            attendance = [str(output[0]), str(timestamp)]
            name_text = str(output[0])
        else:
            rectangle_color = (0, 0, 255)  # Red for unrecognized faces
            text_color = (0, 0, 255)
            name_text = "Unknown"
            print(f"Face not recognized. Mean distance: {mean_distance}")

        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), rectangle_color, 2)
        cv2.putText(frame, name_text, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    # Display the list of recognized names on the right side of the screen
    y0, dy = 50, 30
    for i, name in enumerate(recognized_names):
        cv2.putText(frame, f"{i + 1}. {name}", (frame.shape[1] - 200, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
