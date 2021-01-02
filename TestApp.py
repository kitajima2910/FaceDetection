import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

# Training
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("recognizer/trainingData.yml")

# Connection
conn = sqlite3.connect("data.db")

# Get Profile With ID
def getProfile(id):
    c = conn.cursor()
    
    sql = "select * from people where id = " + str(id)
    cursor = c.execute(sql)

    profile = None

    for row in cursor:
        profile = row

    c.close()
    return profile

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fontface = cv2.FONT_HERSHEY_SIMPLEX

while(True):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        gray_clone = gray[y: y + h, x: x + w]

        id, confidence = recognizer.predict(gray_clone)

        #if confidence < 40:
        profile = getProfile(id)

        print("profile: " + str(profile))

        if((profile != None)):
            if(round(100 - confidence) > 40):
                cv2.putText(frame, "" + str(profile[1]) + " {0}%".format(round(100 - confidence)), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2) 
            else:
                cv2.putText(frame, "Unknow", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknow", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)

    cv2.imshow("App", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
