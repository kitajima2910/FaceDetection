import cv2
import numpy as np
import sqlite3
import os

conn = sqlite3.connect("data.db")

def insertOrUpdate(id, name):
    
    cursor = conn.cursor()

    sql = "select * from people where id = " + str(id)
    cursor = conn.execute(sql)

    isRecordExits = 0
    for row in cursor:
        isRecordExits = 1
    
    if(isRecordExits == 0):
        sql = "insert into people(id, name) values(" + str(id) + ",  '" + str(name) + "')"
    else:
        sql = "update people set name = '" + str(name) + "' where id = " + str(id) 

    conn.execute(sql)
    conn.commit()
    conn.close()


# Load lib    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Insert to database
id = input("Enter Your ID: ")
name = input("Enter Your Name: ")
insertOrUpdate(id, name)

index = 0

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not os.path.exists("dataSet"):
            os.makedirs("dataSet")
        
        index += 1

        cv2.imwrite("dataSet/User." + str(id) + "." + str(index) + ".jpg", gray[y: y + h, x: x + w])
    
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    if(index >= 200):
        break

cap.release()
cv2.destroyAllWindows()
