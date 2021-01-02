import cv2, os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

path = "dataSet"

def getImageWithID(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    # print(imagePaths)

    faceSamples = []
    IDs = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert("L")
        faceNP = np.array(faceImg, "uint8")

        # print(faceNP)

        id = int(imagePath.split('.')[1])
        faces = face_cascade.detectMultiScale(faceNP)

        for (x, y, w, h) in faces:
            faceSamples.append(faceNP[y: y + h, x: x + w])
            IDs.append(id)

        cv2.imshow("Training", faceNP)
        cv2.waitKey(10)
    
    return faceSamples, np.array(IDs)

faceSamples, IDs = getImageWithID(path)

print(len(faceSamples))
print(len(IDs))

recognizer.train(faceSamples, IDs)

if not os.path.exists("recognizer"):
    os.makedirs("recognizer")

recognizer.save("recognizer/trainingData.yml")

cv2.destroyAllWindows()
