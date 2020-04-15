import cv2
import numpy as np
import os
import dlib
from extractfaces import find_face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'faces-training')):
    if len(files) == 0:
        continue

    random = [1, 22, 32, 45, 88]

    for x in range(5):
        file_name = files[random[x]]

        print('Facial recognizing ' + file_name)

        img = cv2.imread(os.path.join(root, file_name))
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # faces = detector(gray)
        # for face in faces:
            # x1 = face.left()
            # y1 = face.top()
            # x2 = face.right()
            # y2 = face.bottom()
            # #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # landmarks = predictor(gray, face)

        landmarks = get_facial_landmarks(os.path.join(root, file_name))
        if landmarks is None:
            continue

        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

        cv2.imwrite(os.path.join(os.getcwd(), 'facial', file_name), img)

def get_facial_landmarks(file_path):
    image = cv2.imread(file_path)

    if image is None:
        print("None")
        return None
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(image)

    if not faces:
        return None

    face = faces[0]
    return predictor(image, face)

