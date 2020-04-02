import cv2
import numpy as np
import os

for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'faces-training')):
    if len(files) == 0:
        continue

    random = [1, 22, 32, 45, 88]

    for x in range(5):
        file_name = files[random[x]]

        print('Sifting ' + file_name)

        img = cv2.imread(os.path.join(root, file_name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        print(len(des))
        print(type(des[0]))
        print(des[0])
    

        # img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imwrite(os.path.join(os.getcwd(), 'sift', file_name), img)
