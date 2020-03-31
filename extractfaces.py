import os
import cv2
import imutils

IMAGE_SRC_DIR = '/home/hawjiaa/Projects/DriverDrowsiness/test'
FACE_DEST_DIR = '/home/hawjiaa/Projects/DriverDrowsiness/faces'


def get_image_paths(folder):
    filenames = os.listdir(folder)
    image_paths = []
    for filename in filenames:
        image_paths.append(folder + '/' + filename)

    return image_paths


def extract_face(image_path):
    print('Processing image: ' + image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image, face_rectangle = find_face(image)
    # Face rectangle returned in the form (x, y, w, h) where x and y are the coordinates of the bottom left point
    # and w and h are the width and height of the rectangle
    if image is None:
        print('Skipping write')
        return
    x, y, w, h = face_rectangle
    image = image[y:y + h, x:x + w]
    image = cv2.resize(image, (360, 360))

    cv2.imwrite(FACE_DEST_DIR + '/' + os.path.basename(image_path), image)
    print('Image written')


def find_face(image):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rotations = 0
    face_detected = False
    while rotations <= 4 and not face_detected:
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
        if len(faces) != 0:
            print('Face detected')
            x, y, w, h = faces[0]
            print(x, y, w, h)
            return image, (x, y, w, h)

        image = imutils.rotate_bound(image, 90)
        rotations = rotations + 1
    print('No face detected')
    return None, None


def main():
    for candidate_folder in os.listdir(IMAGE_SRC_DIR):
        print('Extracting faces from candidate folder: ' + candidate_folder)
        image_paths = get_image_paths(IMAGE_SRC_DIR + '/' + candidate_folder)
        for image_path in image_paths:
            try:
                extract_face(image_path)
            except AttributeError:
                print('Image corrupted')
                continue


if __name__ == '__main__':
    main()
