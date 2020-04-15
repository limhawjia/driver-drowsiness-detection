import os
import cv2
import imutils

IMAGE_SRC_DIR = ['/home/mlsg/DiskA/frames', '/home/mlsg/DiskB/frames']
FACE_DEST_DIR = ['/home/mlsg/DiskA/others/faces', '/home/mlsg/DiskB/others/faces']


def get_image_paths(folder):
    image_paths = []
    for res_dir in os.listdir(folder):
        intermediate_dir = folder + '/' + res_dir
        for filename in os.listdir(intermediate_dir):
            image_paths.append(intermediate_dir + '/' + filename)

    return image_paths


def extract_face(image_path, output_folder):
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
    # image = cv2.resize(image, (360, 360))

    output_path = output_folder + '/' + os.path.basename(image_path)
    cv2.imwrite(output_path, image)
    print('Image written to: ' + output_path)


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
    for i in range(2):
        for candidate_folder in os.listdir(IMAGE_SRC_DIR[i]):
            print('Extracting faces from candidate folder: ' + candidate_folder)
            image_paths = get_image_paths(IMAGE_SRC_DIR[i] + '/' + candidate_folder)
            for image_path in image_paths:
                try:
                    output_folder = FACE_DEST_DIR[i] + '/' + candidate_folder
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    extract_face(image_path, output_folder)
                except AttributeError:
                    print('Image corrupted')
                    continue

if __name__ == '__main__':
    main()
