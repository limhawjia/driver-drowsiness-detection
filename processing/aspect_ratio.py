import csv
import os
import math
from scipy.spatial import distance
from facial import get_facial_landmarks

HOME_DIR = '/home/mlsg/driver-drowsiness-detection'
DISKA_DIR = '/home/mlsg/DiskA'
DISKB_DIR = '/home/mlsg/DiskB'

# Takes in the landmark
# Returns two values, the first being the left eye and the second the right eye
def calc_eyes_aspect_ratio(landmarks):
    left_eye = []
    right_eye = []
    for i in range(36, 42):
        left_eye.append([landmarks.part(i).x, landmarks.part(i).y])
    for i in range(42, 48):
        right_eye.append([landmarks.part(i).x, landmarks.part(i).y])
    return calc_eye_aspect_ratio(left_eye), calc_eye_aspect_ratio(right_eye)

def calc_eye_aspect_ratio(coords):
    height1 = distance.euclidean(coords[1], coords[5])
    height2 = distance.euclidean(coords[2], coords[4])
    length = distance.euclidean(coords[0], coords[3])
    return (height1 + height2) / (2 * length)

def calc_circularities(landmarks):
    left_eye = []
    right_eye = []
    for i in range(36, 42):
        left_eye.append([landmarks.part(i).x, landmarks.part(i).y])
    for i in range(42, 48):
        right_eye.append([landmarks.part(i).x, landmarks.part(i).y])
    return calc_circularity(left_eye), calc_circularity(right_eye)

# Array of tuples of x and y values
def calc_circularity(coords):
    radius = distance.euclidean(coords[0], coords[3])
    area = math.pi * (radius ** 2)
    perimeter = 0
    perimeter += distance.euclidean(coords[0], coords[1])
    perimeter += distance.euclidean(coords[1], coords[2])
    perimeter += distance.euclidean(coords[2], coords[3])
    perimeter += distance.euclidean(coords[3], coords[4])
    perimeter += distance.euclidean(coords[4], coords[5])
    perimeter += distance.euclidean(coords[5], coords[0])
    return 4 * math.pi * area / (perimeter ** 2)

# File path here is one of the folders containing the 3 videos
def create_csv_files(file_path, d_name):
    # Assumes tha the 3 folders are already there
    dirname = os.path.dirname
    disk_path = dirname(dirname(file_path))
    if not os.path.exists(os.path.join(disk_path, 'ratios', d_name)):
        os.makedirs(os.path.join(disk_path, 'ratios', d_name))

    label_0 = []
    label_5 = []
    label_10 = []

    for image in os.listdir(file_path):
        image_file_path = os.path.join(file_path, image)
        _, _, result = image.split("_")
        result = result.split(".")[0]
        if result == "0":
            label_0.append(image_file_path)
        if result == "5":
            label_5.append(image_file_path)
        else:
            label_10.append(image_file_path)
    create_csv_file(label_0, os.path.join(disk_path, 'ratios', d_name, '0'))
    create_csv_file(label_5, os.path.join(disk_path, 'ratios', d_name, '5'))
    create_csv_file(label_10, os.path.join(disk_path, 'ratios', d_name, '10'))


def create_csv_file(images, output_path):
    with open(output_path + '.csv', 'w', newline='') as f:
        print(output_path)
        writer = csv.writer(f)
        for image in images:
            name, _ = os.path.splitext(image)
            _, frame, _ = name.split('_')
            landmarks = get_facial_landmarks(image)
            if landmarks is None:
                print("No landmark")
                continue

            print(image)
            left_ratio, right_ratio = calc_eyes_aspect_ratio(landmarks)
            left_circularity, right_circularity = calc_circularities(landmarks)
            writer.writerow([
                frame,
                left_ratio,
                right_ratio,
                left_circularity,
                right_circularity
            ])


if __name__ == "__main__":
    # Create the output directory for each of the paths
    if not os.path.exists(os.path.join(DISKA_DIR, 'others', 'ratios')):
        os.makedirs(os.path.join(DISKA_DIR, 'ratios'))

    if not os.path.exists(os.path.join(DISKB_DIR, 'others', 'ratios')):
        os.makedirs(os.path.join(DISKB_DIR, 'ratios'))

    for d in os.listdir(os.path.join(DISKA_DIR, 'others', 'faces')):
        create_csv_files(os.path.join(DISKA_DIR, 'others', 'faces', d), d)

    for d in os.listdir(os.path.join(DISKB_DIR, 'others', 'faces')):
        create_csv_files(os.path.join(DISKB_DIR, 'others', 'faces', d), d)
