import csv
import os
import math
import statistics
import cv2
from scipy.spatial import distance
from mlxtend.image import extract_face_landmarks
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

def calc_mouth_aspect_ratio(landmarks):
    left = (landmarks.part(48).x, landmarks.part(48).y)
    right = (landmarks.part(54).x, landmarks.part(54).y)
    top = (landmarks.part(51).x, landmarks.part(51).y)
    bottom = (landmarks.part(57).x, landmarks.part(57).y)
    return distance.euclidean(top, bottom) / distance.euclidean(left, right)

def mar_over_ear(mar, ear):
    return mar / ear

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
    if not os.path.exists(os.path.join(disk_path, 'ratios-new', d_name)):
        os.makedirs(os.path.join(disk_path, 'ratios-new', d_name))

    label_0 = []
    label_5 = []
    label_10 = []

    for image in os.listdir(file_path):
        image_file_path = os.path.join(file_path, image)
        _, _, result = image.split("_")
        result = result.split(".")[0]
        if result == "0":
            label_0.append(image_file_path)
        elif result == "5":
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
            mean_ratio = (left_ratio + right_ratio) / 2
            mean_circularity = (left_circularity + right_circularity) / 2
            mouth_aspect_ratio = calc_mouth_aspect_ratio(landmarks)
            mar_ear_ratio = mar_over_ear(mouth_aspect_ratio, mean_ratio)
            writer.writerow([
                frame,
                mean_ratio,
                mean_circularity,
                mouth_aspect_ratio,
                mar_ear_ratio,
            ])

def calc_norm_value(value, mean, stdev):
    return (value - mean) / stdev

# This function performs "normalization" on the aspect ratio
# Assumes that you already have already created the csv files
# TRIGGER WARNING: SPAGHETTI CODE
def normalization(path):
    disk_ratios = path + "/others/ratios-new"
    for directory in os.listdir(disk_ratios):
        mean_ratio = []
        mean_circularity = []
        mouth_aspect_ratio = []
        mar_ear_ratio = []
        rows = []
        # First open the 0.csv file
        first_file = disk_ratios + '/' + directory + '/0.csv'
        with open(first_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
                mean_ratio.append(float(row[1]))
                mean_circularity.append(float(row[2]))
                mouth_aspect_ratio.append(float(row[3]))
                mar_ear_ratio.append(float(row[4]))

        if not rows or len(rows) < 30:
            continue

        avg_zero_mean_ratio = sum(mean_ratio[:30]) / len(mean_ratio[:30])
        avg_zero_mean_circularity = sum(mean_circularity[:30]) / len(mean_circularity[:30])
        avg_zero_mouth_aspect_ratio = sum(mouth_aspect_ratio[:30]) / len(mouth_aspect_ratio[:30])
        avg_zero_mar_ear_ratio = sum(mar_ear_ratio[:30]) / len(mar_ear_ratio[:30])
        stdev_mean_ratio = statistics.stdev(mean_ratio[:30])
        stdev_mean_circularity = statistics.stdev(mean_circularity[:30])
        stdev_mouth_aspect_ratio = statistics.stdev(mouth_aspect_ratio[:30])
        stdev_mar_ear_ratio = statistics.stdev(mar_ear_ratio[:30])

        with open(first_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(rows)):
                writer.writerow(rows[i] + [
                    calc_norm_value(mean_ratio[i], avg_zero_mean_ratio, stdev_mean_ratio),
                    calc_norm_value(mean_circularity[i], avg_zero_mean_circularity, stdev_mean_circularity),
                    calc_norm_value(mouth_aspect_ratio[i], avg_zero_mouth_aspect_ratio, stdev_mouth_aspect_ratio),
                    calc_norm_value(mar_ear_ratio[i], avg_zero_mar_ear_ratio, stdev_mar_ear_ratio)
                ])

        print("Done processing the 0 file")

        for ending in ['5.csv', '10.csv']:
            next_file = disk_ratios + '/' + directory + '/' + ending
            print(next_file)

            rows = []

            with open(next_file, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(row)

            with open(next_file, 'w', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(rows)):
                    writer.writerow(rows[i] + [
                        calc_norm_value(float(rows[i][1]), avg_zero_mean_ratio, stdev_mean_ratio),
                        calc_norm_value(float(rows[i][2]), avg_zero_mean_circularity, stdev_mean_circularity),
                        calc_norm_value(float(rows[i][3]), avg_zero_mouth_aspect_ratio, stdev_mouth_aspect_ratio),
                        calc_norm_value(float(rows[i][4]), avg_zero_mar_ear_ratio, stdev_mar_ear_ratio)
                    ])
        print("Done processing other files")

if __name__ == "__main__":
    # Create the output directory for each of the paths
    # if not os.path.exists(os.path.join(DISKA_DIR, 'others', 'ratios')):
        # os.makedirs(os.path.join(DISKA_DIR, 'ratios'))

    # if not os.path.exists(os.path.join(DISKB_DIR, 'others', 'ratios')):
        # os.makedirs(os.path.join(DISKB_DIR, 'ratios'))

    for d in os.listdir(os.path.join(DISKA_DIR, 'others', 'faces')):
        create_csv_files(os.path.join(DISKA_DIR, 'others', 'faces', d), d)

    for d in os.listdir(os.path.join(DISKB_DIR, 'others', 'faces')):
        create_csv_files(os.path.join(DISKB_DIR, 'others', 'faces', d), d)
    normalization(DISKA_DIR)
    normalization(DISKB_DIR)
