from scipy.spatial import distance
import math

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

