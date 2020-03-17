import cv2
import os

video_dir = './videos'
output_dir = './frames'
# Percentage of frames to capture
rate = 0.5


def process():
    videos = []
    for (_, _, files) in os.walk(video_dir):
        videos.extend(files)
        break

    for video in videos:
        path = os.path.join(os.path.realpath(video_dir), video)
        name = get_filename(path)
        extract_frames(path, name)


def extract_frames(video_path, video_name):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Unable to open video')
        exit(1)

    dir_name = os.path.join(os.path.realpath(output_dir), video_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    frame_count = 0
    index = 0
    while cap.isOpened() and index < 20:
        if index % round(1/rate) != 0:
            index += 1
            continue

        check, frame = cap.read()

        if not check:
            print('Frame could not be extracted')
            exit(1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_count += 1
        frame_name = str(frame_count) + '.jpg'
        frame_path = os.path.join(dir_name, frame_name)

        cv2.imwrite(frame_path, frame)

        index += 1

    cap.release()
    cv2.destroyAllWindows()


def get_filename(video_path):
    stripped = os.path.splitext(video_path)[0]
    name = os.path.split(stripped)[1]
    return name


if __name__ == '__main__':
    process()
