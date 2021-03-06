import cv2
import os
import sys

output_dir = './frames'
# Percentage of frames to capture
rate = 0.16


def process():
    if len(sys.argv) < 2:
        print('Usage: python process.py [source dir]')
        exit(1)

    video_dir = sys.argv[1]

    sub_dirs = [f.path for f in os.scandir(video_dir) if f.is_dir()]
    for folder in sub_dirs:
        process_folder(folder)


def process_folder(folder_path):
    print('Processing folder: ' + folder_path)

    output_path = os.path.join(os.path.realpath(output_dir), get_filename(folder_path))
    print('Extracting videos to: ' + output_path)

    videos = []
    for (_, _, files) in os.walk(folder_path):
        videos.extend(files)
        break

    for video in videos:
        path = os.path.join(os.path.realpath(folder_path), video)
        name = get_filename(path)

        dir_name = os.path.join(os.path.realpath(output_dir), output_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        extract_frames(path, name, get_filename(folder_path), output_path)


def extract_frames(video_path, video_name, candidate_number, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Unable to open video')
        return

    frame_count = 0
    index = 0

    print('Successfully opened video file: ' + video_name + ' , starting frame extraction')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    while cap.isOpened():
        if index % round(1 / rate) != 0:
            index += 1
            continue

        check, frame = cap.read()

        if not check:
            print('Frame ' + str(index) + ' end of video')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_count += 1
        frame_name = candidate_number + '_' + str(frame_count) + '_' + video_name + '.jpg'
        frame_path = os.path.join(output_path, frame_name)

        cv2.imwrite(frame_path, frame)

        index += 1

    cap.release()
    cv2.destroyAllWindows()


def get_filename(file_path):
    stripped = os.path.splitext(file_path)[0]
    name = os.path.split(stripped)[1]
    return name


if __name__ == '__main__':
    process()
