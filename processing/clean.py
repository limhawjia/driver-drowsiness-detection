import os
import sys
import shutil


def create_output_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, '0')):
        os.makedirs(os.path.join(output_dir, '0'))

    if not os.path.exists(os.path.join(output_dir, '5')):
        os.makedirs(os.path.join(output_dir, '5'))

    if not os.path.exists(os.path.join(output_dir, '10')):
        os.makedirs(os.path.join(output_dir, '10'))


def get_values(file_name):
    try:
        name, _ = os.path.splitext(file_name)
        candidate, frame, label = name.split('_')
        return int(candidate), int(frame), int(label)
    except ValueError:
        print('File "' + file_name + '" in incorrect format')
        exit(1)


def split_by_label_and_sort_by_frame(files_names):
    values = map(get_values, files_names)
    label_0 = sorted(filter(lambda x: x[2] == 0, values), key=lambda x: x[1])
    label_5 = sorted(filter(lambda x: x[2] == 5, values), key=lambda x: x[1])
    label_10 = sorted(filter(lambda x: x[2] == 10, values), key=lambda x: x[1])

    return list(label_0), list(label_5), list(label_10)


def get_every_tenth_frame(values):
    return list(filter(lambda x: x[1] % 10 == 0, values))


def retrieve_file_paths(root, value, output_dir):
    candidate = str(value[0]) if value[0] > 10 else '0' + str(value[0])
    old_file_name = candidate + '_' + \
        str(value[1]) + '_' + str(value[2]) + '.jpg'
    new_file_name = candidate + '_' + \
        str(value[1] / 10) + '_' + str(value[2]) + '.jpg'
    return os.path.join(root, old_file_name), os.path.join(output_dir, str(value[2]), new_file_name)


def get_file_mappings(root, files_names, output):
    label_0, label_5, label_10 = split_by_label_and_sort_by_frame(files_names)

    label_0 = map(lambda x: retrieve_file_paths(
        root, x, output), get_every_tenth_frame(label_0))
    label_5 = map(lambda x: retrieve_file_paths(
        root, x, output), get_every_tenth_frame(label_5))
    label_10 = map(lambda x: retrieve_file_paths(
        root, x, output), get_every_tenth_frame(label_10))

    return list(label_0), list(label_5), list(label_10)


def clean(root, file_names, output):
    label_0, label_5, label_10 = get_file_mappings(root, file_names, output)

    for old_file, new_file in label_0:
        shutil.copyfile(old_file, new_file)
    for old_file, new_file in label_5:
        shutil.copyfile(old_file, new_file)
    for old_file, new_file in label_10:
        shutil.copyfile(old_file, new_file)


def main():
    target_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(target_dir):
        print('Please enter a valid target directory')
        exit(1)

    if not os.path.isabs(target_dir):
        target_dir = os.path.abspath(target_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    create_output_dirs(output_dir)

    for root, dirs, files in os.walk(target_dir):
        clean(root, files, output_dir)


if __name__ == '__main__':
    main()
