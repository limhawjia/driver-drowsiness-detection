import os
import csv
import statistics

DISKA_DIR = '/home/mlsg/DiskA'
DISKB_DIR = '/home/mlsg/DiskB'
ROW_LENGTH = 60
THRESHOLD_VALUE = -1

def check_blink_rate(row):
    if float(row[5]) < THRESHOLD_VALUE:
        return 1
    else:
        return 0

def get_blink_count(new_row):
    return sum([check_blink_rate(row) for row in new_row])

def get_blink_ratio(new_row):
    return sum([check_blink_rate(row) for row in new_row]) / ROW_LENGTH

def get_svm_values(path):
    disk_ratios = path + "/others/ratios"
    for directory in os.listdir(disk_ratios):
        for ending in ['0', '10']:
            try:
                rfile = disk_ratios + '/' + directory + '/' + ending + '.csv'
                rows = []
                with open(rfile, 'r', newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        rows.append(row)
                rows.sort(key=lambda x: int(x[0]))

                if not rows:
                    continue

                new_rows = []
                # Now do the processing
                for i in range(len(rows) - ROW_LENGTH):
                    new_rows.append(rows[i: i + ROW_LENGTH])

                new_file = disk_ratios + '/' + directory + '/' + ending + '_svm.csv'
                with open(new_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for new_row in new_rows:
                        writer.writerow([
                            statistics.mean([float(row[5]) for row in new_row]),
                            statistics.stdev([float(row[5]) for row in new_row]),
                            statistics.mean([float(row[7]) for row in new_row]), # mouth aspect ratio
                            get_blink_count(new_row),
                            get_blink_ratio(new_row)
                        ])
            except:
                print(rfile)



if __name__ == "__main__":
    get_svm_values(DISKA_DIR)
    get_svm_values(DISKB_DIR)

