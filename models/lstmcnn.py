import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM
from keras.optimizers import SGD

training_data_dir = os.getcwd() + '/faces-training'
validation_data_dir = os.getcwd() + '/faces-validation'


def get_data(source_dir):
    image_files = []
    idx = 0
    for root, dirs, files in os.walk(source_dir, topdown=False):
        if len(files) == 0:
            continue

        image_files.append([])
        for name in files:
            image_files[idx].append(os.path.join(root, name))
        idx = idx + 1

    image_data_by_candidates = []
    image_label_by_candidates = []

    for x in range(0, len(image_files)):
        image_data = []
        image_label = []
        for image in image_files[x]:
            im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            
            if im is None:
                continue

            im = im/255
            data = np.array(im)
            label = get_label(image)

            image_data.append(data)
            image_label.append(label)

        image_data_by_candidates.append(np.array(image_data))
        image_label_by_candidates.append(np.array(image_label))

    return np.array(image_data), image_label


def get_label(image_file_name):
    length = len(image_file_name)
    base = image_file_name[length - 6:length]

    if base == '_0.jpg':
        return 0
    if base == '_5.jpg':
        return 1
    if base == '10.jpg':
        return 2

    print('Oops something went wrong!')
    exit(-1)


batch_size = 128
epochs = 20

input_shape = (360, 360)
num_classes = 3

training_data, training_target = get_data(training_data_dir)
validation_data, validation_target = get_data(validation_data_dir)



model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', 
                 #data_format="channels_last", 
                 input_shape=input_shape)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(32))
model.add(Dense(3, activation='softmax'))

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(training_data, training_target, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(validation_data, validation_target))
model.summary()


def plot_train(hist):
    h = hist.history
    if 'accuracy' in h:
        meas = 'accuracy'
        loc = 'lower right'
    else:
        meas = 'loss'
        loc = 'upper right'
    plt.plot(hist.history[meas])
    plt.plot(hist.history['val_'+meas])
    plt.title('model '+meas)
    plt.ylabel(meas)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc=loc)

plot_train(history)