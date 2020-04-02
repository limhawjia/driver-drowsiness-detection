import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

training_data_dir = os.getcwd() + '/faces-training'
validation_data_dir = os.getcwd() + '/faces-validation'


def get_data(source_dir):
    image_files = []
    for root, dirs, files in os.walk(source_dir, topdown=False):
        for name in files:
            image_files.append(os.path.join(root, name))

    image_data = []
    image_label = []
    for image in image_files:
        im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        
        if im is None:
            continue

        im = im/255
        data = np.array(im)
        flattened = data.flatten()
        label = get_label(image)

        image_data.append(flattened)
        image_label.append(label)

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
epochs = 50
learning_rate = 0.000001

input_size = 129600
hidden_layers_size = 512
num_classes = 3

training_data, training_target = get_data(training_data_dir)
validation_data, validation_target = get_data(validation_data_dir)

training_target = keras.utils.to_categorical(training_target, num_classes)
validation_target = keras.utils.to_categorical(validation_target, num_classes)

model = Sequential()
model.add(Dense(hidden_layers_size, activation='relu', input_shape=(input_size, )))
# model.add(Dropout(0.2))
model.add(Dense(hidden_layers_size, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=learning_rate),
              metrics=['accuracy'])

model.summary()

history = model.fit(training_data, training_target, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(validation_data, validation_target))


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
    plt.savefig('mlp.png')

plot_train(history)