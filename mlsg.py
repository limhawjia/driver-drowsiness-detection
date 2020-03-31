import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model

from PIL import Image

IMAGE_SRC_DIR = '/mnt/disks/a/frames/'

# Declare dictionary to store image and label
image_label_dict = {}

# Iterate through the folders of individual candidates, obtain the image as pixels and the label, store to dictionary
root, candidates, _ = os.walk(IMAGE_SRC_DIR)[0]
for candidates in candidates:
    dir_path = root + candidates
    print(dir_path)

    images = os.listdir(dir_path)
    for image in images:
        im = Image.open(image)
        label = image[len(image) - 5]  # minus 5 as the files end with .png
        pix_val = list(im.getdata())
        image_label_dict[image] = (pix_val, label)

# This part contains sample code for the training
'''
# Training Parameters for basic MNIST
learning_rate = 0.001
training_epochs = 10
batch_size = 100 # How many images we want it to process at any given time

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28 flattened to be 784)
n_hidden_1 = 384 # 1st layer number of neurons
n_hidden_2 = 100 # 2nd layer number of neurons
n_classes = 10 # MNIST classes for prediction(digits 0-9 )

# The model
Inp = Input(shape = (n_input,))
x = Dense(n_hidden_1, activation='relu', name = "Dense_1")(Inp)
x = Dense(n_hidden_2, activation='relu', name = "Dense_2")(x)
output = Dense(n_classes, activation='softmax', name = "OutputLayer")(x) # Softmax ensures all the numbers add up to 1

# This creates a model that includes our input, 2 dense hidden layers, output layer
model = Model(Inp, output, name = "our_dense_model")

model.summary()

opt = SGD(lr=learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''
