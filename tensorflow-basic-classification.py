#!/usr/bin/env python3

# TensorFlow Basic Classification Tutorial
# https://www.tensorflow.org/tutorials/keras/basic_classification

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

"""
Labels:
    0 - T-shirt/top
    1 - Trouser
    2 - Pullover
    3 - Dress
    4 - Coat
    5 - Sandal
    6 - Shirt
    7 - Sneaker
    8 - Bag
    9 - Ankle boot
"""
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# print(train_images.shape) # (60000, 28, 28)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# scale the image pixel values between 0 and 1
scale_from_0_to_1 = lambda x: x / 255.0
train_images = scale_from_0_to_1(train_images)
test_images = scale_from_0_to_1(test_images)

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# each item in the list is a layer in the neural network
model = keras.Sequential([
    # flattens the 2D images data into a 1D array of 784 neurons
    keras.layers.Flatten(input_shape=(28, 28)),
    # dense (fully connected to previous layer), hidden layer with 128 neurons
    keras.layers.Dense(128, activation=tf.nn.relu),
    # final, dense layer, array of 10 probabilities that sum to 1, individual probabilities singifying likelihood that the input image is in that class
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(
    # how the model is updated based on the input data and result of the loss function
    optimizer='adam',
    # loss function, measures how accurate the model is. need to minize this
    loss='sparse_categorical_crossentropy',
    # array of metrics to monitor training and testing steps. in this case, monitor accuracy (percent correcnly identified).
    metrics=['accuracy']
)

# train the model
model.fit(train_images, train_labels, epochs=5)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

# make predictions
# makes a prediction for each of the test images
predictions = model.predict(test_images)
# Prints and array of ten confidence values (one for each label), total sums to one.
print(predictions[0])
# Gets the max prediction value, i.e. the predicted result.
print(np.argmax(predictions[0]))
# Checks that the actual and predicted values are the same.
assert(test_labels[0] == np.argmax(predictions[0]))

# plot the image and some statistics
def plot_image(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100 * np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def plot_stats_for_ith_result(i=0):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(predictions[i], test_labels[i], test_images[i])
    plt.subplot(1, 2, 2)
    plot_value_array(predictions[i], test_labels[i])
    plt.show()

plot_stats_for_ith_result()
plot_stats_for_ith_result(12)

def plot_stats_for_multiple_results(num_rows=1, num_cols=1):
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(predictions[i], test_labels[i], test_images[i])
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(predictions[i], test_labels[i])
    plt.show()

plot_stats_for_multiple_results(5, 3)

# make a prediction on a single image
img = test_images[0]
# tf needs to process and array of inputs, so add an additional dimension to the image at index 0.
img = (np.expand_dims(img, 0))
prediction_single = model.predict(img)
print(prediction_single)

plot_value_array(prediction_single[0], test_labels[0])
plt.xticks(range(10), class_names, rotation=45)
plt.show()
