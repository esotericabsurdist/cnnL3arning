#===============================================================================

# Robert Mitchell
# 1/31/2018
#
# Content from:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/03C_Keras_API.ipyn
#
#
#
#===============================================================================
'''                             Imports                                     '''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

# this pulls test data from tensoflow's website and saves it in a library?
from tensorflow.examples.tutorials.mnist import input_data

# import an optimizer for back propagation. this will be used after we define
# the model below.
from tensorflow.python.keras.optimizers import Adam
optimizer = Adam(lr=1e-3)

#===============================================================================
'''                         Define our data                                  '''
# this will hold our test data. how do we use our own images?
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

# display the details of our data
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# convert our vectorized classes to digits
data.test.cls = np.argmax(data.test.labels, axis=1)

#===============================================================================
'''               Define our image dimensions as variables                   '''

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_size, img_size)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, 1)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

#===============================================================================
'''               Helper Function To Verify Image Data                       '''
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


#===============================================================================

'''
    Uncomment this section to print a sample of the test data to verify that the
    images have been downloaded successfully.
'''

# # Get the first images from the test-set.
# images = data.test.images[0:9]
#
# # Get the true classes for those images.
# cls_true = data.test.cls[0:9]
#
# # Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true)

#===============================================================================

'''          Define the neural network using the Keras interface              '''

# Start construction of the Keras Sequential model.
model = Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size.
model.add(InputLayer(input_shape=(img_size_flat,)))

# The input is a flattened array with 784 elements,
# but the convolutional layers expect images with shape (28, 28, 1)
model.add(Reshape(img_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the 4-rank output of the convolutional layers
# to 2-rank that can be input to a fully-connected / dense layer.
model.add(Flatten())

# First fully-connected / dense layer with ReLU-activation.
model.add(Dense(128, activation='relu'))

# Last fully-connected / dense layer with softmax-activation
# for use in classification.
model.add(Dense(num_classes, activation='softmax'))

#===============================================================================
'''             Build the network now that it has been defined               '''

# Keras calls building the network, 'compilation' --> we do that here:
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#===============================================================================

'''                         Train the model !                                '''


# a single epoch is one full iteration over all the training images.
# the number of training images to iterate over in this case is 128,
# the batch size.
model.fit(x=data.train.images, y=data.train.labels, epochs=1, batch_size=128)



#===============================================================================

'''            Print the performance of the model on the training set        '''


# The following line
result = model.evaluate(x=data.test.images, y=data.test.labels)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))



#===============================================================================

'''     FINALY!!!! Send to some data through the network and classify it     '''

# get some images from the test imageset to use as input.
images = data.test.images[0:9]

# cls_true AKA 'class truth', holds the correct (true) classification values for
# the the images we are interested in, the first 9 images in the set.
cls_true = data.test.cls[0:9]

# the momement we've been waiting for!
y_pred = model.predict(x=images)

# but wait, we can't read the output stored in 'y_pred', change the predictions
# for the images into integers
cls_pred = np.argmax(y_pred,axis=1)

# plot the output
plot_images(images=images,
            cls_true=cls_true,
            cls_pred=cls_pred)
