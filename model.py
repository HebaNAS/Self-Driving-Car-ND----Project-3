import pickle
import tensorflow as tf
tf.python.control_flow_ops = tf

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from keras import backend as K
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
from numpy import random
import matplotlib.image as mpimg
import csv
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Activation, Convolution2D, MaxPooling2D, ELU, Dropout
from keras.models import Sequential, Model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'data/driving_log.csv', "Data file")
flags.DEFINE_string('images_folder', 'data/', "Images Folder")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_integer('steering_adjustment', 0.25, "Steering adjustment")


def load_training_data(training_file):
    """
    Utility function to load training file and extract features and labels.

    Arguments:
        training_file - String

    Output:
        numpy arrays of input data split into training and validation sets
    """
    K.set_image_dim_ordering('tf')

    # Read data file, create empty arrays to hold various data
    colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(training_file, skiprows=[0], names=colnames)
    data_center = []
    data_left = []
    data_right = []
    steer_straight = []
    steer_left = []
    steer_right = []  

    # Create lists from data
    imgs_center = data.center.tolist()
    imgs_recover = data.center.tolist() 
    imgs_left = data.left.tolist()
    imgs_right = data.right.tolist()
    steering = data.steering.tolist()
    steering_recover = data.steering.tolist()

    # Split data to get un-corrected validation data
    imgs_center, steering = shuffle(imgs_center, steering)
    train_img, X_val, train_str, y_val = train_test_split(imgs_center, steering, test_size=0.15, random_state=42)

    # Filter steering values to left and right arrays
    for i in steering:
        index = steering.index(i)
        # For angles > 0.15, the car is turning right
        if i > 0.15:
            data_right.append(str(imgs_center[index].strip()))
            steer_right.append(i)
        # For angles > -0.15, the car is turning left
        if i < -0.15:
            data_left.append(str(imgs_center[index].strip()))
            steer_left.append(i)
        # For angles between -0.15 and 0.15, the car is driving straight
        else:
            data_center.append(str(imgs_center[index].strip()))
            steer_straight.append(i)

    #  Find difference between driving straight & driving left, driving straight & driving right 
    imgs_center_no, imgs_left_no, imgs_right_no = len(data_center), len(data_left), len(data_right)
    total = len(imgs_recover)
    unbalanced_left = imgs_center_no - imgs_left_no
    unbalanced_right = imgs_center_no - imgs_right_no
 
    # Generate random list for left and right recovery images
    left_recovery = np.random.uniform(range(total), unbalanced_left).astype(int)
    right_recovery = np.random.uniform(range(total), unbalanced_right).astype(int)

    # Loop through recovery images, for angles < -0.15, the car is soft steering to the left,
    # so we subtract an adjustment value and append the image to the corresponding array
    for i in left_recovery:
        if steering_recover[i] < -0.15:
            data_left.append(str(imgs_right[i].strip()))
            steer_left.append(steering_recover[i] - FLAGS.steering_adjustment)

    # Loop through recovery images, for angles > 0.15, the car is soft steering to the right,
    # so we add an adjustment value and append the image to the corresponding array
    for i in right_recovery:
        if steering_recover[i] > 0.15:
            data_right.append(str(imgs_left[i].strip()))
            steer_right.append(steering_recover[i] + FLAGS.steering_adjustment)
   
    # Add a step for balancing right, left and straight data
    data_left = shuffle(np.repeat(data_left, 3)).tolist()
    data_right = shuffle(np.repeat(data_right, 6)).tolist()
    steer_left = shuffle(np.repeat(steer_left, 3)).tolist()
    steer_right = shuffle(np.repeat(steer_right, 6)).tolist()
    print (len(data_center), len(data_left), len(data_right))
    print (len(steer_straight), len(steer_left), len(steer_right))  

    # Collect all data into X_train, y_train arrays
    X_train = data_center + data_left + data_right
    y_train = np.float32(steer_straight + steer_left + steer_right)

    plt.hist(y_train, bins='auto')
    plt.savefig('hist.png')
    #plt.show()

    return (X_train, y_train, X_val, y_val)

def adjust_image(img):
    """
    Function that takes an image as input and returns an image as output.
    The function crops and resizes images.
    """
    image = cv2.resize(img[70:140,:], (200, 66))
    image = image.reshape(-1, 66, 200, 3)

    return image

def flip_image(img, ang):
    """
    Function that takes an image and angle as input and returns flipped image and angles as output.
    """
    image = cv2.flip(img, 1)
    angle = ang * -1.0
  
    return image, angle

def data_generator(X_train, y_train):
    """
    Function that creates batches of data to save computational resources while training.
    It takes training data array, which contains a list of images, and corresponding steering angles array.
    """
    train = np.zeros((FLAGS.batch_size, 66, 200, 3), dtype = np.float32)
    angle = np.zeros((FLAGS.batch_size,), dtype = np.float32)

    while True:
        data, ang = shuffle(X_train, y_train)

        for i in range(FLAGS.batch_size):
          choice = int(np.random.choice(len(data), 1))
          if data[choice] != None:
              train[i] = adjust_image(mpimg.imread(FLAGS.images_folder + data[choice].strip()))
              angle[i] = ang[choice] * (1 + np.random.uniform(-0.10,0.10))
          else:
              pass

          # Flip images randomly
          flip_coin = np.random.randint(0,1)
          if flip_coin == 1:
            train[i], angle[i] = flip_image(train[i], angle[i])

        yield train, angle

def valid_generator(X_val, y_val):
    """
    Function that creates batches of data to save computational resources while training.
    It takes validation data array, which contains a list of images, and corresponding steering angles array.
    """
    train = np.zeros((FLAGS.batch_size, 66, 200, 3), dtype = np.float32)
    angle = np.zeros((FLAGS.batch_size,), dtype = np.float32)
    while True:
      data, ang = shuffle(X_val, y_val)
      for i in range(FLAGS.batch_size):
        rand = int(np.random.choice(len(data), 1))
        if data[rand] != None:
            train[i] = adjust_image(mpimg.imread(FLAGS.images_folder + data[rand].strip()))
            angle[i] = ang[rand]
        else:
            pass
      
      yield train, angle

def main(_):
    # Load training data
    X_train, y_train, X_val, y_val = load_training_data(FLAGS.training_file)

    # Define model
    model = Sequential()
    model.add(Lambda(lambda x: x/255-0.5, input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    print(model.summary())

    #Train model
    model.fit_generator(data_generator(X_train, y_train), samples_per_epoch=len(X_train), nb_epoch=FLAGS.epochs, validation_data=valid_generator(X_val, y_val), nb_val_samples=len(X_val))

    # Save model to HDF5
    model.save("model.h5")

# Parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
