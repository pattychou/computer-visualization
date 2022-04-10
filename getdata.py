
import numpy as np
import matplotlib.pyplot as plt
from neural_net import TwoLayerNet
import mnist


def get_mnist_data(num_training=49000, num_validation=1000, num_test=1000):
  """
  Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
  it for the two-layer neural net classifier. These are the same steps as
  we used for the SVM, but condensed to a single function.
  """

  X_train, y_train, X_test, y_test = mnist.load()

  # Subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask].astype('float64')
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask].astype('float64')
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask].astype('float64')
  y_test = y_test[mask]

  # Normalize the data: subtract the mean image

  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image

  # Reshape data to rows
  X_train = X_train.reshape(num_training, -1)
  X_val = X_val.reshape(num_validation, -1)
  X_test = X_test.reshape(num_test, -1)

  return X_train, y_train, X_val, y_val, X_test, y_test



# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_mnist_data()

'''
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
'''
