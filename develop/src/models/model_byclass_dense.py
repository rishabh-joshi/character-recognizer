print("Loading the libraries.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

print("Reading data.")

train = pd.read_csv("data/emnist-byclass-train.csv", header = None)
test = pd.read_csv("data/emnist-byclass-test.csv", header = None)

X_train = train.iloc[:, 1:]
X_test = test.iloc[:, 1:]

Y_train = train[0]
Y_test = test[0]

X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values

print("Reshaping data.")

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255




number_of_classes = 62

Y_train = to_categorical(Y_train, number_of_classes)
Y_test = to_categorical(Y_test, number_of_classes)


print("Building the ConvNet.")

# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(number_of_classes))

model.add(Activation('softmax'))





model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])




print("Training the ConvNet.")

batch_size=256
epochs=10

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))

print("Testing the ConvNet.")

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print("Saving the ConvNet.")

model.save('deep_cnn_byclass2.h5')

print("All tasks completed.")