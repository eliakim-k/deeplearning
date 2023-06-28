## Import modules

import numpy as np # For arrays
from tensorflow import keras
import matplotlib.pyplot as plt # For viewing images
from keras.datasets import mnist # For MNIST data
from keras.models import Sequential # For building the model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten  

## Obtain and preprocess data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data to 4D tensors for ConvNet input
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

"""import matplotlib.pyplot as plt
plt.imshow(x_train[1,: , :, 0])
plt.show()"""

# num classes
num_classes = 10

# Convert class vectors to binary class matrices 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## Build the model

# Build the model  
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2)))  
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## Train the model

# Train model  
#model.fit(x_train, y_train, epochs=5)

# After training
#model.save('digit_classifier.h5')

# To load the model (see main_console.py)
# saved_model = load_model('digit_classifier')

## Evaluate the model

# Evaluate on test data
#model.evaluate(x_test,  y_test)