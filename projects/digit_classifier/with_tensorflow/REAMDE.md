# Digit Classifier

This program is a simple digit classifier that can recognize handwritten digits in images. It consists of the following:

## How it works

### 1. model_trainer.py

This module trains a convolutional neural network model using the MNIST dataset. The model architecture consists of:

- Convolutional layers  
- Max pooling layers
- Flatten layer
- Dense layers   

After training the model, it is saved as `digit_classifier.h5`

### 2. img_preprocess.py

This module contains a function `img_to_array_28()` that preprocesses an input image to a 28x28 grayscale array, which is the required input shape for the trained model.  

### 3. main_gui.py

This module contains the GUI code. It loads the saved model `digit_classifier.h5` and uses the `img_to_array_28()` function from `img_preprocess.py` to preprocess images uploaded by the user.

It then makes a prediction on that preprocessed image and displays the predicted digit on the GUI.

## How to use

1. Run `model_trainer.py` to train the model and save it. For this, you need to remove the hashtag sign (#) before the lines `model.fit(x_train, y_train, epochs=5)`and `model.save('digit_classifier.h5')`
2. Run `main_gui.py` to open the GUI. 
3. Click the "Upload Image" button and select an image of a handwritten digit.
4. The predicted digit will be displayed on the GUI.

Caution: All the files have got to be in the same folder.