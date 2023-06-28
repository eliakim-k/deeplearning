from img_preprocess import img_to_array
import sys
import numpy as np
from keras.models import load_model

if __name__ == '__main__':

    # To load later
    saved_model = load_model('digit_classifier.h5')

    ## Use the model
    # img_path = sys.argv[1]
    img_path = r"C:\Users\eliakim\deeplearning\projects\img\raw_img\four.jpg"

    # get a image and cast it to a tensor
    img_array = img_to_array(img_path)
    
    #print(img_array.min(), img_array.max())

    # Reshape to 4D tensor for model input  
    img_array = img_array.reshape(1,28,28,1)

    # print(img_array.shape)

    # Make predicton with model
    pred = saved_model.predict(img_array)

    # Get the index of the highest probability  
    digit = np.argmax(pred, axis=1)[0]   
    
    print(f"Predicted digit: {digit}")