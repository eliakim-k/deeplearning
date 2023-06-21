# Import libraries
from tkinter import * 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from image_preprocess import img_to_array
from keras.models import load_model 

# Create window
window = Tk()

saved_model = load_model(r'C:\Users\eliakim\deeplearning\projects\digit_classifier.h5')
  
# Upload image function
def upload():
    # Open file dialog 
    file = askopenfilename()  
  
    # Get image path 
    img_path = file
      
    # Preprocess image   
    img_array = img_to_array(img_path)
      
    # Make prediction  
    pred = saved_model.predict(img_array)
     
    # Get digit
    digit = np.argmax(pred, axis=1)[0]
    
    # Update label
    digit_label["text"] = digit

# Label to show digit       
digit_label = Label(window, text="Digit Here")  
digit_label.grid(column=0, row=0) 

# Upload button 
upload_btn = Button(window,text="Upload Image", command=upload)
upload_btn.grid(column=0, row=1)