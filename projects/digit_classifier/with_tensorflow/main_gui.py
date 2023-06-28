# Import libraries
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from img_preprocess import img_to_array_28
from keras.models import load_model 

class DigitClassifier:
    def __init__(self):
        # Pre-trained model
        self.model = load_model('./digit_classifier.h5') 
        
        # GUI setup
        self.window = Tk()
        self.window.title("Digit Classifier")
        
        # Label 
        self.digit_label = Label(self.window, text="Digit Here", font=("Arial",48))  
        self.digit_label.grid(column=0, row=0)  
        
        # Upload button
        self.upload_btn = Button(self.window, text="Upload Image", 
                            command=self.upload, font=("Arial",14))
        self.upload_btn.grid(column=0, row=1)

        # Exit button   
        self.exit_btn = Button(self.window, text="Exit", 
                            command=self.window.destroy, font=("Arial", 14))
        self.exit_btn.grid(column=1, row=1, sticky='e')
        
        # Image display area   
        self.image = Label(self.window)
        self.image.grid(columnspan=2, row=2)   
        
        # Predicted digit label
        #self.pred_label = Label(self.window, text="Predicted Digit:", font=("Arial",14))   
        #self.pred_label.grid(column=0, row=3, sticky='w')
        
    def upload(self):
        # Get image path 
        img_path = askopenfilename()
        
        # Preprocess image   
        #img_array = img_to_array_28(img_path)  
        img_array = img_to_array_28(img_path)  
        
        # Reshape to 4D tensor
        #img_array = img_array.reshape(1,28,28,1) # 4D for model input
        img_array = img_array.reshape(1,28,28,1) # 4D for model input
        
        # Make prediction   
        pred = self.model.predict(img_array)
        
        # Get digit
        digit = np.argmax(pred, axis=1)[0]

        # Update labels
        #self.pred_label['text'] = "Predicted Digit:" 
        self.digit_label['text'] = str(digit)  

class App:
    def __init__(self):
        self.digit_classifier = DigitClassifier()
        
if __name__ == '__main__':    
    app = App()
    app.digit_classifier.window.mainloop()