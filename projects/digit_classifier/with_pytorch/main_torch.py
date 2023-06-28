import torch
from tkinter import *
from tkinter.filedialog import askopenfilename
from img_preprocess_torch import image_to_tensor
from model_architecture_torch import NeuralNet, get_prediction

class DigitClassifier:
    def __init__(self):
        # create the model
        self.input_size = 784 # 28x28
        self.hidden_size = 500 
        self.num_classes = 10
        self.PATH = './with_pytorch/digit_classifier.pth'
        self.model = NeuralNet(self.input_size, self.hidden_size, self.num_classes)
        # Load the model
        self.model = self.model.load_state_dict(torch.load(self.PATH)) 
        
        # GUI setup
        self.window = Tk()
        self.window.title("Digit Classifier")
        
        # Label 
        self.digit_label = Label(self.window, text="Digit Here", font=("Arial", 48))  
        self.digit_label.grid(column=0, row=0)  
        
        # Upload button
        self.upload_btn = Button(self.window, text="Upload Image", 
                            command=self.upload, font=("Arial", 14))
        self.upload_btn.grid(column=0, row=1)

        # Exit button   
        self.exit_btn = Button(self.window, text="Exit", 
                            command=self.window.destroy, font=("Arial", 14))
        self.exit_btn.grid(column=1, row=1, sticky='e')
        
        # Image display area   
        self.image = Label(self.window)
        self.image.grid(columnspan=2, row=2)
        
    def upload(self):
        # Get image path 
        self.img_path = askopenfilename()
        
        # Preprocess image   
        self.img_tensor = image_to_tensor(self.img_path) 
        
        # Reshape to 1D tensor
        self.img_tensor = self.img_tensor.reshape(-1, 28*28) # 1D for model input
            
        # Make prediction   
        self.pred = get_prediction(self.img_tensor)

        # Update labels
        self.digit_label['text'] = str(self.pred.item())

class App:
    def __init__(self):
        self.digit_classifier = DigitClassifier()
        
if __name__ == '__main__':    
    app = App()
    app.digit_classifier.window.mainloop()