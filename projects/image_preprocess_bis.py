import cv2
import numpy as np
from imutils import contours
import sys
import matplotlib.pyplot as plt

def img_to_array(img_path):
    """Perform image preprocessing and return a grayscale NumPy array"""
    # Read the image 
    img = cv2.imread(img_path)
    
    # Resize to 48x48 for better resolution 
    resized_img = cv2.resize(img,(48,48))

    # Convert to grayscale 
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
 
    # Apply histogram equalization for contrast enhancement
    equ = cv2.equalizeHist(gray_img)
    
    # Apply Gaussian blur to remove noise 
    blur = cv2.GaussianBlur(equ,(5,5),0)
    
    # Apply adaptive thresholding  
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        
    # Normalize pixel values from 0-1  
    thresh = thresh.astype("float") / 255.0
        
    # Convert to grayscale NumPy array 
    img_array = np.array(thresh)
    
    # Return the preprocessed image array  
    return img_array

if __name__ == "__main__":

    img_path = str(sys.argv[1])
    img_array = img_to_array(img_path)

    plt.imshow(img_array)
    plt.show()