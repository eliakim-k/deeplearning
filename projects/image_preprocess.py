## Import OpenCV

import cv2
import os
import sys
import matplotlib.pyplot as plt

def img_to_array(img_path):
    """This function does the following:
    - Reads the input image using OpenCV's imread()
    - Gets the height and width of the image
    - Calculates the ratio needed to resize the image to 28 pixels wide
    - Resizes the image using resize(), keeping the aspect ratio
    - Converts the resized image to grayscale using cvtColor()
    - Saves the resized grayscale image
    """
    # Read the image 
    img = cv2.imread(f'{img_path}')

    # check image shape
    img.shape

    # Get the height and width of the image
    height, width = img.shape[:2]

    # Calculate the ratio of the image 
    # after resizing to 28x28
    ratio = 28.0/width 

    # Resize the image keeping the ratio
    resized_img = cv2.resize(img, (28, int(height*ratio)))

    # Convert to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Save the resized grayscale image 
    #cv2.imwrite('./img/preprocessed_img/onetris_resized_grayscale.jpg', gray_img)
 
    # Convert to numpy array
    #image_array = np.array(gray_img)
    return gray_img

if __name__ == '__main__':

    img_path = str(sys.argv[1])
    try:
        img_array = img_to_array(img_path)

        plt.imshow(img_array)
        plt.show()
    except:
        sys.exit("An unexpected error occurred.")