from torchvision import transforms
from PIL import Image, ImageOps
import io

# normalization parameters
mean, std = 0.1307, 0.3081

# image -> tensor
def image_to_tensor(PATH):
    """
    Transforms an image located at PATH into a tensor for use in a PyTorch model.

    Args:
        PATH (str): The path to the image file.

    Returns:
        numpy.ndarray: A 4D NumPy array with shape (1, 1, 28, 28),
            representing the transformed image as a tensor.

    """

    # Read the image bytes from file
    with open(PATH, 'rb') as f:
        image_bytes = f.read()

    # Define the transformations to apply to the image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Convert the bytes to a PIL Image object
    image = Image.open(io.BytesIO(image_bytes))

    # Invert the colors of the image
    inverted_image = ImageOps.invert(image)

    # Apply the transformations to the inverted image and return as a 4D NumPy array with batch dimension
    return transform(inverted_image).unsqueeze(0)
    #return transform(image).unsqueeze(0)