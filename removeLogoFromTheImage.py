import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

def apply_filters(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a PIL image from array
    pil_im = Image.fromarray(image_rgb)

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(pil_im)
    pil_im = enhancer.enhance(1.2)  # The factor 1.0 means no change, greater than 1.0 increases brightness

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(pil_im)
    pil_im = enhancer.enhance(3.0)  # The factor 1.0 means no change, greater than 1.0 increases contrast

    # Adjust sharpness
    enhancer = ImageEnhance.Sharpness(pil_im)
    pil_im = enhancer.enhance(2.0)  # The factor 1.0 means no change, greater than 1.0 increases sharpness

    # Optionally, convert to grayscale
    # pil_im = pil_im.convert('L')

    # Apply a filter if desired, like an edge-enhancing filter
    pil_im = pil_im.filter(ImageFilter.EDGE_ENHANCE)

    # Save or display the resulting image
    pil_im.save(output_path)

    # Show the resulting image using plt.show()
    plt.imshow(np.array(pil_im))
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

# Example usage:
image_path = 'bl.jpg'  # Replace with your image path
output_path = 'filter_bl.jpg'  # Replace with the desired output path


apply_filters(image_path, output_path)
