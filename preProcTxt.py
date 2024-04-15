import cv2
import numpy as np
from PIL import Image
import tempfile

def normalize_image(image):
    if image is None:
        return None
    
    # Normalize image intensity to a specific range (e.g., [0, 255])
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    normalized_image = cv2.normalize(image, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_image

def correct_skew(image):
    if image is None:
        return None
    
    # Find coordinates of non-zero pixels
    co_ords = np.column_stack(np.where(image > 0))
    
    # Calculate the minimum area rectangle enclosing the non-zero pixels and its orientation angle
    angle = cv2.minAreaRect(co_ords)[-1]
    
    # Adjust angle to be within -45 to 45 degrees range
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Calculate rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation to deskew the image
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def scale_image(image, width=None, height=None):
    if image is None:
        return None
    
    # Open the image file
    im = Image.open(image)
    
    # Get the dimensions of the image
    length_x, width_y = im.size
    
    # Calculate the resizing factor based on a maximum length of 1024 pixels
    factor = min(1, float(1024.0 / length_x))
    
    # Resize the image
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    
    # Create a temporary file to save the resized image with specified DPI
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_filename = temp_file.name
    
    # Save the resized image with specified DPI (300 DPI)
    im_resized.save(temp_filename, dpi=(300, 300))
    
    return temp_filename

def remove_noise(image):
    if image is None:
        return None
    
    # Apply Gaussian blur to remove noise
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

def thinning_and_skeletonization(image):
    if image is None:
        return None
    
    # Define the erosion kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # Perform erosion operation
    erosion = cv2.erode(image, kernel, iterations=1)
    
    return erosion

def convert_to_grayscale(image):
    if image is None:
        return None
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def binarize_image(image):
    if image is None:
        return None
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to binarize the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return binary

# Example usage:
image_path = 'bl.jpg'

# Load the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load or read the image file. Please check the file path and try again.")
else:
    # Normalize image intensity
    normalized_image = normalize_image(image)

    # Correct skew
    skew_corrected_image = correct_skew(image)

    # Scale image
    scaled_image = scale_image(image, width=800)

    # Remove noise
    noise_removed_image = remove_noise(image)

    # Thinning and skeletonization
    thinned_image = thinning_and_skeletonization(image)

    # Convert to grayscale
    grayscale_image = convert_to_grayscale(image)

    # Binarize image
    binarized_image = binarize_image(image)

    # Display or save the images (optional)
    # Ensure to handle None values for images
    
    cv2.imshow('Normalized Image', normalized_image)
    cv2.imshow('Skew Corrected Image', skew_corrected_image)
    cv2.imshow('Scaled Image', scaled_image)
    cv2.imshow('Noise Removed Image', noise_removed_image)
    cv2.imshow('Thinned Image', thinned_image)
    cv2.imshow('Grayscale Image', grayscale_image)
    cv2.imshow('Binarized Image', binarized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
