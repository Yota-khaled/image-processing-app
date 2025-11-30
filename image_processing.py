"""
Image processing functions
Contains all image transformation and processing operations
"""

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, laplace, sobel


def to_grayscale(image):
    """
    Convert image to grayscale
    
    Args:
        image: PIL Image object
        
    Returns:
        Grayscale PIL Image object
    """
    if image:
        return image.convert("L").convert("RGB")
    return None


def to_binary(image, threshold=128):
    """
    Convert image to binary (black and white)
    
    Args:
        image: PIL Image object
        threshold: Threshold value (0-255) for binary conversion
        
    Returns:
        Binary PIL Image object
    """
    if image:
        gray = image.convert("L")
        binary = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
        return binary.convert("RGB")
    return None


def translate(image, tx, ty):
    """
    Translate (shift) image
    
    Args:
        image: PIL Image object
        tx: Translation in X direction (pixels)
        ty: Translation in Y direction (pixels)
        
    Returns:
        Translated PIL Image object
    """
    if image:
        # Create transformation matrix
        matrix = (1, 0, tx, 0, 1, ty)
        return image.transform(image.size, Image.AFFINE, matrix, fillcolor=(0, 0, 0))
    return None


def scale(image, scale_x, scale_y):
    """
    Scale image
    
    Args:
        image: PIL Image object
        scale_x: Scale factor in X direction
        scale_y: Scale factor in Y direction
        
    Returns:
        Scaled PIL Image object
    """
    if image:
        new_size = (int(image.size[0] * scale_x), int(image.size[1] * scale_y))
        return image.resize(new_size, Image.LANCZOS)
    return None


def rotate(image, angle):
    """
    Rotate image
    
    Args:
        image: PIL Image object
        angle: Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        Rotated PIL Image object
    """
    if image:
        return image.rotate(angle, expand=True, fillcolor=(0, 0, 0))
    return None


def shear_x(image, shear_factor):
    """
    Apply horizontal shear transformation
    
    Args:
        image: PIL Image object
        shear_factor: Shear factor (typically -1.0 to 1.0)
        
    Returns:
        Sheared PIL Image object
    """
    if image:
        matrix = (1, shear_factor, 0, 0, 1, 0)
        return image.transform(image.size, Image.AFFINE, matrix, fillcolor=(0, 0, 0))
    return None


def shear_y(image, shear_factor):
    """
    Apply vertical shear transformation
    
    Args:
        image: PIL Image object
        shear_factor: Shear factor (typically -1.0 to 1.0)
        
    Returns:
        Sheared PIL Image object
    """
    if image:
        matrix = (1, 0, 0, shear_factor, 1, 0)
        return image.transform(image.size, Image.AFFINE, matrix, fillcolor=(0, 0, 0))
    return None


def nearest_neighbor_interpolation(image, new_size):
    """
    Resize image using nearest neighbor interpolation
    
    Args:
        image: PIL Image object
        new_size: Tuple of (width, height) for new size
        
    Returns:
        Resized PIL Image object
    """
    if image:
        return image.resize(new_size, Image.NEAREST)
    return None


def bilinear_interpolation(image, new_size):
    """
    Resize image using bilinear interpolation
    
    Args:
        image: PIL Image object
        new_size: Tuple of (width, height) for new size
        
    Returns:
        Resized PIL Image object
    """
    if image:
        return image.resize(new_size, Image.BILINEAR)
    return None


def bicubic_interpolation(image, new_size):
    """
    Resize image using bicubic interpolation
    
    Args:
        image: PIL Image object
        new_size: Tuple of (width, height) for new size
        
    Returns:
        Resized PIL Image object
    """
    if image:
        return image.resize(new_size, Image.BICUBIC)
    return None


def show_histogram(image):
    """
    Calculate and return histogram data
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with histogram data for R, G, B channels
    """
    if image:
        img_array = np.array(image)
        histograms = {}
        for i, color in enumerate(['Red', 'Green', 'Blue']):
            histograms[color] = np.histogram(img_array[:, :, i], bins=256, range=(0, 256))[0]
        return histograms
    return None


def histogram_equalization(image):
    """
    Apply histogram equalization to enhance image contrast
    
    Args:
        image: PIL Image object
        
    Returns:
        Histogram equalized PIL Image object
    """
    if image:
        img_array = np.array(image)
        equalized = np.zeros_like(img_array)
        
        for i in range(3):  # For each RGB channel
            channel = img_array[:, :, i]
            # Calculate histogram
            hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
            # Calculate cumulative distribution
            cdf = hist.cumsum()
            cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
            # Apply equalization
            equalized[:, :, i] = cdf_normalized[channel]
        
        return Image.fromarray(equalized.astype('uint8'))
    return None


def gaussian_filter_func(image, sigma=1.0):
    """
    Apply Gaussian blur filter (low-pass)
    
    Args:
        image: PIL Image object
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Filtered PIL Image object
    """
    if image:
        img_array = np.array(image)
        filtered = np.zeros_like(img_array)
        
        for i in range(3):  # For each RGB channel
            filtered[:, :, i] = gaussian_filter(img_array[:, :, i], sigma=sigma)
        
        return Image.fromarray(filtered.astype('uint8'))
    return None


def median_filter_func(image, size=3):
    """
    Apply median filter (low-pass, noise reduction)
    
    Args:
        image: PIL Image object
        size: Size of the filter kernel (must be odd)
        
    Returns:
        Filtered PIL Image object
    """
    if image:
        img_array = np.array(image)
        filtered = np.zeros_like(img_array)
        
        for i in range(3):  # For each RGB channel
            filtered[:, :, i] = median_filter(img_array[:, :, i], size=size)
        
        return Image.fromarray(filtered.astype('uint8'))
    return None


def laplacian_filter(image):
    """
    Apply Laplacian filter (high-pass, edge detection)
    
    Args:
        image: PIL Image object
        
    Returns:
        Filtered PIL Image object
    """
    if image:
        gray = image.convert("L")
        img_array = np.array(gray)
        laplacian_result = laplace(img_array)
        # Normalize to 0-255
        laplacian_result = np.abs(laplacian_result)
        laplacian_result = (laplacian_result / laplacian_result.max() * 255).astype('uint8')
        return Image.fromarray(laplacian_result).convert("RGB")
    return None


def sobel_filter(image):
    """
    Apply Sobel filter (high-pass, edge detection)
    
    Args:
        image: PIL Image object
        
    Returns:
        Filtered PIL Image object
    """
    if image:
        gray = image.convert("L")
        img_array = np.array(gray)
        
        # Apply Sobel filters
        sobel_x = sobel(img_array, axis=1)
        sobel_y = sobel(img_array, axis=0)
        
        # Combine
        sobel_result = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_result = (sobel_result / sobel_result.max() * 255).astype('uint8')
        
        return Image.fromarray(sobel_result).convert("RGB")
    return None


def gradient_filter(image):
    """
    Apply gradient filter (high-pass, edge detection)
    
    Args:
        image: PIL Image object
        
    Returns:
        Filtered PIL Image object
    """
    if image:
        gray = image.convert("L")
        img_array = np.array(gray)
        
        # Calculate gradients
        grad_x = np.gradient(img_array, axis=1)
        grad_y = np.gradient(img_array, axis=0)
        
        # Combine
        gradient_result = np.sqrt(grad_x**2 + grad_y**2)
        gradient_result = (gradient_result / gradient_result.max() * 255).astype('uint8')
        
        return Image.fromarray(gradient_result).convert("RGB")
    return None

