"""
Image processing functions
Contains all image transformation and processing operations
"""

from typing import Optional
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, laplace, sobel

# image_processing.py
from PIL import Image
import numpy as np



# -------------------------
# Helper: Otsu
# -------------------------
def otsu_threshold(arr: np.ndarray) -> int:
    """
    Compute Otsu threshold for a 2D uint8 array. Returns int in [0,255].
    """
    flat = arr.ravel()
    counts, _ = np.histogram(flat, bins=256, range=(0,256))
    total = flat.size
    sum_total = (np.arange(256) * counts).sum()
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0
    for i in range(256):
        w_b += counts[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * counts[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    return int(threshold)

def _between_var_from_counts(counts: np.ndarray, t: int) -> float:
    """Compute between-class variance given histogram counts and threshold t (counts length 256)."""
    counts = counts.astype(np.float64)
    w_b = counts[: t+1].sum()
    w_f = counts[t+1 :].sum()
    if w_b == 0 or w_f == 0:
        return 0.0
    mu_b = (np.arange(0, t+1) * counts[: t+1]).sum() / w_b
    mu_f = (np.arange(t+1, 256) * counts[t+1 :]).sum() / w_f
    return w_b * w_f * (mu_b - mu_f) ** 2

# -------------------------
# Public: to_grayscale (backwards-compatible name)
# -------------------------
def to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert image to grayscale using luminance formula (returns RGB image for compatibility with UI).
    Y = 0.2126 R + 0.7152 G + 0.0722 B
    """
    if image.mode == 'L':
        # convert to RGB to keep UI consistent (3 channels)
        return image.convert("RGB")

    arr = np.array(image.convert('RGB'), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    # Return RGB image so UI that expects 3 channels doesn't break
    return Image.fromarray(gray, mode='L').convert("RGB")

# -------------------------
# Enhanced: to_binary_info (returns details)
# -------------------------
def to_binary_info(image: Image.Image, method: str = 'auto'):
    """
    Produce binary image and info.
    Args:
      method: 'auto' | 'avg' | 'otsu'
    Returns:
      (binary_image (PIL.Image, mode 'RGB'), threshold (float), chosen_method (str), reason (optional str))
    """
    gray = image.convert('L')
    arr = np.array(gray, dtype=np.uint8)

    # average threshold
    avg_t = float(arr.mean())
    # otsu threshold
    otsu_t = float(otsu_threshold(arr))

    # binary arrays
    avg_bin_arr = (arr >= avg_t).astype(np.uint8) * 255
    otsu_bin_arr = (arr >= otsu_t).astype(np.uint8) * 255

    chosen = method
    reason = ""
    if method == 'auto':
        counts, _ = np.histogram(arr.ravel(), bins=256, range=(0,256))
        bv_avg = _between_var_from_counts(counts, int(round(avg_t)))
        bv_otsu = _between_var_from_counts(counts, int(round(otsu_t)))
        # heuristic: prefer otsu if its between-class variance > 1.1 * avg's
        if bv_otsu > 1.1 * bv_avg:
            chosen = 'otsu'
        else:
            chosen = 'avg'
        reason = f"bv_otsu={bv_otsu:.1f}, bv_avg={bv_avg:.1f}"
    else:
        reason = f"forced method={method}"

    if chosen == 'otsu':
        img = Image.fromarray(otsu_bin_arr, mode='L').convert("RGB")
        return img, otsu_t, 'otsu', reason
    else:
        img = Image.fromarray(avg_bin_arr, mode='L').convert("RGB")
        return img, avg_t, 'avg', reason

# -------------------------
# Backwards-compatible simple function
# -------------------------
def to_binary(image: Image.Image, threshold: Optional[int] = None) -> Image.Image:
    """
    Backwards-compatible to_binary used by app.py.
    - If `threshold` is an int (0-255): use it (manual mode).
    - If `threshold` is None: perform auto selection between Average and Otsu.

    Returns a PIL.Image (RGB) — safe to display in Streamlit.
    """
    if image is None:
        raise ValueError("to_binary: image is None")

    # ensure grayscale array for computations
    gray = image.convert('L')
    arr = np.array(gray, dtype=np.uint8)

    # Manual threshold path
    if threshold is not None:
        t = int(max(0, min(255, int(round(threshold)))))
        bin_arr = (arr >= t).astype(np.uint8) * 255
        return Image.fromarray(bin_arr, mode='L').convert("RGB")

    # Auto-selection path: compute both and choose by between-class variance
    # average
    avg_t = float(arr.mean())
    avg_bin_arr = (arr >= avg_t).astype(np.uint8) * 255
    # otsu
    otsu_t = otsu_threshold(arr)
    otsu_bin_arr = (arr >= otsu_t).astype(np.uint8) * 255

    counts, _ = np.histogram(arr.ravel(), bins=256, range=(0,256))
    bv_avg = _between_var_from_counts(counts, int(round(avg_t)))
    bv_otsu = _between_var_from_counts(counts, int(round(otsu_t)))

    # Heuristic: choose Otsu if it gives a noticeably higher between-class variance
    chosen = 'otsu' if (bv_otsu > 1.1 * bv_avg) else 'avg'

    if chosen == 'otsu':
        return Image.fromarray(otsu_bin_arr, mode='L').convert("RGB")
    else:
        return Image.fromarray(avg_bin_arr, mode='L').convert("RGB")

# def to_grayscale(image):
#     """
#     Convert image to grayscale
    
#     Args:
#         image: PIL Image object
        
#     Returns:
#         Grayscale PIL Image object
#     """
#     if image:
#         return image.convert("L").convert("RGB")
#     return None


# def to_binary(image, threshold=128):
#     """
#     Convert image to binary (black and white)
    
#     Args:
#         image: PIL Image object
#         threshold: Threshold value (0-255) for binary conversion
        
#     Returns:
#         Binary PIL Image object
#     """
#     if image:
#         gray = image.convert("L")
#         binary = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
#         return binary.convert("RGB")
#     return None


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


# -------------------------
# Affine transform operations
# -------------------------
def apply_affine(image: Image.Image, operation: str, **kwargs) -> Image.Image:
    if image is None:
        raise ValueError("apply_affine: image is None")

    op = operation.lower()
    if op == 'translate':
        tx = int(kwargs.get('tx', kwargs.get('x', 0)))
        ty = int(kwargs.get('ty', kwargs.get('y', 0)))
        return translate(image, tx, ty)
    elif op == 'scale':
        sx = float(kwargs.get('sx', kwargs.get('scale_x', kwargs.get('scale', 1.0))))
        sy = float(kwargs.get('sy', kwargs.get('scale_y', kwargs.get('scale', 1.0))))
        return scale(image, sx, sy)
    elif op == 'rotate':
        angle = float(kwargs.get('angle', kwargs.get('deg', 0)))
        return rotate(image, angle)
    elif op in ('shear_x', 'shearx', 'shear-x'):
        f = float(kwargs.get('factor', kwargs.get('shear_x_factor', kwargs.get('shear', 0.0))))
        return shear_x(image, f)
    elif op in ('shear_y', 'sheary', 'shear-y'):
        f = float(kwargs.get('factor', kwargs.get('shear_y_factor', kwargs.get('shear', 0.0))))
        return shear_y(image, f)
    else:
        raise ValueError(f"Unknown affine operation: {operation}")



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

