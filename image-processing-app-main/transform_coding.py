"""
Transform Coding + Predictive + Wavelet
Implementation of various transform coding techniques
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy.fft import dct, idct
from scipy import signal



try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    pywt = None


def visualize_predictive_error(errors):
    """
    Visualize predictive coding errors.
    Maps error range [-255, 255] to [0, 255] for display.
    Zero error becomes mid-gray (128).
    """
    # Scale errors to 0-255 range centered at 128
    # errors are typically small, so we might want to amplify them for visibility
    # For now, let's just shift them: 0 -> 128.
    vis = (errors / 2) + 128
    return np.clip(vis, 0, 255).astype(np.uint8)

def visualize_wavelet_coeffs(coeffs):
    """
    Visualize wavelet coefficients as a mosaic image.
    """
    if not isinstance(coeffs, list) or len(coeffs) == 0:
        return None
    
    # coeffs[0] is cA (Approximation)
    # coeffs[i] is (cH, cV, cD) for level i
    
    def normalize(arr):
        # Normalize to 0-255 for visualization
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        return (255 * (arr - arr_min) / (arr_max - arr_min)).astype(np.uint8)

    cA = coeffs[0]
    
    # Start with the approximation
    full_image = normalize(cA)
    
    # Iteratively add details
    # coeffs list is [cA, (cH1, cV1, cD1), (cH2, cV2, cD2), ...]
    # where index 1 is the coarsest level (closest to cA) and last index is finest level
    

    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        
        # Normalize details
        cH_vis = normalize(cH)
        cV_vis = normalize(cV)
        cD_vis = normalize(cD)
        
        # Get target shape from details (they should be consistent within the tuple)
        h, w = cH_vis.shape
        
        # Resize current full_image (approximation) to match details if needed
        # This handles cases where different wavelet filters produce slightly different sizes
        # or when odd dimensions cause rounding differences.
        if full_image.shape != (h, w):
            # Using simple scaling (could use cv2.resize or just zoom)
            # Since we don't want to depend on cv2, and scipy.zoom is heavy...
            # We can use a simple numpy generic zoom or just PIL if available in this scope?
            # PIL is not imported in this file. Let's use scipy.ndimage if available or simple slicing/repeating?
            # Or just use the already imported signal or similar...
            # Actually, `transform_coding` imports `numpy`.
            # Let's check imports.
            
            # Simple nearest neighbor resize using integer indexing (fast, dependency-free)
            fh, fw = full_image.shape
            # Create grid
            x = np.linspace(0, fw - 1, w)
            y = np.linspace(0, fh - 1, h)
            # Round to nearest int
            xi = np.round(x).astype(int)
            yi = np.round(y).astype(int)
            # Index
            # usage of meshgrid for 2D indexing
            # full_image[yi[:, None], xi]
            full_image = full_image[yi[:, None], xi]
            
        
        top = np.hstack((full_image, cH_vis))
        bot = np.hstack((cV_vis, cD_vis))
        full_image = np.vstack((top, bot))
        
        # Update cA... logic is implicit as full_image becomes the approx for next level (if we were going up)
        # But wait, we are traversing coeffs list.
        # This logic builds a SINGLE mosaic. 
        # The result 'top' + 'bot' IS the representation of the image at THIS level (containing all previous coarser levels inside 'full_image').
    
    return full_image.astype(np.uint8)



# -------------------------
# Symbol-based Coding
# -------------------------
def symbol_based_encode(data):
    """
    Symbol-based coding: Group consecutive identical symbols
    """
    if len(data) == 0:
        return []
    
    encoded = []
    current_symbol = data[0]
    count = 1
    
    for i in range(1, len(data)):
        if data[i] == current_symbol:
            count += 1
        else:
            encoded.append((current_symbol, count))
            current_symbol = data[i]
            count = 1
    
    encoded.append((current_symbol, count))
    return encoded


def symbol_based_decode(encoded):
    """Decode symbol-based encoded data"""
    decoded = []
    for symbol, count in encoded:
        decoded.extend([symbol] * count)
    return decoded


# -------------------------
# Bit-plane Coding
# -------------------------
def bit_plane_encode(image_array):
    """
    Bit-plane coding: Extract and encode each bit plane separately
    """
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    
    height, width = image_array.shape
    bit_planes = []
    
    for bit in range(8):
        bit_plane = (image_array >> bit) & 1
        bit_planes.append(bit_plane)
    
    return bit_planes, (height, width)


def bit_plane_decode(bit_planes, shape):
    """Decode bit-plane encoded data"""
    height, width = shape
    reconstructed = np.zeros((height, width), dtype=np.uint8)
    
    for bit, bit_plane in enumerate(bit_planes):
        reconstructed |= (bit_plane.astype(np.uint8) << bit)
    
    return reconstructed


# -------------------------
# Block Transform Coding (DCT)
# -------------------------
def dct_block_encode(image_array, block_size=8):
    """
    Block Transform coding using DCT
    """
    if len(image_array.shape) == 2:
        height, width = image_array.shape
        channels = 1
        image_array = image_array[:, :, np.newaxis]
    else:
        height, width, channels = image_array.shape
    
    # Pad image to be divisible by block_size
    pad_h = (block_size - height % block_size) % block_size
    pad_w = (block_size - width % block_size) % block_size
    
    if pad_h > 0 or pad_w > 0:
        image_array = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    
    new_height, new_width = image_array.shape[0], image_array.shape[1]
    dct_blocks = []
    
    for c in range(channels):
        channel_blocks = []
        for i in range(0, new_height, block_size):
            for j in range(0, new_width, block_size):
                block = image_array[i:i+block_size, j:j+block_size, c]
                dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                channel_blocks.append(dct_block)
        dct_blocks.append(channel_blocks)
    
    return dct_blocks, (height, width, channels), (new_height, new_width), block_size


def dct_block_decode(dct_blocks, original_shape, padded_shape, block_size):
    """
    Decode DCT block transform
    """
    height, width, channels = original_shape
    new_height, new_width = padded_shape
    
    if channels == 1:
        reconstructed = np.zeros((new_height, new_width), dtype=np.float32)
        channel_blocks = dct_blocks[0]
    else:
        reconstructed = np.zeros((new_height, new_width, channels), dtype=np.float32)
    
    for c in range(channels):
        channel_blocks = dct_blocks[c]
        block_idx = 0
        for i in range(0, new_height, block_size):
            for j in range(0, new_width, block_size):
                dct_block = channel_blocks[block_idx]
                idct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
                
                if channels == 1:
                    reconstructed[i:i+block_size, j:j+block_size] = idct_block
                else:
                    reconstructed[i:i+block_size, j:j+block_size, c] = idct_block
                block_idx += 1
    
    # Crop to original size
    reconstructed = reconstructed[:height, :width]
    if channels == 1:
        reconstructed = reconstructed.squeeze()
    
    # Clip and convert to uint8
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed


# -------------------------
# Predictive Coding
# -------------------------
def predictive_encode(image_array, predictor='previous'):
    """
    Predictive coding: Encode prediction errors
    """
    if len(image_array.shape) == 3:
        height, width, channels = image_array.shape
        result = np.zeros_like(image_array)
        for c in range(channels):
            result[:, :, c] = _predictive_encode_channel(image_array[:, :, c], predictor)
    else:
        result = _predictive_encode_channel(image_array, predictor)
    
    return result


def _predictive_encode_channel(channel, predictor):
    """Encode a single channel using predictive coding"""
    height, width = channel.shape
    errors = np.zeros_like(channel, dtype=np.int16)
    
    if predictor == 'previous':
        # Use previous pixel as predictor
        errors[0, 0] = channel[0, 0]
        for i in range(height):
            for j in range(width):
                if i == 0 and j == 0:
                    continue
                if j == 0:
                    predictor_value = channel[i-1, j]
                else:
                    predictor_value = channel[i, j-1]
                errors[i, j] = channel[i, j] - predictor_value
    elif predictor == 'average':
        # Use average of left and top pixels
        errors[0, 0] = channel[0, 0]
        for i in range(height):
            for j in range(width):
                if i == 0 and j == 0:
                    continue
                if i == 0:
                    predictor_value = channel[i, j-1]
                elif j == 0:
                    predictor_value = channel[i-1, j]
                else:
                    predictor_value = (channel[i, j-1] + channel[i-1, j]) // 2
                errors[i, j] = channel[i, j] - predictor_value
    
    return errors


def predictive_decode(errors, predictor='previous'):
    """
    Decode predictive coded image
    """
    if len(errors.shape) == 3:
        height, width, channels = errors.shape
        result = np.zeros_like(errors, dtype=np.uint8)
        for c in range(channels):
            result[:, :, c] = _predictive_decode_channel(errors[:, :, c], predictor)
    else:
        result = _predictive_decode_channel(errors, predictor)
    
    return result


def _predictive_decode_channel(errors, predictor):
    """Decode a single channel using predictive coding"""
    height, width = errors.shape
    reconstructed = np.zeros_like(errors, dtype=np.uint8)
    
    if predictor == 'previous':
        reconstructed[0, 0] = np.clip(errors[0, 0], 0, 255)
        for i in range(height):
            for j in range(width):
                if i == 0 and j == 0:
                    continue
                if j == 0:
                    predictor_value = reconstructed[i-1, j]
                else:
                    predictor_value = reconstructed[i, j-1]
                reconstructed[i, j] = np.clip(errors[i, j] + predictor_value, 0, 255)
    elif predictor == 'average':
        reconstructed[0, 0] = np.clip(errors[0, 0], 0, 255)
        for i in range(height):
            for j in range(width):
                if i == 0 and j == 0:
                    continue
                if i == 0:
                    predictor_value = reconstructed[i, j-1]
                elif j == 0:
                    predictor_value = reconstructed[i-1, j]
                else:
                    predictor_value = (reconstructed[i, j-1] + reconstructed[i-1, j]) // 2
                reconstructed[i, j] = np.clip(errors[i, j] + predictor_value, 0, 255)
    
    return reconstructed


# -------------------------
# Wavelet Coding
# -------------------------
def wavelet_encode(image_array, wavelet='haar', level=3):
    """
    Wavelet transform coding
    """
    if not HAS_PYWT:
        raise ImportError("PyWavelets (pywt) is required for wavelet transform. Install with: pip install PyWavelets")
    
    if len(image_array.shape) == 3:
        height, width, channels = image_array.shape
        coeffs_list = []
        for c in range(channels):
            coeffs = pywt.wavedec2(image_array[:, :, c], wavelet, level=level)
            coeffs_list.append(coeffs)
        return coeffs_list, (height, width, channels), wavelet, level
    else:
        coeffs = pywt.wavedec2(image_array, wavelet, level=level)
        return coeffs, image_array.shape, wavelet, level


def wavelet_decode(coeffs_data, original_shape, wavelet, level):
    """
    Decode wavelet transform
    """
    if not HAS_PYWT:
        raise ImportError("PyWavelets (pywt) is required for wavelet transform. Install with: pip install PyWavelets")
    
    if isinstance(coeffs_data, list):
        # Multi-channel
        height, width, channels = original_shape
        reconstructed = np.zeros((height, width, channels), dtype=np.float32)
        for c in range(channels):
            coeffs = coeffs_data[c]
            reconstructed[:, :, c] = pywt.waverec2(coeffs, wavelet)
    else:
        # Single channel
        coeffs = coeffs_data
        reconstructed = pywt.waverec2(coeffs, wavelet)
    
    # Clip and convert to uint8
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    # Crop to original size if needed
    if reconstructed.shape[:2] != original_shape[:2]:
        reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
    
    return reconstructed


# -------------------------
# Performance Metrics
# -------------------------
def calculate_compression_metrics(original, compressed_size, original_size=None):
    """Calculate compression metrics"""
    if original_size is None:
        if isinstance(original, np.ndarray):
            original_size = original.nbytes
        else:
            original_size = len(original) if hasattr(original, '__len__') else 1
    
    compression_ratio = compressed_size / original_size if original_size > 0 else 0
    space_saving = (1 - compression_ratio) * 100
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'space_saving_percent': space_saving
    }

