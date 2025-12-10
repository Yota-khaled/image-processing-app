"""
Gradio app handlers that orchestrate UI interactions using
the pure image processing functions and transform/compression utilities.
This keeps UI logic separate from core image processing functions.
"""

import io
import time
import tempfile
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

import image_processing as ip
import compression
import transform_coding

# Shared state for Gradio app
global_state = {
    'original_pil': None,
    'current_processed_pil': None
}


def pil_to_numpy(img):
    if img is None:
        return None
    return np.array(img)


def numpy_to_pil(arr):
    if arr is None:
        return None
    return Image.fromarray(arr)


def load_image(filepath):
    """Load image from path and initialize state."""
    if filepath is None:
        return None, None, "No image selected", "Upload an image to begin"
    try:
        img = Image.open(filepath)
        global_state['original_pil'] = img
        global_state['current_processed_pil'] = img.copy()

        info = f"Size: {img.width} x {img.height}\n"
        info += f"Mode: {img.mode}\n"
        info += f"Loaded: {time.strftime('%H:%M:%S')}"

        return pil_to_numpy(img), pil_to_numpy(img), info, "Image loaded successfully"
    except Exception as e:
        return None, None, f"Error: {str(e)}", "Failed to load image"


def _get_image_by_source(source: str):
    if source == "original":
        return global_state['original_pil']
    return global_state['current_processed_pil']


def apply_operation(operation_func, operation_name, *args, source: str = "processed"):
    """Apply an operation to selected image and update state."""
    img_to_process = _get_image_by_source(source)
    if img_to_process is None:
        return None, "No image available"
    try:
        result = operation_func(img_to_process, *args)
        if result is None:
            return None, "Operation failed"
        global_state['current_processed_pil'] = result
        return pil_to_numpy(result), f"{operation_name} applied"
    except Exception as e:
        return None, f"Error: {str(e)}"


# Basic operations on original image
def handle_grayscale():
    if global_state['original_pil'] is None:
        return None, "No original image available"
    try:
        img = global_state['original_pil'].copy()
        result = ip.to_grayscale(img)
        if result is None:
            return None, "Grayscale operation failed"
        global_state['current_processed_pil'] = result
        return pil_to_numpy(result), "Grayscale applied to original image"
    except Exception as e:
        return None, f"Error: {str(e)}"


def handle_binary_auto():
    if global_state['original_pil'] is None:
        return None, "No original image available"
    try:
        img = global_state['original_pil'].copy()
        result = ip.to_binary(img)
        if result is None:
            return None, "Binary operation failed"
        global_state['current_processed_pil'] = result
        return pil_to_numpy(result), "Binary (Auto) applied to original image"
    except Exception as e:
        return None, f"Error: {str(e)}"


def handle_binary_manual(threshold):
    if global_state['original_pil'] is None:
        return None, "No original image available"
    try:
        img = global_state['original_pil'].copy()
        result = ip.to_binary(img, threshold)
        if result is None:
            return None, "Binary operation failed"
        global_state['current_processed_pil'] = result
        return pil_to_numpy(result), f"Binary (Thresh={threshold}) applied to original image"
    except Exception as e:
        return None, f"Error: {str(e)}"


# Affine, interpolation, filters, histogram, operations
def handle_translate(tx, ty):
    return apply_operation(lambda img: ip.translate(img, tx, ty), f"Translate (x={tx}, y={ty})")


def handle_scale(sx, sy):
    return apply_operation(lambda img: ip.scale(img, sx, sy), f"Scale (x={sx:.2f}, y={sy:.2f})")


def handle_rotate(angle):
    return apply_operation(lambda img: ip.rotate(img, angle), f"Rotate ({angle}°)")


def handle_shear(direction, factor):
    if direction == "X":
        return apply_operation(lambda img: ip.shear_x(img, factor), f"Shear X (factor={factor:.2f})")
    return apply_operation(lambda img: ip.shear_y(img, factor), f"Shear Y (factor={factor:.2f})")


def handle_nearest(width, height):
    return apply_operation(lambda img: ip.nearest_neighbor_interpolation(img, (width, height)),
                          f"Nearest ({width}x{height})")


def handle_bilinear(width, height):
    return apply_operation(lambda img: ip.bilinear_interpolation(img, (width, height)),
                          f"Bilinear ({width}x{height})")


def handle_bicubic(width, height):
    return apply_operation(lambda img: ip.bicubic_interpolation(img, (width, height)),
                          f"Bicubic ({width}x{height})")


def handle_gaussian(source):
    return apply_operation(lambda img: ip.gaussian_filter_func(img, sigma=3),
                          "Gaussian (19x19, σ=3)", source=source)


def handle_median(source):
    return apply_operation(lambda img: ip.median_filter_func(img, size=7),
                          "Median (7x7)", source=source)


def handle_laplacian(source):
    return apply_operation(ip.laplacian_filter, "Laplacian", source=source)


def handle_sobel(source):
    return apply_operation(ip.sobel_filter, "Sobel", source=source)


def handle_gradient(source):
    return apply_operation(ip.gradient_filter, "Gradient", source=source)


def handle_equalize(mode="gray"):
    return apply_operation(lambda img: ip.histogram_equalization(img, mode=mode),
                          "Histogram Equalization")


def handle_crop(left, top, right, bottom):
    return apply_operation(lambda img: ip.crop_image(img, left, top, right, bottom),
                          f"Crop ({left},{top})-({right},{bottom})")


def handle_zoom(factor):
    return apply_operation(lambda img: ip.zoom_image(img, factor), f"Zoom ({factor:.2f}x)")


def handle_flip_h():
    return apply_operation(lambda img: ip.flip_image(img, 'horizontal'), "Flip Horizontal")


def handle_flip_v():
    return apply_operation(lambda img: ip.flip_image(img, 'vertical'), "Flip Vertical")


def handle_brightness(factor):
    return apply_operation(lambda img: ip.adjust_brightness(img, factor), f"Brightness ({factor})")


def handle_contrast(factor):
    return apply_operation(lambda img: ip.adjust_contrast(img, factor), f"Contrast ({factor})")


# Transform coding handlers
def handle_symbol(source):
    img = _get_image_by_source(source)
    if img is None:
        return None, "No image available"
    arr = np.array(img.convert("L"))
    data = arr.flatten().tolist()
    encoded = transform_coding.symbol_based_encode(data)
    decoded = transform_coding.symbol_based_decode(encoded)
    reconstructed = np.array(decoded, dtype=np.uint8).reshape(arr.shape)
    result = Image.fromarray(reconstructed, mode="L").convert("RGB")
    global_state['current_processed_pil'] = result
    return pil_to_numpy(result), "Symbol-based coding applied"


def handle_bitplane(source):
    img = _get_image_by_source(source)
    if img is None:
        return None, "No image available"
    arr = np.array(img.convert("L"))
    bit_planes, shape = transform_coding.bit_plane_encode(arr)
    reconstructed = transform_coding.bit_plane_decode(bit_planes, shape)
    result = Image.fromarray(reconstructed, mode="L").convert("RGB")
    global_state['current_processed_pil'] = result
    return pil_to_numpy(result), "Bit-plane coding applied"


def handle_dct(block_size, source):
    img = _get_image_by_source(source)
    if img is None:
        return None, "No image available"
    arr = np.array(img.convert("L"))
    dct_blocks, orig_shape, pad_shape, bs = transform_coding.dct_block_encode(arr, int(block_size))
    reconstructed = transform_coding.dct_block_decode(dct_blocks, orig_shape, pad_shape, bs)
    result = Image.fromarray(reconstructed, mode="L").convert("RGB")
    global_state['current_processed_pil'] = result
    return pil_to_numpy(result), f"DCT block (size={int(block_size)}) applied"



def handle_predictive(predictor, source):
    img = _get_image_by_source(source)
    if img is None:
        return None, "No image available"
    arr = np.array(img.convert("L"))
    errors = transform_coding.predictive_encode(arr, predictor)
    
    # VISUALIZE: Show errors instead of reconstruction
    vis_image = transform_coding.visualize_predictive_error(errors)
    
    result = Image.fromarray(vis_image, mode="L").convert("RGB")
    global_state['current_processed_pil'] = result
    return pil_to_numpy(result), f"Predictive ({predictor}) Errors (128=Zero)"


def handle_wavelet(wavelet, level, source):
    if not transform_coding.HAS_PYWT:
        return None, "PyWavelets not installed"
    img = _get_image_by_source(source)
    if img is None:
        return None, "No image available"
    arr = np.array(img.convert("L"))
    coeffs, orig_shape, wv, lvl = transform_coding.wavelet_encode(arr, wavelet, int(level))
    
    # VISUALIZE: Show wavelet coefficients
    vis_image = transform_coding.visualize_wavelet_coeffs(coeffs)
    
    if vis_image is None:
        return None, "Wavelet visualization failed"
        
    result = Image.fromarray(vis_image, mode="L").convert("RGB")
    global_state['current_processed_pil'] = result
    return pil_to_numpy(result), f"Wavelet ({wavelet}, level {int(level)}) Coefficients"


def _hist_quality_message(gray_array: np.ndarray) -> str:
    std_val = float(gray_array.std())
    if std_val > 50:
        return f"Histogram spread looks good (std={std_val:.1f})."
    if std_val > 25:
        return f"Histogram moderate (std={std_val:.1f}) — some contrast enhancement may help."
    return f"Histogram narrow (std={std_val:.1f}) — low contrast detected."


def handle_histogram(mode):
    if global_state['original_pil'] is None:
        return None, "Upload an image first", "Upload an image first"
    try:
        base_img = global_state['original_pil']
        hist_data = ip.show_histogram(base_img, mode=mode)
        if hist_data is None:
            return None, "Histogram unavailable", "Histogram unavailable"

        fig, ax = plt.subplots(figsize=(8, 4))
        if mode == "gray":
            gray_hist = hist_data.get("Gray")
            ax.plot(gray_hist, color='blue', alpha=0.7)
            ax.set_title('Image Histogram (GRAY)')
        else:
            for color in ['Red', 'Green', 'Blue']:
                hist = hist_data.get(color)
                ax.plot(hist, label=color, alpha=0.7, color=color.lower())
            ax.set_title('Image Histogram (RGB)')
            ax.legend()

        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)

        hist_img = Image.open(buf)
        hist_np = pil_to_numpy(hist_img)

        quality_msg = _hist_quality_message(np.array(base_img.convert("L")))

        return hist_np, quality_msg, "Histogram generated"
    except Exception as e:
        return None, f"Error: {str(e)}", f"Error: {str(e)}"


def handle_equalize_from_original(mode):
    if global_state['original_pil'] is None:
        return None, "Upload an image first"
    try:
        base_img = global_state['original_pil']
        result = ip.histogram_equalization(base_img, mode=mode)
        if result is None:
            return None, "Equalization failed"
        global_state['current_processed_pil'] = result
        return pil_to_numpy(result), f"Equalization applied ({mode.upper()})"
    except Exception as e:
        return None, f"Error: {str(e)}"


def handle_compress(selected_algs, golomb_m):
    if global_state['original_pil'] is None:
        return None, "Upload an image first", "Upload an image first"
    try:
        img_array = np.array(global_state['original_pil'].convert("L"))
        data_list = img_array.flatten().tolist()
        data_str = ''.join([chr(int(p)) for p in data_list])
        results = []
        if "Huffman" in selected_algs:
            res = compression.compress_and_report(data_str, "Huffman", compression.huffman_encode, compression.huffman_decode)
            if res:
                results.append(res)
        if "Golomb-Rice" in selected_algs:
            res = compression.compress_and_report(data_list, "Golomb-Rice", compression.golomb_rice_encode, compression.golomb_rice_decode, golomb_m)
            if res:
                results.append(res)
        if "Arithmetic" in selected_algs:
            res = compression.compress_and_report(data_str, "Arithmetic", compression.arithmetic_encode, compression.arithmetic_decode)
            if res:
                results.append(res)
        if "LZW" in selected_algs:
            res = compression.compress_and_report(data_str, "LZW", compression.lzw_encode, compression.lzw_decode)
            if res:
                results.append(res)
        if "RLE" in selected_algs:
            res = compression.compress_and_report(data_str, "RLE", compression.rle_encode, compression.rle_decode)
            if res:
                results.append(res)
        if not results:
            return None, "No results", "No compression results"
        lines = ["| Algorithm | Original | Compressed | Ratio% | Encode s | Decode s | Correct/Error |",
                 "|---|---|---|---|---|---|---|"]

        def _fmt(val, fmt="{:.2f}"):
            try:
                return fmt.format(val) if val is not None else "-"
            except Exception:
                return "-"

        had_error = False
        for r in results:
            if r.get("error"):
                had_error = True
                lines.append(f"| {r.get('algorithm')} | - | - | - | - | - | {r.get('error')} |")
                continue
            lines.append(
                "| {alg} | {orig} | {comp} | {ratio} | {enc} | {dec} | {corr} |".format(
                    alg=r.get("algorithm"),
                    orig=r.get("original_size"),
                    comp=r.get("compressed_size"),
                    ratio=_fmt(r.get("compression_ratio")),
                    enc=_fmt(r.get("encode_time"), "{:.4f}"),
                    dec=_fmt(r.get("decode_time"), "{:.4f}"),
                    corr=r.get("is_correct"),
                )
            )
        md = "\n".join(lines)
        img_np = pil_to_numpy(global_state['current_processed_pil']) if global_state['current_processed_pil'] is not None else None
        status_msg = "Compression complete" if not had_error else "Compression finished (some errors)"
        return img_np, md, status_msg
    except Exception as e:
        return None, f"Compression error: {str(e)}", f"Compression error: {str(e)}"


def download_processed_image():
    if global_state['current_processed_pil'] is None:
        raise gr.Error("No processed image to download")
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_path = temp_file.name
        temp_file.close()
        global_state['current_processed_pil'].save(temp_path, 'PNG')
        return gr.update(value=temp_path, visible=True)
    except Exception as e:
        raise gr.Error(f"Download error: {str(e)}")


def reset_all():
    global_state['original_pil'] = None
    global_state['current_processed_pil'] = None
    return (
        None,  # image_input
        None,  # original_display
        None,  # processed_display
        "Upload an image to see information",  # image_info
        "Application reset. Upload an image to begin",  # status_text
        gr.update(value=None, visible=False)   # download_file
    )
