"""
Gradio Image Processing App - FINAL FIXED VERSION
"""
import gradio as gr
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import time
import compression
import transform_coding

# Import your existing image processing functions
try:
    from image_processing import (
        to_grayscale, to_binary,
        translate, scale, rotate, shear_x, shear_y,
        nearest_neighbor_interpolation, bilinear_interpolation, bicubic_interpolation,
        show_histogram, histogram_equalization,
        gaussian_filter_func, median_filter_func,
        laplacian_filter, sobel_filter, gradient_filter,
        crop_image, zoom_image, flip_image, adjust_brightness, adjust_contrast
    )
    print("Successfully imported image_processing functions")
except ImportError as e:
    print(f"Error importing image_processing: {e}")
    print("Using fallback functions")
    
    def to_grayscale(img):
        return img.convert('L').convert('RGB')
    
    def to_binary(img, threshold=128):
        gray = img.convert('L')
        binary = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
        return binary.convert('RGB')
    
    def translate(img, tx, ty):
        return img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=(0, 0, 0))
    
    def scale(img, sx, sy):
        new_size = (int(img.size[0] * sx), int(img.size[1] * sy))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    def rotate(img, angle):
        return img.rotate(angle, expand=True, fillcolor=(0, 0, 0))
    
    def shear_x(img, factor):
        return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), fillcolor=(0, 0, 0))
    
    def shear_y(img, factor):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), fillcolor=(0, 0, 0))
    
    def nearest_neighbor_interpolation(img, new_size):
        return img.resize(new_size, Image.Resampling.NEAREST)
    
    def bilinear_interpolation(img, new_size):
        return img.resize(new_size, Image.Resampling.BILINEAR)
    
    def bicubic_interpolation(img, new_size):
        return img.resize(new_size, Image.Resampling.BICUBIC)
    
    def gaussian_filter_func(img, sigma):
        from PIL import ImageFilter
        return img.filter(ImageFilter.GaussianBlur(sigma))
    
    def median_filter_func(img, size):
        from PIL import ImageFilter
        return img.filter(ImageFilter.MedianFilter(size))
    
    def laplacian_filter(img):
        from PIL import ImageFilter
        return img.convert('L').filter(ImageFilter.FIND_EDGES).convert('RGB')
    
    def sobel_filter(img):
        from PIL import ImageFilter
        return img.convert('L').filter(ImageFilter.FIND_EDGES).convert('RGB')
    
    def gradient_filter(img):
        return sobel_filter(img)
    
    def histogram_equalization(img):
        from PIL import ImageOps
        return ImageOps.equalize(img.convert('L')).convert('RGB')
    
    def crop_image(img, left, top, right, bottom):
        return img.crop((left, top, right, bottom))
    
    def zoom_image(img, factor):
        new_size = (int(img.size[0] * factor), int(img.size[1] * factor))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    def flip_image(img, direction):
        if direction == 'horizontal':
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    
    def adjust_brightness(img, factor):
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.0 + factor/100.0)
    
    def adjust_contrast(img, factor):
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.0 + factor/100.0)

def pil_to_numpy(img):
    if img is None:
        return None
    return np.array(img)

def numpy_to_pil(arr):
    if arr is None:
        return None
    return Image.fromarray(arr)

def create_gradio_app():
    # Global state
    global_state = {
        'original_pil': None,
        'current_processed_pil': None
    }
    
    # Custom CSS to make images side by side
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .image-row {
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        gap: 20px !important;
        margin-bottom: 20px !important;
    }
    .image-col {
        flex: 1 !important;
        min-width: 0 !important;
    }
    .image-container {
        border: 2px solid #4A70A9;
        border-radius: 10px;
        padding: 10px;
        background: #1E2B3A;
    }
    .header {
        text-align: center;
        background: linear-gradient(135deg, #4A70A9 0%, #2A4A7A 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .large-image {
        min-height: 400px !important;
    }

    .processed-image-gap {
        margin-top: 50px !important;
    }
    """
    
    # Use Blocks without css arg (not supported), inject CSS manually
    with gr.Blocks(title="Image Processing Application", theme=gr.themes.Soft()) as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        # Header
        gr.Markdown("""
        <div class="header">
            <h1>Image Processing Application</h1>
        </div>
        """)
        
        with gr.Row():
            # ===== LEFT COLUMN: UPLOAD AND ORIGINAL IMAGE =====
            with gr.Column(scale=2):
                # Upload section
                gr.Markdown("### Upload Image")
                image_input = gr.Image(
                    label="Drag & Drop or Click to Upload",
                    type="filepath",
                    height=200
                )
                
                # ORIGINAL IMAGE ONLY in left column
                with gr.Column(elem_classes="image-col"):
                    gr.Markdown("**Original Image**", elem_classes="image-container")
                    original_display = gr.Image(
                        label="",
                        type="numpy",
                        height=400,
                        show_label=False,
                        elem_classes="large-image"
                    )
                
                # Information and controls
                with gr.Accordion("Image Information", open=False):
                    image_info = gr.Textbox(
                        label="Info",
                        lines=3,
                        interactive=False,
                        value="Upload an image to see information"
                    )
                
                # Control buttons
                with gr.Row():
                    reset_btn = gr.Button("Reset All", variant="secondary", size="sm")
                    download_btn = gr.Button("Download Processed", variant="primary", size="sm")
                
                download_file = gr.File(label="Download", visible=False)
                
                # Status
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Upload an image to begin"
                )
            
            # ===== RIGHT COLUMN: OPERATIONS AND PROCESSED IMAGE =====
            with gr.Column(scale=3):        
                # Operations section
                gr.Markdown("### Processing Operations")
                
                with gr.Tabs():
                    # Basic Operations
                    with gr.TabItem("Basic"):
                        with gr.Row():
                            grayscale_btn = gr.Button("Grayscale", variant="primary", size="sm")
                            binary_auto_btn = gr.Button("Binary (Auto)", variant="primary", size="sm")
                        
                        with gr.Accordion("Binary (Manual)", open=False):
                            binary_threshold = gr.Slider(0, 255, 128, label="Threshold")
                            binary_manual_btn = gr.Button("Apply Binary", variant="secondary", size="sm")
                    
                    # Affine Transformations
                    with gr.TabItem("Affine"):
                        with gr.Tabs():
                            with gr.TabItem("Translate"):
                                tx = gr.Slider(-100, 100, 0, label="X Translation")
                                ty = gr.Slider(-100, 100, 0, label="Y Translation")
                                translate_btn = gr.Button("Apply Translation", variant="primary", size="sm")
                            
                            with gr.TabItem("Scale"):
                                scale_x = gr.Slider(0.1, 3.0, 1.0, label="Scale X")
                                scale_y = gr.Slider(0.1, 3.0, 1.0, label="Scale Y")
                                scale_btn = gr.Button("Apply Scale", variant="primary", size="sm")
                            
                            with gr.TabItem("Rotate"):
                                angle = gr.Slider(-180, 180, 0, label="Angle")
                                rotate_btn = gr.Button("Apply Rotation", variant="primary", size="sm")
                            
                            with gr.TabItem("Shear"):
                                shear_dir = gr.Radio(["X", "Y"], label="Direction", value="X")
                                shear_factor = gr.Slider(-1.0, 1.0, 0.0, label="Factor")
                                shear_btn = gr.Button("Apply Shear", variant="primary", size="sm")
                    
                    # Interpolation
                    with gr.TabItem("Interpolation"):
                        new_width = gr.Slider(100, 2000, 800, label="Width")
                        new_height = gr.Slider(100, 2000, 600, label="Height")
                        
                        with gr.Row():
                            nn_btn = gr.Button("Nearest", variant="primary", size="sm")
                            bilinear_btn = gr.Button("Bilinear", variant="primary", size="sm")
                            bicubic_btn = gr.Button("Bicubic", variant="primary", size="sm")
                    
                    # Filters
                    with gr.TabItem("Filters"):
                        with gr.Tabs():
                            with gr.TabItem("Low-Pass"):
                                filter_source = gr.Dropdown(["original", "processed"], value="processed", label="Apply on")
                                with gr.Row():
                                    gaussian_btn = gr.Button("Gaussian (19x19, σ=3)", variant="primary", size="sm")
                                    median_btn = gr.Button("Median (7x7)", variant="primary", size="sm")
                            
                            with gr.TabItem("High-Pass"):
                                filter_source_hp = gr.Dropdown(["original", "processed"], value="processed", label="Apply on")
                                with gr.Row():
                                    laplacian_btn = gr.Button("Laplacian", variant="primary", size="sm")
                                    sobel_btn = gr.Button("Sobel", variant="primary", size="sm")
                                    gradient_btn = gr.Button("Gradient", variant="primary", size="sm")
                    
                    # Histogram
                    with gr.TabItem("Histogram"):
                        with gr.Row():
                            hist_mode_dd = gr.Dropdown(["gray", "rgb"], value="gray", label="Histogram mode (from original)")
                            eq_mode_dd = gr.Dropdown(["gray", "rgb"], value="gray", label="Equalization mode (from original)")
                        with gr.Row():
                            show_hist_btn = gr.Button("Show Histogram", variant="primary", size="sm")
                            equalize_btn = gr.Button("Equalize", variant="primary", size="sm")
                        with gr.Row():
                            hist_quality_box = gr.Textbox(label="Histogram quality", interactive=False)
                    
                    # Compression
                    with gr.TabItem("Compression"):
                        with gr.Row():
                            algs = gr.CheckboxGroup(
                                ["Huffman", "Golomb-Rice", "Arithmetic", "LZW", "RLE"],
                                value=["Huffman", "Golomb-Rice"],
                                label="Algorithms (original image, grayscale)"
                            )
                            golomb_m = gr.Slider(2, 16, value=4, step=2, label="Golomb-Rice m (power of 2)")
                        compress_btn = gr.Button("Run Compression", variant="primary")
                        compress_output = gr.Markdown("Compression results will appear here.")
                    
                    # Image Operations
                    with gr.TabItem("Operations"):
                        with gr.Tabs():
                            with gr.TabItem("Crop"):
                                crop_left = gr.Number(0, label="Left")
                                crop_top = gr.Number(0, label="Top")
                                crop_right = gr.Number(100, label="Right")
                                crop_bottom = gr.Number(100, label="Bottom")
                                crop_btn = gr.Button("Crop", variant="primary", size="sm")
                            
                            with gr.TabItem("Zoom/Flip"):
                                zoom_factor = gr.Slider(0.1, 5.0, 1.0, label="Zoom Factor")
                                
                                with gr.Row():
                                    zoom_in_btn = gr.Button("Zoom In (2x)", variant="primary", size="sm")
                                    zoom_out_btn = gr.Button("Zoom Out (0.5x)", variant="primary", size="sm")
                                    zoom_custom_btn = gr.Button("Custom Zoom", variant="primary", size="sm")
                                
                                with gr.Row():
                                    flip_h_btn = gr.Button("Flip Horizontal", variant="primary", size="sm")
                                    flip_v_btn = gr.Button("Flip Vertical", variant="primary", size="sm")
                            
                            with gr.TabItem("Brightness/Contrast"):
                                brightness_val = gr.Slider(-100, 100, 0, label="Brightness")
                                contrast_val = gr.Slider(-100, 100, 0, label="Contrast")
                                
                                with gr.Row():
                                    brightness_btn = gr.Button("Brightness", variant="primary", size="sm")
                                    contrast_btn = gr.Button("Contrast", variant="primary", size="sm")

                            with gr.TabItem("Transform Coding"):
                                tc_source = gr.Dropdown(["original", "processed"], value="processed", label="Apply on")
                                with gr.Row():
                                    symbol_btn = gr.Button("Symbol-based", variant="primary", size="sm")
                                    bitplane_btn = gr.Button("Bit-plane", variant="primary", size="sm")
                                with gr.Row():
                                    dct_block = gr.Slider(4, 16, value=8, step=2, label="DCT Block Size")
                                    dct_btn = gr.Button("DCT Block", variant="primary", size="sm")
                                with gr.Row():
                                    predictor_dd = gr.Dropdown(["previous", "average"], value="previous", label="Predictor")
                                    predictive_btn = gr.Button("Predictive", variant="primary", size="sm")
                                with gr.Row():
                                    wavelet_dd = gr.Dropdown(["haar", "db4", "bior2.2"], value="haar", label="Wavelet")
                                    wavelet_level = gr.Slider(1, 5, value=3, step=1, label="Level")
                                    wavelet_btn = gr.Button("Wavelet", variant="primary", size="sm")


                    # PROCESSED IMAGE at the top of right column
                    with gr.Column(elem_classes="image-col processed-image-gap"):  
                        gr.Markdown("**Processed Image**", elem_classes="image-container")
                        processed_display = gr.Image(
                            label="",
                            type="numpy",
                            height=400,
                            show_label=False,
                            elem_classes="large-image" 
                        )           



        
        # ===== EVENT HANDLERS =====
        
        def load_image(filepath):
            """Load and initialize image"""
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
            """Apply operation to chosen source image ('original' or 'processed')."""
            img_to_process = _get_image_by_source(source)
            if img_to_process is None:
                return None, "No image available"
            
            try:
                result = operation_func(img_to_process, *args)
                
                if result is None:
                    return None, f"Operation failed"
                
                # Update state with the result
                global_state['current_processed_pil'] = result
                
                return pil_to_numpy(result), f"{operation_name} applied"
            except Exception as e:
                return None, f"Error: {str(e)}"

        # ===== NEW: FUNCTIONS THAT APPLY TO ORIGINAL IMAGE =====
        
        def handle_grayscale():
            """Apply grayscale to ORIGINAL image"""
            if global_state['original_pil'] is None:
                return None, "No original image available"
            
            try:
                # Get the ORIGINAL image
                img_to_process = global_state['original_pil'].copy()
                
                # Apply grayscale to original image
                result = to_grayscale(img_to_process)
                
                if result is None:
                    return None, "Grayscale operation failed"
                
                # Update state with the result
                global_state['current_processed_pil'] = result
                
                return pil_to_numpy(result), "Grayscale applied to original image"
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        def handle_binary_auto():
            """Apply binary to ORIGINAL image"""
            if global_state['original_pil'] is None:
                return None, "No original image available"
            
            try:
                # Get the ORIGINAL image
                img_to_process = global_state['original_pil'].copy()
                
                # Apply binary to original image
                result = to_binary(img_to_process)
                
                if result is None:
                    return None, "Binary operation failed"
                
                # Update state with the result
                global_state['current_processed_pil'] = result
                
                return pil_to_numpy(result), "Binary (Auto) applied to original image"
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        def handle_binary_manual(threshold):
            """Apply binary with threshold to ORIGINAL image"""
            if global_state['original_pil'] is None:
                return None, "No original image available"
            
            try:
                # Get the ORIGINAL image
                img_to_process = global_state['original_pil'].copy()
                
                # Apply binary to original image
                result = to_binary(img_to_process, threshold)
                
                if result is None:
                    return None, "Binary operation failed"
                
                # Update state with the result
                global_state['current_processed_pil'] = result
                
                return pil_to_numpy(result), f"Binary (Thresh={threshold}) applied to original image"
            except Exception as e:
                return None, f"Error: {str(e)}"

        # ===== OLD FUNCTIONS (keep as is for other operations) =====
        
        def handle_translate(tx, ty):
            return apply_operation(lambda img: translate(img, tx, ty), f"Translate (x={tx}, y={ty})")
        
        def handle_scale(sx, sy):
            return apply_operation(lambda img: scale(img, sx, sy), f"Scale (x={sx:.2f}, y={sy:.2f})")
        
        def handle_rotate(angle):
            return apply_operation(lambda img: rotate(img, angle), f"Rotate ({angle}°)")
        
        def handle_shear(direction, factor):
            if direction == "X":
                return apply_operation(lambda img: shear_x(img, factor), f"Shear X (factor={factor:.2f})")
            else:
                return apply_operation(lambda img: shear_y(img, factor), f"Shear Y (factor={factor:.2f})")
        
        def handle_nearest(width, height):
            return apply_operation(lambda img: nearest_neighbor_interpolation(img, (width, height)), 
                                  f"Nearest ({width}x{height})")
        
        def handle_bilinear(width, height):
            return apply_operation(lambda img: bilinear_interpolation(img, (width, height)), 
                                  f"Bilinear ({width}x{height})")
        
        def handle_bicubic(width, height):
            return apply_operation(lambda img: bicubic_interpolation(img, (width, height)), 
                                  f"Bicubic ({width}x{height})")
        
        def handle_gaussian(source):
            # Fixed 19x19 (truncate approx 3.0) and sigma=3
            return apply_operation(lambda img: gaussian_filter_func(img, sigma=3), 
                                  "Gaussian (19x19, σ=3)", source=source)
        
        def handle_median(source):
            return apply_operation(lambda img: median_filter_func(img, size=7), 
                                  "Median (7x7)", source=source)
        
        def handle_laplacian(source):
            return apply_operation(laplacian_filter, "Laplacian", source=source)
        
        def handle_sobel(source):
            return apply_operation(sobel_filter, "Sobel", source=source)
        
        def handle_gradient(source):
            return apply_operation(gradient_filter, "Gradient", source=source)
        
        def handle_equalize():
            return apply_operation(histogram_equalization, "Histogram Equalization")
        
        def handle_crop(left, top, right, bottom):
            return apply_operation(lambda img: crop_image(img, left, top, right, bottom), 
                                  f"Crop ({left},{top})-({right},{bottom})")
        
        def handle_zoom(factor):
            return apply_operation(lambda img: zoom_image(img, factor), f"Zoom ({factor:.2f}x)")
        
        def handle_flip_h():
            return apply_operation(lambda img: flip_image(img, 'horizontal'), "Flip Horizontal")
        
        def handle_flip_v():
            return apply_operation(lambda img: flip_image(img, 'vertical'), "Flip Vertical")
        
        def handle_brightness(factor):
            return apply_operation(lambda img: adjust_brightness(img, factor), f"Brightness ({factor})")
        
        def handle_contrast(factor):
            return apply_operation(lambda img: adjust_contrast(img, factor), f"Contrast ({factor})")
        
        # Transform coding handlers (use grayscale conversion inside)
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
            reconstructed = transform_coding.predictive_decode(errors, predictor)
            result = Image.fromarray(reconstructed, mode="L").convert("RGB")
            global_state['current_processed_pil'] = result
            return pil_to_numpy(result), f"Predictive ({predictor}) applied"

        def handle_wavelet(wavelet, level, source):
            if not transform_coding.HAS_PYWT:
                return None, "PyWavelets not installed"
            img = _get_image_by_source(source)
            if img is None:
                return None, "No image available"
            arr = np.array(img.convert("L"))
            coeffs, orig_shape, wv, lvl = transform_coding.wavelet_encode(arr, wavelet, int(level))
            reconstructed = transform_coding.wavelet_decode(coeffs, orig_shape, wv, lvl)
            result = Image.fromarray(reconstructed, mode="L").convert("RGB")
            global_state['current_processed_pil'] = result
            return pil_to_numpy(result), f"Wavelet ({wavelet}, level {int(level)}) applied"
        
        def _hist_quality_message(gray_array: np.ndarray) -> str:
            std_val = float(gray_array.std())
            if std_val > 50:
                return f"Histogram spread looks good (std={std_val:.1f})."
            elif std_val > 25:
                return f"Histogram moderate (std={std_val:.1f}) — some contrast enhancement may help."
            else:
                return f"Histogram narrow (std={std_val:.1f}) — low contrast detected."

        def handle_histogram(mode):
            # Always start from ORIGINAL image
            if global_state['original_pil'] is None:
                return None, "Upload an image first"
            
            try:
                base_img = global_state['original_pil']
                if mode == "gray":
                    img = base_img.convert('L')
                    img_array = np.array(img)
                    hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(hist, color='blue', alpha=0.7)
                    ax.set_xlabel('Pixel Intensity')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Image Histogram (GRAY)')
                    ax.grid(True, alpha=0.3)
                else:
                    img = base_img.convert('RGB')
                    img_array = np.array(img)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['red', 'green', 'blue']
                    for i, c in enumerate(colors):
                        hist, _ = np.histogram(img_array[:, :, i].flatten(), bins=256, range=(0, 256))
                        ax.plot(hist, color=c, alpha=0.7, label=c.title())
                    ax.set_xlabel('Pixel Intensity')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Image Histogram (RGB)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=100)
                plt.close(fig)
                buf.seek(0)
                
                hist_img = Image.open(buf)
                hist_np = pil_to_numpy(hist_img)
                
                quality_msg = _hist_quality_message(np.array(base_img.convert("L")))
                
                # Do not change processed image for histogram view
                return hist_np, quality_msg
            except Exception as e:
                return None, f"Error: {str(e)}"

        def handle_equalize(mode):
            # Equalize from ORIGINAL image using selected mode
            if global_state['original_pil'] is None:
                return None, "Upload an image first"
            try:
                base_img = global_state['original_pil']
                result = histogram_equalization(base_img, mode=mode)
                if result is None:
                    return None, "Equalization failed"
                global_state['current_processed_pil'] = result
                return pil_to_numpy(result), f"Equalization applied ({mode.upper()})"
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        # Compression (Member 9) - run on ORIGINAL image grayscale
        def handle_compress(selected_algs, golomb_m):
            if global_state['original_pil'] is None:
                return None, "Upload an image first"
            try:
                img_array = np.array(global_state['original_pil'].convert("L"))
                data_list = img_array.flatten().tolist()
                data_str = ''.join([chr(int(p)) for p in data_list])
                results = []
                if "Huffman" in selected_algs:
                    res = compression.compress_and_report(data_str, "Huffman", compression.huffman_encode, compression.huffman_decode)
                    if res: results.append(res)
                if "Golomb-Rice" in selected_algs:
                    res = compression.compress_and_report(data_list, "Golomb-Rice", compression.golomb_rice_encode, compression.golomb_rice_decode, golomb_m)
                    if res: results.append(res)
                if "Arithmetic" in selected_algs:
                    res = compression.compress_and_report(data_str, "Arithmetic", compression.arithmetic_encode, compression.arithmetic_decode)
                    if res: results.append(res)
                if "LZW" in selected_algs:
                    res = compression.compress_and_report(data_str, "LZW", compression.lzw_encode, compression.lzw_decode)
                    if res: results.append(res)
                if "RLE" in selected_algs:
                    res = compression.compress_and_report(data_str, "RLE", compression.rle_encode, compression.rle_decode)
                    if res: results.append(res)
                if not results:
                    return None, "No results"
                # Format results as markdown table
                lines = ["| Algorithm | Original | Compressed | Ratio% | Encode s | Decode s | Correct |",
                         "|---|---|---|---|---|---|---|"]
                for r in results:
                    lines.append(f"| {r.get('algorithm')} | {r.get('original_size')} | {r.get('compressed_size')} | {r.get('compression_ratio'):.2f} | {r.get('encode_time'):.4f} | {r.get('decode_time'):.4f} | {r.get('is_correct')} |")
                md = "\n".join(lines)
                # Do not alter the current processed image; return current display image (if any)
                img_np = pil_to_numpy(global_state['current_processed_pil']) if global_state['current_processed_pil'] is not None else None
                return img_np, md
            except Exception as e:
                return None, f"Compression error: {str(e)}"
        
        def download_processed_image():
            if global_state['current_processed_pil'] is None:
                raise gr.Error("No processed image to download")
            
            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_path = temp_file.name
                temp_file.close()
                
                global_state['current_processed_pil'].save(temp_path, 'PNG')
                return temp_path
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
                None   # download_file
            )
        
        # ===== CONNECT HANDLERS =====
        
        # Load image
        image_input.change(
            load_image,
            inputs=[image_input],
            outputs=[original_display, processed_display, image_info, status_text]
        )
        
        # Connect ALL buttons to their handlers
        # Basic operations - NOW APPLY TO ORIGINAL IMAGE
        grayscale_btn.click(
            handle_grayscale,
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        binary_auto_btn.click(
            handle_binary_auto,
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        binary_manual_btn.click(
            handle_binary_manual,
            inputs=[binary_threshold],
            outputs=[processed_display, status_text]
        )
        
        # Affine transformations
        translate_btn.click(
            handle_translate,
            inputs=[tx, ty],
            outputs=[processed_display, status_text]
        )
        
        scale_btn.click(
            handle_scale,
            inputs=[scale_x, scale_y],
            outputs=[processed_display, status_text]
        )
        
        rotate_btn.click(
            handle_rotate,
            inputs=[angle],
            outputs=[processed_display, status_text]
        )
        
        shear_btn.click(
            handle_shear,
            inputs=[shear_dir, shear_factor],
            outputs=[processed_display, status_text]
        )
        
        # Interpolation
        nn_btn.click(
            handle_nearest,
            inputs=[new_width, new_height],
            outputs=[processed_display, status_text]
        )
        
        bilinear_btn.click(
            handle_bilinear,
            inputs=[new_width, new_height],
            outputs=[processed_display, status_text]
        )
        
        bicubic_btn.click(
            handle_bicubic,
            inputs=[new_width, new_height],
            outputs=[processed_display, status_text]
        )
        
        # Filters
        gaussian_btn.click(
            handle_gaussian,
            inputs=[filter_source],
            outputs=[processed_display, status_text]
        )
        
        median_btn.click(
            handle_median,
            inputs=[filter_source],
            outputs=[processed_display, status_text]
        )
        
        laplacian_btn.click(
            handle_laplacian,
            inputs=[filter_source_hp],
            outputs=[processed_display, status_text]
        )
        
        sobel_btn.click(
            handle_sobel,
            inputs=[filter_source_hp],
            outputs=[processed_display, status_text]
        )
        
        gradient_btn.click(
            handle_gradient,
            inputs=[filter_source_hp],
            outputs=[processed_display, status_text]
        )
        
        # Histogram
        show_hist_btn.click(
            handle_histogram,
            inputs=[hist_mode_dd],
            outputs=[processed_display, hist_quality_box]
        )
        
        equalize_btn.click(
            handle_equalize,
            inputs=[eq_mode_dd],
            outputs=[processed_display, status_text]
        )

        # Compression
        compress_btn.click(
            handle_compress,
            inputs=[algs, golomb_m],
            outputs=[processed_display, compress_output]
        )
        
        # Image operations
        crop_btn.click(
            handle_crop,
            inputs=[crop_left, crop_top, crop_right, crop_bottom],
            outputs=[processed_display, status_text]
        )
        
        zoom_in_btn.click(
            lambda: handle_zoom(2.0),
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        zoom_out_btn.click(
            lambda: handle_zoom(0.5),
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        zoom_custom_btn.click(
            handle_zoom,
            inputs=[zoom_factor],
            outputs=[processed_display, status_text]
        )
        
        flip_h_btn.click(
            handle_flip_h,
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        flip_v_btn.click(
            handle_flip_v,
            inputs=[],
            outputs=[processed_display, status_text]
        )

        # Transform coding
        symbol_btn.click(
            handle_symbol,
            inputs=[tc_source],
            outputs=[processed_display, status_text]
        )
        
        bitplane_btn.click(
            handle_bitplane,
            inputs=[tc_source],
            outputs=[processed_display, status_text]
        )
        
        dct_btn.click(
            handle_dct,
            inputs=[dct_block, tc_source],
            outputs=[processed_display, status_text]
        )
        
        predictive_btn.click(
            handle_predictive,
            inputs=[predictor_dd, tc_source],
            outputs=[processed_display, status_text]
        )
        
        wavelet_btn.click(
            handle_wavelet,
            inputs=[wavelet_dd, wavelet_level, tc_source],
            outputs=[processed_display, status_text]
        )
        
        brightness_btn.click(
            handle_brightness,
            inputs=[brightness_val],
            outputs=[processed_display, status_text]
        )
        
        contrast_btn.click(
            handle_contrast,
            inputs=[contrast_val],
            outputs=[processed_display, status_text]
        )
        
        # Control buttons
        reset_btn.click(
            reset_all,
            inputs=[],
            outputs=[image_input, original_display, processed_display, image_info, status_text, download_file]
        )
        
        download_btn.click(
            download_processed_image,
            inputs=[],
            outputs=download_file
        )
        
        
    return demo

# ==================== RUN THE APP ====================
if __name__ == "__main__":

    print("=" * 60)
    print("App will be available at http://localhost:7860")
    print("=" * 60)
    
    app = create_gradio_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )