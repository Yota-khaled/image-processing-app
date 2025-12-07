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
    
    with gr.Blocks(title="Image Processing Application", css=custom_css, theme=gr.themes.Soft()) as demo:
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
                                sigma = gr.Slider(0.1, 5.0, 1.0, label="Gaussian Sigma")
                                median_size = gr.Slider(3, 15, 3, step=2, label="Median Size")
                                
                                with gr.Row():
                                    gaussian_btn = gr.Button("Gaussian", variant="primary", size="sm")
                                    median_btn = gr.Button("Median", variant="primary", size="sm")
                            
                            with gr.TabItem("High-Pass"):
                                with gr.Row():
                                    laplacian_btn = gr.Button("Laplacian", variant="primary", size="sm")
                                    sobel_btn = gr.Button("Sobel", variant="primary", size="sm")
                                    gradient_btn = gr.Button("Gradient", variant="primary", size="sm")
                    
                    # Histogram
                    with gr.TabItem("Histogram"):
                        with gr.Row():
                            show_hist_btn = gr.Button("Show Histogram", variant="primary", size="sm")
                            equalize_btn = gr.Button("Equalize", variant="primary", size="sm")
                    
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
        
        def apply_operation(operation_func, operation_name, *args):
            """Apply operation to CURRENT processed image (not just grayscale)"""
            if global_state['current_processed_pil'] is None:
                return None, "No image available"
            
            try:
                # Get the CURRENT processed image (could be grayscale, binary, or any other)
                img_to_process = global_state['current_processed_pil']
                
                # Apply operation to whatever image we have
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
        
        def handle_gaussian(sigma):
            return apply_operation(lambda img: gaussian_filter_func(img, sigma), 
                                  f"Gaussian (σ={sigma:.1f})")
        
        def handle_median(size):
            return apply_operation(lambda img: median_filter_func(img, size), 
                                  f"Median (size={size})")
        
        def handle_laplacian():
            return apply_operation(laplacian_filter, "Laplacian")
        
        def handle_sobel():
            return apply_operation(sobel_filter, "Sobel")
        
        def handle_gradient():
            return apply_operation(gradient_filter, "Gradient")
        
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
        
        def handle_histogram():
            if global_state['current_processed_pil'] is None:
                return None, "Upload an image first"
            
            try:
                img = global_state['current_processed_pil'].convert('L')
                img_array = np.array(img)
                
                hist, bins = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(hist, color='blue', alpha=0.7)
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Frequency')
                ax.set_title('Image Histogram')
                ax.grid(True, alpha=0.3)
                
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=100)
                plt.close(fig)
                buf.seek(0)
                
                hist_img = Image.open(buf)
                
                # Store histogram as processed image
                hist_np = pil_to_numpy(hist_img)
                global_state['current_processed_pil'] = numpy_to_pil(hist_np)
                
                return hist_np, "Histogram generated"
            except Exception as e:
                return None, f"Error: {str(e)}"
        
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
            inputs=[sigma],
            outputs=[processed_display, status_text]
        )
        
        median_btn.click(
            handle_median,
            inputs=[median_size],
            outputs=[processed_display, status_text]
        )
        
        laplacian_btn.click(
            handle_laplacian,
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        sobel_btn.click(
            handle_sobel,
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        gradient_btn.click(
            handle_gradient,
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        # Histogram
        show_hist_btn.click(
            handle_histogram,
            inputs=[],
            outputs=[processed_display, status_text]
        )
        
        equalize_btn.click(
            handle_equalize,
            inputs=[],
            outputs=[processed_display, status_text]
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