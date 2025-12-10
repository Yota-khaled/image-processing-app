"""
Gradio Image Processing App 
"""
import gradio as gr
from ui_components import GRADIO_CUSTOM_CSS

# import handlers (UI orchestration) and pure functions
from app_handlers import (
    global_state,
    pil_to_numpy,
    numpy_to_pil,
    load_image,
    handle_grayscale,
    handle_binary_auto,
    handle_binary_manual,
    handle_translate,
    handle_scale,
    handle_rotate,
    handle_shear,
    handle_nearest,
    handle_bilinear,
    handle_bicubic,
    handle_gaussian,
    handle_median,
    handle_laplacian,
    handle_sobel,
    handle_gradient,
    handle_equalize_from_original,
    handle_crop,
    handle_zoom,
    handle_flip_h,
    handle_flip_v,
    handle_brightness,
    handle_contrast,
    handle_symbol,
    handle_bitplane,
    handle_dct,
    handle_predictive,
    handle_wavelet,
    handle_histogram,
    handle_compress,
    download_processed_image,
    reset_all,
)


def create_gradio_app():
    # Use Blocks without css arg (not supported), inject CSS manually
    with gr.Blocks(title="Image Processing Application", theme=gr.themes.Soft()) as demo:
        gr.HTML(f"<style>{GRADIO_CUSTOM_CSS}</style>")
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
                        show_fullscreen_button=True,
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
                            show_fullscreen_button=True,
                            elem_classes="large-image" 
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
            outputs=[processed_display, hist_quality_box, status_text]
        )
        
        equalize_btn.click(
            handle_equalize_from_original,
            inputs=[eq_mode_dd],
            outputs=[processed_display, status_text]
        )

        # Compression
        compress_btn.click(
            handle_compress,
            inputs=[algs, golomb_m],
            outputs=[processed_display, compress_output, status_text]
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
