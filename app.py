"""
Main Streamlit application for Image Processing
This is the entry point of the application
"""

import streamlit as st
import io

# Import local modules
from config import APP_CONFIG, COLORS
from ui_components import apply_custom_css, render_footer, initialize_session_state
from utils import load_image, display_image_info
from image_processing import (
    to_grayscale, to_binary,
    translate, scale, rotate, shear_x, shear_y,
    nearest_neighbor_interpolation, bilinear_interpolation, bicubic_interpolation,
    show_histogram, histogram_equalization,
    gaussian_filter_func, median_filter_func,
    laplacian_filter, sobel_filter, gradient_filter
)

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)

# Apply custom CSS
apply_custom_css()

# Initialize session state
initialize_session_state()

# Main App
st.title("Image Processing Application")
st.markdown("---")

# Sidebar for image upload and controls
with st.sidebar:
    st.header("Image Controls")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image to process"
    )
    
    if uploaded_file is not None:
        st.session_state.original_image = load_image(uploaded_file)
        st.session_state.processed_image = None
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", use_container_width=True):
            if st.session_state.processed_image:
                buf = io.BytesIO()
                st.session_state.processed_image.save(buf, format='PNG')
                st.download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
            else:
                st.warning("No processed image to save")
    
    with col2:
        if st.button("Reset", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = st.session_state.original_image.copy()
                st.rerun()
            else:
                st.warning("No image to reset")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    if st.session_state.original_image:
        st.image(st.session_state.original_image, use_container_width=True)
        with st.expander("Image Information"):
            st.markdown(display_image_info(st.session_state.original_image))
    else:
        st.info("Upload an image from the sidebar to get started")

with col2:
    st.subheader("Processed Image")
    if st.session_state.processed_image:
        st.image(st.session_state.processed_image, use_container_width=True)
    elif st.session_state.original_image:
        st.info("Select an operation below to process the image")
    else:
        st.info("Result will appear here after processing")

st.markdown("---")

# Processing Tools
st.header("Processing Tools")

# Basic Operations
with st.expander("Basic Operations", expanded=True):
    # Initialize binary selected state
    if 'binary_selected' not in st.session_state:
        st.session_state.binary_selected = False
    
    # All buttons side by side in the same line
    col1, col2, col3 = st.columns(3)
    

    with col1:
        if st.button("Grayscale", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = to_grayscale(st.session_state.original_image)
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col2:
        if st.button("Binary", use_container_width=True, key="binary_button"):
            if st.session_state.original_image:
                st.session_state.binary_selected = not st.session_state.binary_selected
                # Initialize threshold if not set
                if 'binary_threshold' not in st.session_state:
                    st.session_state.binary_threshold = 128
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    # Show threshold slider when Binary is selected
    if st.session_state.binary_selected:
        st.markdown("---")
        threshold = st.slider(
            "Binary Threshold", 
            0, 255, 
            st.session_state.get('binary_threshold', 128), 
            key="binary_threshold_slider"
        )
        st.session_state.binary_threshold = threshold
        
        # Apply button
        if st.button("Apply Binary", use_container_width=True, key="apply_binary"):
            if st.session_state.original_image:
                st.session_state.processed_image = to_binary(
                    st.session_state.original_image, 
                    st.session_state.binary_threshold
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")

# Affine Transformations
with st.expander("Affine Transformations"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Translate**")
        tx = st.slider("X", -100, 100, 0, key="tx")
        ty = st.slider("Y", -100, 100, 0, key="ty")
        if st.button("Translate", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = translate(
                    st.session_state.original_image, tx, ty
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col2:
        st.write("**Scale**")
        scale_x = st.slider("Scale X", 0.1, 3.0, 1.0, 0.1, key="scale_x")
        scale_y = st.slider("Scale Y", 0.1, 3.0, 1.0, 0.1, key="scale_y")
        if st.button("Scale", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = scale(
                    st.session_state.original_image, scale_x, scale_y
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col3:
        st.write("**Rotate**")
        angle = st.slider("Angle (degrees)", -180, 180, 0, key="angle")
        if st.button("Rotate", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = rotate(
                    st.session_state.original_image, angle
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    col4, col5 = st.columns(2)
    with col4:
        shear_x_val = st.slider("Shear X", -1.0, 1.0, 0.0, 0.1, key="shear_x")
        if st.button("Shear X", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = shear_x(
                    st.session_state.original_image, shear_x_val
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col5:
        shear_y_val = st.slider("Shear Y", -1.0, 1.0, 0.0, 0.1, key="shear_y")
        if st.button("Shear Y", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = shear_y(
                    st.session_state.original_image, shear_y_val
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")

# Image Interpolation
with st.expander("Image Interpolation"):
    col1, col2, col3 = st.columns(3)
    
    new_width = st.slider("New Width", 100, 2000, 800, 50, key="interp_width")
    new_height = st.slider("New Height", 100, 2000, 600, 50, key="interp_height")
    
    with col1:
        if st.button("Nearest Neighbor", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = nearest_neighbor_interpolation(
                    st.session_state.original_image, (new_width, new_height)
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col2:
        if st.button("Bilinear", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = bilinear_interpolation(
                    st.session_state.original_image, (new_width, new_height)
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col3:
        if st.button("Bicubic", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = bicubic_interpolation(
                    st.session_state.original_image, (new_width, new_height)
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")

# Histogram
with st.expander("Histogram"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Show Histogram", use_container_width=True):
            if st.session_state.original_image:
                hist_data = show_histogram(st.session_state.original_image)
                if hist_data:
                    st.info("Histogram data calculated. Visualization coming soon!")
                else:
                    st.warning("Could not calculate histogram")
            else:
                st.warning("Please upload an image first")
    
    with col2:
        if st.button("Equalization", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = histogram_equalization(
                    st.session_state.original_image
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")

# Filters
with st.expander("Filters"):
    st.write("**Low-Pass Filters**")
    col1, col2 = st.columns(2)
    
    sigma = st.slider("Gaussian Sigma", 0.1, 5.0, 1.0, 0.1, key="gaussian_sigma")
    median_size = st.slider("Median Filter Size", 3, 15, 3, 2, key="median_size")
    
    with col1:
        if st.button("Gaussian", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = gaussian_filter_func(
                    st.session_state.original_image, sigma
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col2:
        if st.button("Median", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = median_filter_func(
                    st.session_state.original_image, median_size
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    st.write("**High-Pass Filters**")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("Laplacian", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = laplacian_filter(
                    st.session_state.original_image
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col4:
        if st.button("Sobel", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = sobel_filter(
                    st.session_state.original_image
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col5:
        if st.button("Gradient", use_container_width=True):
            if st.session_state.original_image:
                st.session_state.processed_image = gradient_filter(
                    st.session_state.original_image
                )
                st.rerun()
            else:
                st.warning("Please upload an image first")

# Footer
render_footer()


if __name__ == "__main__":
    pass
