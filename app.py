"""
Main Streamlit application for Image Processing
Entry point - paste over your existing app.py
"""
# cd /home/aya/aya_backup/FCAI_4/ImageProcessing/Project/image-processing-app 
# # source ../.venv/bin/activate 
# # python -m streamlit run app.py

import streamlit as st
import io
import numpy as np
from PIL import Image

# Local modules
from config import APP_CONFIG, COLORS
from ui_components import apply_custom_css, render_footer, initialize_session_state
from utils import load_image, display_image_info
from image_processing import (
    to_grayscale, to_binary, to_binary_info,
    apply_affine, translate, scale, rotate, shear_x, shear_y,
    nearest_neighbor_interpolation, bilinear_interpolation, bicubic_interpolation,
    show_histogram, histogram_equalization,
    gaussian_filter_func, median_filter_func,
    laplacian_filter, sobel_filter, gradient_filter,
    crop_image, zoom_image, flip_image, adjust_brightness, adjust_contrast
)
import compression
import transform_coding
import matplotlib.pyplot as plt
import pandas as pd

# Page config
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)

apply_custom_css()
initialize_session_state()

# Helper to display images across Streamlit versions that may
# accept either `use_container_width` or the older `use_column_width`.
def safe_st_image(obj, use_container_width=True, **kwargs):
    try:
        return st.image(obj, use_container_width=use_container_width, **kwargs)
    except TypeError:
        try:
            return st.image(obj, use_column_width=use_container_width, **kwargs)
        except TypeError:
            # Last resort: call without width parameter
            return st.image(obj, **kwargs)

st.title("Image Processing Application")
st.markdown("---")

# ---------------- Sidebar - upload + actions ----------------
with st.sidebar:
    st.header("Image Controls")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image to process"
    )
    if uploaded_file is not None:
        try:
            st.session_state.original_image = load_image(uploaded_file)
            # reset processed image when new file uploaded
            st.session_state.processed_image = None
            # clear member2 cached results so user must compute
            for k in ("member2_gray", "member2_binary", "member2_bin_thresh", "member2_bin_method", "member2_bin_reason"):
                if k in st.session_state:
                    del st.session_state[k]
        except Exception as e:
            st.error("Failed to load image:")
            st.exception(e)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save", use_container_width=True, key="save_btn"):
            if st.session_state.get("processed_image") is not None:
                try:
                    # Preserve the processed_image in session state
                    saved_image = st.session_state.processed_image.copy()
                    buf = io.BytesIO()
                    saved_image.save(buf, format='PNG')
                    st.download_button(
                        label="Download Image",
                        data=buf.getvalue(),
                        file_name="processed_image.png",
                        mime="image/png",
                        key="download_processed"
                    )
                    # Ensure processed_image is still set after download button
                    st.session_state.processed_image = saved_image
                except Exception as e:
                    st.error("Failed to prepare download:")
                    st.exception(e)
            else:
                st.warning("No processed image to save")
    with c2:
        if st.button("Reset", use_container_width=True, key="reset_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    st.session_state.processed_image = st.session_state.original_image.copy()
                    # clear member2 caches as well
                    for k in ("member2_gray", "member2_binary"):
                        if k in st.session_state:
                            del st.session_state[k]
                    st.success("Reset processed image to original")
                except Exception as e:
                    st.error("Reset failed:")
                    st.exception(e)
            else:
                st.warning("No image to reset")

# ---------------- Main panels ----------------
left_col, right_col = st.columns(2)
with left_col:
    st.subheader("Original Image")
    if st.session_state.get("original_image") is not None:
        safe_st_image(st.session_state.original_image, use_container_width=True)
        with st.expander("Image Information"):
            try:
                st.markdown(display_image_info(st.session_state.original_image))
            except Exception as e:
                st.warning("Could not display image info.")
                st.exception(e)
    else:
        st.info("Upload an image from the sidebar to get started")

with right_col:
    st.subheader("Processed Image")
    if st.session_state.get("processed_image") is not None:
        try:
            safe_st_image(st.session_state.processed_image, use_container_width=True)
        except Exception as e:
            st.error("Failed to render processed image:")
            st.exception(e)
            st.write("Processed object type:", type(st.session_state.processed_image))
    elif st.session_state.get("original_image") is not None:
        st.info("Select an operation below to process the image")
    else:
        st.info("Result will appear here after processing")

st.markdown("---")
st.header("Processing Tools")

# ---------------- Basic Operations ----------------
with st.expander("Basic Operations", expanded=True):
    # initialize manual binary toggles
    if 'binary_selected' not in st.session_state:
        st.session_state.binary_selected = False
    if 'binary_threshold' not in st.session_state:
        st.session_state.binary_threshold = 128

    bcol1, bcol2, bcol3 = st.columns(3)

    # --- Grayscale button ---
    with bcol1:
        if st.button("Grayscale", key="grayscale_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    gray_res = to_grayscale(st.session_state.original_image)
                    # store both for comparison and processed panel
                    st.session_state['member2_gray'] = gray_res
                    st.session_state.processed_image = gray_res
                    st.success("Grayscale applied.")
                except Exception as e:
                    st.error("Grayscale failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")

    # --- Binary (auto) button ---
    with bcol2:
        if st.button("Binary (auto)", key="binary_auto_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    try:
                        bin_img, thresh, method_name, reason = to_binary_info(
                            st.session_state.original_image, method="auto"
                        )
                    except TypeError:
                        bin_img = to_binary(st.session_state.original_image)
                        thresh = None
                        method_name = "auto"
                        reason = ""
                    st.session_state['member2_binary'] = bin_img
                    st.session_state['member2_bin_thresh'] = thresh
                    st.session_state['member2_bin_method'] = method_name
                    st.session_state['member2_bin_reason'] = reason
                    st.session_state.processed_image = bin_img
                    st.success(f"Binary applied (auto). Method: {method_name.upper()}" + (f" — Threshold: {float(thresh):.2f}" if thresh is not None else ""))
                except Exception as e:
                    st.error("Auto binary conversion failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")

    # --- Binary manual toggle ---
    with bcol3:
        if st.button("Binary", key="binary_toggle_btn"):
            if st.session_state.get("original_image") is not None:
                st.session_state.binary_selected = not st.session_state.binary_selected
                if st.session_state.binary_selected:
                    st.info("Manual binary enabled — adjust slider and click Apply Binary.")
                else:
                    st.info("Manual binary disabled.")
            else:
                st.warning("Please upload an image first")

    # Manual slider & Apply
    if st.session_state.binary_selected:
        st.markdown("---")
        thr = st.slider("Binary Threshold", 0, 255, st.session_state.get('binary_threshold', 128), key="binary_threshold_slider")
        st.session_state.binary_threshold = int(thr)
        if st.button("Apply Binary", key="apply_binary_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    gray = st.session_state.original_image.convert("L")
                    arr = np.array(gray, dtype=np.uint8)
                    bin_arr = (arr >= st.session_state.binary_threshold).astype(np.uint8) * 255
                    bin_img = Image.fromarray(bin_arr, mode="L").convert("RGB")
                    st.session_state['member2_binary'] = bin_img
                    st.session_state['member2_bin_thresh'] = st.session_state.binary_threshold
                    st.session_state['member2_bin_method'] = "manual"
                    st.session_state['member2_bin_reason'] = "manual threshold"
                    st.session_state.processed_image = bin_img
                    st.success(f"Manual binary applied. Threshold = {st.session_state.binary_threshold}")
                except Exception as e:
                    st.error("Manual binary conversion failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")

    # ----------------- Comparison panel (original / grayscale / binary) -----------------
    st.markdown("---")
    st.subheader("Original — Grayscale — Binary")
    # get stored images (if exist)
    orig = st.session_state.get("original_image", None)
    gray = st.session_state.get("member2_gray", None)
    bin_img = st.session_state.get("member2_binary", None)

    # Show three columns side-by-side
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    with comp_col1:
        st.markdown("**Original**")
        if orig is not None:
            safe_st_image(orig, use_container_width=True)
        else:
            st.info("No original")
    with comp_col2:
        st.markdown("**Grayscale**")
        if gray is not None:
            safe_st_image(gray, use_container_width=True)
        else:
            st.info("Grayscale not computed (click Grayscale)")
    with comp_col3:
        st.markdown("**Binary**")
        if bin_img is not None:
            safe_st_image(bin_img, use_container_width=True)
            # show method & threshold if present
            method = st.session_state.get("member2_bin_method", "unknown")
            thresh = st.session_state.get("member2_bin_thresh", None)
            caption = f"Method: {method.upper()}"
            if thresh is not None:
                caption += f" — Threshold: {float(thresh):.2f}"
            st.caption(caption)
        else:
            st.info("Binary not computed (click Binary (auto) or Apply Binary)")

# ---------------- Affine, Interpolation, Histogram, Filters (unchanged) ----------------
with st.expander("Affine Transformations"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Translate**")
        tx = st.slider("X", -100, 100, 0, key="tx")
        ty = st.slider("Y", -100, 100, 0, key="ty")
        if st.button("Translate", key="translate_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    st.session_state.processed_image = apply_affine(st.session_state.original_image, 'translate', tx=tx, ty=ty)
                    st.success("Translate applied.")
                except Exception as e:
                    st.error("Translate failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")
    with c2:
        st.write("**Scale**")
        scale_x = st.slider("Scale X", 0.1, 3.0, 1.0, 0.1, key="scale_x")
        scale_y = st.slider("Scale Y", 0.1, 3.0, 1.0, 0.1, key="scale_y")
        if st.button("Scale", key="scale_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    st.session_state.processed_image = apply_affine(st.session_state.original_image, 'scale', sx=scale_x, sy=scale_y)
                    st.success("Scale applied.")
                except Exception as e:
                    st.error("Scale failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")
    with c3:
        st.write("**Rotate**")
        angle = st.slider("Angle (degrees)", -180, 180, 0, key="angle")
        if st.button("Rotate", key="rotate_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    st.session_state.processed_image = apply_affine(st.session_state.original_image, 'rotate', angle=angle)
                    st.success("Rotate applied.")
                except Exception as e:
                    st.error("Rotate failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")

    # --- Shear (X & Y) ---
    st.markdown("---")
    s1, s2 = st.columns(2)
    with s1:
        st.write("**Shear X (horizontal)**")
        shear_x_factor = st.slider("Shear X factor", -1.0, 1.0, 0.0, 0.01, key="shear_x_factor")
        if st.button("Apply X-direction Shear", key="shearx_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    st.session_state.processed_image = apply_affine(st.session_state.original_image, 'shear_x', factor=shear_x_factor)
                    st.success("X-direction shear applied.")
                except Exception as e:
                    st.error("X-direction shear failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")
    with s2:
        st.write("**Shear Y (vertical)**")
        shear_y_factor = st.slider("Shear Y factor", -1.0, 1.0, 0.0, 0.01, key="shear_y_factor")
        if st.button("Apply Y-direction Shear", key="sheary_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    st.session_state.processed_image = apply_affine(st.session_state.original_image, 'shear_y', factor=shear_y_factor)
                    print("st.session_state.processed_image: ", st.session_state.processed_image)
                    st.success("Y-direction shear applied.")
                except Exception as e:
                    st.error("Y-direction shear failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")

    # --- Multi-transformation queue ---
    st.markdown("---")
    st.subheader("Multiple Transformations")

    # Initialize session state for queued operations
    if 'queued_ops' not in st.session_state:
        st.session_state.queued_ops = []

    # Create checkboxes to queue operations
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    with qc1:
        queue_translate = st.checkbox("Translate", value=False, key="queue_translate")
    with qc2:
        queue_scale = st.checkbox("Scale", value=False, key="queue_scale")
    with qc3:
        queue_rotate = st.checkbox("Rotate", value=False, key="queue_rotate")
    with qc4:
        queue_shear_x = st.checkbox("Shear X", value=False, key="queue_shear_x")
    with qc5:
        queue_shear_y = st.checkbox("Shear Y", value=False, key="queue_shear_y")

    # Build queue based on selected checkboxes
    current_queue = []
    if queue_translate:
        current_queue.append(('translate', {'tx': tx, 'ty': ty}))
    if queue_scale:
        current_queue.append(('scale', {'sx': scale_x, 'sy': scale_y}))
    if queue_rotate:
        current_queue.append(('rotate', {'angle': angle}))
    if queue_shear_x:
        current_queue.append(('shear_x', {'factor': shear_x_factor}))
    if queue_shear_y:
        current_queue.append(('shear_y', {'factor': shear_y_factor}))

    # Display current queue
    if current_queue:
        st.info(f"Queue: {' → '.join([op[0] for op in current_queue])}")

    # Apply All Transformations button
    apply_all_col = st.columns(1)[0]
    if apply_all_col.button("Apply All Transformations", key="apply_all_transforms", use_container_width=True):
        if st.session_state.get("original_image") is not None:
            if current_queue:
                try:
                    result_image = st.session_state.original_image
                    for op_name, op_params in current_queue:
                        result_image = apply_affine(result_image, op_name, **op_params)
                    st.session_state.processed_image = result_image
                    st.success(f"Applied {len(current_queue)} transformation(s) in sequence!")
                except Exception as e:
                    st.error("Multi-transformation failed:")
                    st.exception(e)
            else:
                st.warning("Please select at least one transformation to apply.")
        else:
            st.warning("Please upload an image first")

    # Preview area inside the transformations expander: show original vs transformed
    st.markdown("---")
    st.subheader("Transform Preview")
    pv1, pv2 = st.columns(2)
    with pv1:
        st.markdown("**Original**")
        if st.session_state.get("original_image") is not None:
            safe_st_image(st.session_state.original_image, use_container_width=True)
        else:
            st.info("No original image")
    with pv2:
        st.markdown("**Transformed**")
        if st.session_state.get("processed_image") is not None:
            try:
                safe_st_image(st.session_state.processed_image, use_container_width=True)
            except Exception as e:
                st.error("Could not show transformed image:")
                st.exception(e)
        else:
            st.info("No transformed image — apply an operation above")

with st.expander("Image Interpolation"):
    c1, c2, c3 = st.columns(3)
    new_width = st.slider("New Width", 100, 2000, 800, 50, key="interp_width")
    new_height = st.slider("New Height", 100, 2000, 600, 50, key="interp_height")
    if c1.button("Nearest Neighbor", key="nn_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = nearest_neighbor_interpolation(st.session_state.original_image, (new_width, new_height))
                st.success("Nearest neighbor applied.")
            except Exception as e:
                st.error("Nearest neighbor failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    if c2.button("Bilinear", key="bilinear_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = bilinear_interpolation(st.session_state.original_image, (new_width, new_height))
                st.success("Bilinear applied.")
            except Exception as e:
                st.error("Bilinear failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    if c3.button("Bicubic", key="bicubic_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = bicubic_interpolation(st.session_state.original_image, (new_width, new_height))
                st.success("Bicubic applied.")
            except Exception as e:
                st.error("Bicubic failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    
    # Preview area for interpolation
    st.markdown("---")
    st.subheader("Interpolation Preview")
    interp_pv1, interp_pv2 = st.columns(2)
    with interp_pv1:
        st.markdown("**Original**")
        if st.session_state.get("original_image") is not None:
            safe_st_image(st.session_state.original_image, use_container_width=True)
        else:
            st.info("No original image")
    with interp_pv2:
        st.markdown("**Interpolated**")
        if st.session_state.get("processed_image") is not None:
            try:
                safe_st_image(st.session_state.processed_image, use_container_width=True)
            except Exception as e:
                st.error("Could not show interpolated image:")
                st.exception(e)
        else:
            st.info("No interpolated image — apply an interpolation method above")

with st.expander("Histogram"):
    c1, c2 = st.columns(2)
    if c1.button("Show Histogram", key="hist_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                hist_data = show_histogram(st.session_state.original_image)
                if hist_data:
                    st.success("Histogram calculated. See visualization below.")
                    # Display histogram visualization
                    fig, ax = plt.subplots()
                    for color, values in hist_data.items():
                        ax.plot(values, label=color, alpha=0.7)
                    ax.set_xlabel('Pixel Intensity')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Image Histogram')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.warning("Could not calculate histogram")
            except Exception as e:
                st.error("Histogram failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    if c2.button("Equalization", key="equalize_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                result = histogram_equalization(st.session_state.original_image)
                if result is not None:
                    st.session_state.processed_image = result
                    st.success("Equalization applied.")
                else:
                    st.error("Equalization returned None. Please check the image.")
            except Exception as e:
                st.error("Equalization failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    
    # Preview area for histogram operations
    st.markdown("---")
    st.subheader("Histogram Preview")
    hist_pv1, hist_pv2 = st.columns(2)
    with hist_pv1:
        st.markdown("**Original**")
        if st.session_state.get("original_image") is not None:
            safe_st_image(st.session_state.original_image, use_container_width=True)
        else:
            st.info("No original image")
    with hist_pv2:
        st.markdown("**Processed**")
        if st.session_state.get("processed_image") is not None:
            try:
                safe_st_image(st.session_state.processed_image, use_container_width=True)
            except Exception as e:
                st.error("Could not show processed image:")
                st.exception(e)
        else:
            st.info("No processed image — apply an operation above")

with st.expander("Filters"):
    st.write("**Low-Pass Filters**")
    c1, c2 = st.columns(2)
    sigma = st.slider("Gaussian Sigma", 0.1, 5.0, 1.0, 0.1, key="gaussian_sigma")
    median_size = st.slider("Median Filter Size", 3, 15, 3, 2, key="median_size")
    if c1.button("Gaussian", key="gaussian_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                result = gaussian_filter_func(st.session_state.original_image, sigma)
                if result is not None:
                    st.session_state.processed_image = result
                    st.success("Gaussian applied.")
                else:
                    st.error("Gaussian filter returned None. Please check the image.")
            except Exception as e:
                st.error("Gaussian failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    if c2.button("Median", key="median_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                result = median_filter_func(st.session_state.original_image, median_size)
                if result is not None:
                    st.session_state.processed_image = result
                    st.success("Median applied.")
                else:
                    st.error("Median filter returned None. Please check the image.")
            except Exception as e:
                st.error("Median failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")

    st.write("**High-Pass Filters**")
    c3, c4, c5 = st.columns(3)
    if c3.button("Laplacian", key="laplacian_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                result = laplacian_filter(st.session_state.original_image)
                if result is not None:
                    st.session_state.processed_image = result
                    st.success("Laplacian applied.")
                else:
                    st.error("Laplacian filter returned None. Please check the image.")
            except Exception as e:
                st.error("Laplacian failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    if c4.button("Sobel", key="sobel_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                result = sobel_filter(st.session_state.original_image)
                if result is not None:
                    st.session_state.processed_image = result
                    st.success("Sobel applied.")
                else:
                    st.error("Sobel filter returned None. Please check the image.")
            except Exception as e:
                st.error("Sobel failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    if c5.button("Gradient", key="gradient_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                result = gradient_filter(st.session_state.original_image)
                if result is not None:
                    st.session_state.processed_image = result
                    st.success("Gradient applied.")
                else:
                    st.error("Gradient filter returned None. Please check the image.")
            except Exception as e:
                st.error("Gradient failed:")
                st.exception(e)
        else:
            st.warning("Please upload an image first")
    
    # Preview area for filters
    st.markdown("---")
    st.subheader("Filter Preview")
    filter_pv1, filter_pv2 = st.columns(2)
    with filter_pv1:
        st.markdown("**Original**")
        if st.session_state.get("original_image") is not None:
            safe_st_image(st.session_state.original_image, use_container_width=True)
        else:
            st.info("No original image")
    with filter_pv2:
        st.markdown("**Filtered**")
        if st.session_state.get("processed_image") is not None:
            try:
                safe_st_image(st.session_state.processed_image, use_container_width=True)
            except Exception as e:
                st.error("Could not show filtered image:")
                st.exception(e)
        else:
            st.info("No filtered image — apply a filter above")

# ---------------- Member 5: Image Operations (Crop + Extra Features) ----------------
with st.expander("Member 5 — Image Operations (Crop + Extra Features)", expanded=False):
    st.subheader("Crop Image")
    crop_col1, crop_col2 = st.columns(2)
    
    if st.session_state.get("original_image") is not None:
        img_width, img_height = st.session_state.original_image.size
        
        with crop_col1:
            left = st.number_input("Left", 0, img_width, 0, key="crop_left")
            top = st.number_input("Top", 0, img_height, 0, key="crop_top")
        
        with crop_col2:
            right = st.number_input("Right", 0, img_width, img_width, key="crop_right")
            bottom = st.number_input("Bottom", 0, img_height, img_height, key="crop_bottom")
        
        if st.button("Apply Crop", key="crop_btn"):
            try:
                if left < right and top < bottom:
                    result_img = crop_image(
                        st.session_state.original_image, left, top, right, bottom
                    )
                    if result_img is not None:
                        st.session_state.processed_image = result_img
                        st.success("Crop applied.")
                    else:
                        st.error("Crop returned None. Please check the coordinates.")
                else:
                    st.error("Invalid crop coordinates. Left < Right and Top < Bottom required.")
            except Exception as e:
                st.error("Crop failed:")
                st.exception(e)
    
    # Crop preview section
    if st.session_state.get("original_image") is not None:
        st.markdown("---")
        st.subheader("Crop Preview")
        crop_pv1, crop_pv2 = st.columns(2)
        with crop_pv1:
            st.markdown("**Original**")
            safe_st_image(st.session_state.original_image, use_container_width=True)
        with crop_pv2:
            st.markdown("**Cropped**")
            if st.session_state.get("processed_image") is not None:
                try:
                    safe_st_image(st.session_state.processed_image, use_container_width=True)
                except Exception as e:
                    st.error("Could not show cropped image:")
                    st.exception(e)
            else:
                st.info("No cropped image — apply crop above")
    
    st.markdown("---")
    st.subheader("Zoom In/Out")
    zoom_col1, zoom_col2 = st.columns(2)
    
    with zoom_col1:
        zoom_factor = st.slider("Zoom Factor", 0.1, 5.0, 1.0, 0.1, key="zoom_factor")
        if st.button("Apply Zoom", key="zoom_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    result_img = zoom_image(
                        st.session_state.original_image, zoom_factor
                    )
                    if result_img is not None:
                        st.session_state.processed_image = result_img
                        st.success(f"Zoom applied (factor: {zoom_factor:.2f}).")
                    else:
                        st.error("Zoom returned None.")
                except Exception as e:
                    st.error("Zoom failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")
    
    with zoom_col2:
        st.write("**Quick Zoom Buttons**")
        qz_col1, qz_col2 = st.columns(2)
        with qz_col1:
            if st.button("Zoom In (2x)", key="zoom_in_btn"):
                if st.session_state.get("original_image") is not None:
                    try:
                        result_img = zoom_image(
                            st.session_state.original_image, 2.0
                        )
                        if result_img is not None:
                            st.session_state.processed_image = result_img
                            st.success("Zoomed in 2x.")
                        else:
                            st.error("Zoom returned None.")
                    except Exception as e:
                        st.error("Zoom failed:")
                        st.exception(e)
                else:
                    st.warning("Please upload an image first")
        with qz_col2:
            if st.button("Zoom Out (0.5x)", key="zoom_out_btn"):
                if st.session_state.get("original_image") is not None:
                    try:
                        result_img = zoom_image(
                            st.session_state.original_image, 0.5
                        )
                        if result_img is not None:
                            st.session_state.processed_image = result_img
                            st.success("Zoomed out 0.5x.")
                        else:
                            st.error("Zoom returned None.")
                    except Exception as e:
                        st.error("Zoom failed:")
                        st.exception(e)
                else:
                    st.warning("Please upload an image first")
    
    st.markdown("---")
    st.subheader("Flip Image")
    flip_col1, flip_col2 = st.columns(2)
    
    with flip_col1:
        if st.button("Flip Horizontal", key="flip_h_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    result_img = flip_image(
                        st.session_state.original_image, 'horizontal'
                    )
                    if result_img is not None:
                        st.session_state.processed_image = result_img
                        st.success("Flipped horizontally.")
                    else:
                        st.error("Flip returned None.")
                except Exception as e:
                    st.error("Flip failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")
    
    with flip_col2:
        if st.button("Flip Vertical", key="flip_v_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    result_img = flip_image(
                        st.session_state.original_image, 'vertical'
                    )
                    if result_img is not None:
                        st.session_state.processed_image = result_img
                        st.success("Flipped vertically.")
                    else:
                        st.error("Flip returned None.")
                except Exception as e:
                    st.error("Flip failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")
    
    st.markdown("---")
    st.subheader("Brightness & Contrast")
    bc_col1, bc_col2 = st.columns(2)
    
    with bc_col1:
        brightness = st.slider("Brightness", -100, 100, 0, key="brightness_slider")
        if st.button("Apply Brightness", key="brightness_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    result_img = adjust_brightness(
                        st.session_state.original_image, brightness
                    )
                    if result_img is not None:
                        st.session_state.processed_image = result_img
                        st.success(f"Brightness adjusted ({brightness}).")
                    else:
                        st.error("Brightness adjustment returned None.")
                except Exception as e:
                    st.error("Brightness adjustment failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")
    
    with bc_col2:
        contrast = st.slider("Contrast", -100, 100, 0, key="contrast_slider")
        if st.button("Apply Contrast", key="contrast_btn"):
            if st.session_state.get("original_image") is not None:
                try:
                    result_img = adjust_contrast(
                        st.session_state.original_image, contrast
                    )
                    if result_img is not None:
                        st.session_state.processed_image = result_img
                        st.success(f"Contrast adjusted ({contrast}).")
                    else:
                        st.error("Contrast adjustment returned None.")
                except Exception as e:
                    st.error("Contrast adjustment failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")

# ---------------- Member 9: Lossless Compression Algorithms ----------------
with st.expander("Member 9 — Lossless Compression Algorithms", expanded=False):
    st.subheader("Compression Algorithms")
    
    if st.session_state.get("original_image") is not None:
        # Convert image to data for compression
        img_array = np.array(st.session_state.original_image.convert("L"))
        img_data = img_array.flatten().tolist()
        img_data_str = ''.join([chr(int(p)) for p in img_data])
        
        st.write("**Select compression algorithms to test:**")
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            test_huffman = st.checkbox("Huffman", value=True, key="test_huffman")
            test_golomb = st.checkbox("Golomb-Rice", value=True, key="test_golomb")
        
        with comp_col2:
            test_arithmetic = st.checkbox("Arithmetic", value=True, key="test_arithmetic")
            test_lzw = st.checkbox("LZW", value=True, key="test_lzw")
        
        with comp_col3:
            test_rle = st.checkbox("RLE", value=True, key="test_rle")
        
        golomb_m = st.slider("Golomb-Rice parameter (m, must be power of 2)", 2, 16, 4, key="golomb_m")
        
        if st.button("Run Compression Tests", key="compression_test_btn"):
            results = []
            
            if test_huffman:
                try:
                    result = compression.compress_and_report(
                        img_data_str, "Huffman",
                        compression.huffman_encode,
                        compression.huffman_decode
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    st.warning(f"Huffman failed: {str(e)}")
            
            if test_golomb:
                try:
                    result = compression.compress_and_report(
                        img_data, "Golomb-Rice",
                        compression.golomb_rice_encode,
                        compression.golomb_rice_decode,
                        golomb_m
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    st.warning(f"Golomb-Rice failed: {str(e)}")
            
            if test_arithmetic:
                try:
                    result = compression.compress_and_report(
                        img_data_str, "Arithmetic",
                        compression.arithmetic_encode,
                        compression.arithmetic_decode
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    st.warning(f"Arithmetic failed: {str(e)}")
            
            if test_lzw:
                try:
                    result = compression.compress_and_report(
                        img_data_str, "LZW",
                        compression.lzw_encode,
                        compression.lzw_decode
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    st.warning(f"LZW failed: {str(e)}")
            
            if test_rle:
                try:
                    result = compression.compress_and_report(
                        img_data_str, "RLE",
                        compression.rle_encode,
                        compression.rle_decode
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    st.warning(f"RLE failed: {str(e)}")
            
            if results:
                st.session_state.compression_results = results
                st.success(f"Compression tests completed for {len(results)} algorithm(s).")
            else:
                st.error("All compression tests failed.")
        
        # Display results
        if 'compression_results' in st.session_state:
            st.markdown("---")
            st.subheader("Compression Report")
            
            results_df = pd.DataFrame(st.session_state.compression_results)
            
            # Display table
            st.dataframe(results_df[['algorithm', 'original_size', 'compressed_size', 
                                     'compression_ratio', 'encode_time', 'decode_time']], 
                        use_container_width=True)
            
            # Display charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.write("**Compression Ratio**")
                fig1, ax1 = plt.subplots()
                ax1.bar(results_df['algorithm'], results_df['compression_ratio'])
                ax1.set_ylabel('Compression Ratio')
                ax1.set_title('Compression Ratio by Algorithm')
                plt.xticks(rotation=45)
                st.pyplot(fig1)
            
            with chart_col2:
                st.write("**Encoding Time**")
                fig2, ax2 = plt.subplots()
                ax2.bar(results_df['algorithm'], results_df['encode_time'])
                ax2.set_ylabel('Time (seconds)')
                ax2.set_title('Encoding Time by Algorithm')
                plt.xticks(rotation=45)
                st.pyplot(fig2)
            
            # Combined comparison chart
            st.write("**Size Comparison**")
            fig3, ax3 = plt.subplots()
            x = np.arange(len(results_df))
            width = 0.35
            ax3.bar(x - width/2, results_df['original_size'], width, label='Original')
            ax3.bar(x + width/2, results_df['compressed_size'], width, label='Compressed')
            ax3.set_ylabel('Size (bytes)')
            ax3.set_title('Original vs Compressed Size')
            ax3.set_xticks(x)
            ax3.set_xticklabels(results_df['algorithm'], rotation=45)
            ax3.legend()
            st.pyplot(fig3)
    else:
        st.info("Please upload an image to test compression algorithms.")

# ---------------- Member 10: Transform Coding + Predictive + Wavelet ----------------
with st.expander("Member 10 — Transform Coding + Predictive + Wavelet", expanded=False):
    st.subheader("Transform Coding Techniques")
    
    if st.session_state.get("original_image") is not None:
        img_array = np.array(st.session_state.original_image.convert("L"))
        
        transform_tabs = st.tabs(["Symbol-based", "Bit-plane", "DCT Block", "Predictive", "Wavelet"])
        
        with transform_tabs[0]:
            st.write("**Symbol-based Coding**")
            if st.button("Apply Symbol-based Coding", key="symbol_btn"):
                try:
                    data = img_array.flatten().tolist()
                    encoded = transform_coding.symbol_based_encode(data)
                    decoded = transform_coding.symbol_based_decode(encoded)
                    reconstructed = np.array(decoded).reshape(img_array.shape).astype(np.uint8)
                    result_img = Image.fromarray(reconstructed)
                    st.session_state.processed_image = result_img
                    
                    st.success("Symbol-based coding applied. See result in 'Processed Image' section above.")
                    
                    # Calculate metrics - show below
                    st.markdown("**Compression Metrics:**")
                    original_size = len(data)
                    compressed_size = len(encoded) * 2  # Approximate
                    metrics = transform_coding.calculate_compression_metrics(
                        data, compressed_size, original_size
                    )
                    st.json(metrics)
                except Exception as e:
                    st.error("Symbol-based coding failed:")
                    st.exception(e)
        
        with transform_tabs[1]:
            st.write("**Bit-plane Coding**")
            if st.button("Apply Bit-plane Coding", key="bitplane_btn"):
                try:
                    bit_planes, shape = transform_coding.bit_plane_encode(img_array)
                    reconstructed = transform_coding.bit_plane_decode(bit_planes, shape)
                    result_img = Image.fromarray(reconstructed)
                    st.session_state.processed_image = result_img
                    
                    st.success("Bit-plane coding applied. See result in 'Processed Image' section above.")
                    
                    # Calculate metrics - show below
                    st.markdown("**Compression Metrics:**")
                    original_size = img_array.nbytes
                    compressed_size = sum(bp.nbytes for bp in bit_planes)
                    metrics = transform_coding.calculate_compression_metrics(
                        img_array, compressed_size, original_size
                    )
                    st.json(metrics)
                except Exception as e:
                    st.error("Bit-plane coding failed:")
                    st.exception(e)
        
        with transform_tabs[2]:
            st.write("**Block Transform Coding (DCT)**")
            block_size = st.slider("Block Size", 4, 16, 8, 2, key="dct_block_size")
            if st.button("Apply DCT Block Transform", key="dct_btn"):
                try:
                    dct_blocks, orig_shape, pad_shape, bs = transform_coding.dct_block_encode(
                        img_array, block_size
                    )
                    reconstructed = transform_coding.dct_block_decode(
                        dct_blocks, orig_shape, pad_shape, bs
                    )
                    result_img = Image.fromarray(reconstructed)
                    st.session_state.processed_image = result_img
                    
                    st.success("DCT block transform applied. See result in 'Processed Image' section above.")
                    
                    # Calculate metrics - show below
                    st.markdown("**Compression Metrics:**")
                    original_size = img_array.nbytes
                    compressed_size = sum(sum(block.nbytes for block in channel) for channel in dct_blocks)
                    metrics = transform_coding.calculate_compression_metrics(
                        img_array, compressed_size, original_size
                    )
                    st.json(metrics)
                except Exception as e:
                    st.error("DCT block transform failed:")
                    st.exception(e)
        
        with transform_tabs[3]:
            st.write("**Predictive Coding**")
            predictor_type = st.selectbox("Predictor Type", ["previous", "average"], key="predictor_type")
            if st.button("Apply Predictive Coding", key="predictive_btn"):
                try:
                    errors = transform_coding.predictive_encode(img_array, predictor_type)
                    reconstructed = transform_coding.predictive_decode(errors, predictor_type)
                    result_img = Image.fromarray(reconstructed)
                    st.session_state.processed_image = result_img
                    
                    st.success("Predictive coding applied. See result in 'Processed Image' section above.")
                    
                    # Calculate metrics - show below
                    st.markdown("**Compression Metrics:**")
                    original_size = img_array.nbytes
                    compressed_size = errors.nbytes
                    metrics = transform_coding.calculate_compression_metrics(
                        img_array, compressed_size, original_size
                    )
                    st.json(metrics)
                except Exception as e:
                    st.error("Predictive coding failed:")
                    st.exception(e)
        
        with transform_tabs[4]:
            st.write("**Wavelet Coding**")
            wavelet_type = st.selectbox("Wavelet Type", ["haar", "db4", "bior2.2"], key="wavelet_type")
            wavelet_level = st.slider("Wavelet Level", 1, 5, 3, key="wavelet_level")
            if st.button("Apply Wavelet Transform", key="wavelet_btn"):
                try:
                    coeffs, orig_shape, wv, level = transform_coding.wavelet_encode(
                        img_array, wavelet_type, wavelet_level
                    )
                    reconstructed = transform_coding.wavelet_decode(
                        coeffs, orig_shape, wv, level
                    )
                    result_img = Image.fromarray(reconstructed)
                    st.session_state.processed_image = result_img
                    
                    st.success("Wavelet transform applied. See result in 'Processed Image' section above.")
                    
                    # Calculate metrics - show below
                    st.markdown("**Compression Metrics:**")
                    original_size = img_array.nbytes
                    if isinstance(coeffs, list):
                        compressed_size = sum(sum(c.nbytes if isinstance(c, np.ndarray) else 0 
                                                  for c in channel if isinstance(c, np.ndarray)) 
                                              for channel in coeffs)
                    else:
                        compressed_size = sum(c.nbytes if isinstance(c, np.ndarray) else 0 
                                            for c in coeffs if isinstance(c, np.ndarray))
                    metrics = transform_coding.calculate_compression_metrics(
                        img_array, compressed_size, original_size
                    )
                    st.json(metrics)
                except Exception as e:
                    st.error("Wavelet transform failed:")
                    st.exception(e)
        
        # Performance comparison
        if st.button("Compare All Transform Methods", key="compare_transforms_btn"):
            try:
                comparison_results = []
                
                # Symbol-based
                try:
                    data = img_array.flatten().tolist()
                    encoded = transform_coding.symbol_based_encode(data)
                    compressed_size = len(encoded) * 2
                    metrics = transform_coding.calculate_compression_metrics(
                        data, compressed_size, len(data)
                    )
                    comparison_results.append({
                        'method': 'Symbol-based',
                        'compression_ratio': metrics['compression_ratio'],
                        'space_saving': metrics['space_saving_percent']
                    })
                except:
                    pass
                
                # Bit-plane
                try:
                    bit_planes, shape = transform_coding.bit_plane_encode(img_array)
                    compressed_size = sum(bp.nbytes for bp in bit_planes)
                    metrics = transform_coding.calculate_compression_metrics(
                        img_array, compressed_size, img_array.nbytes
                    )
                    comparison_results.append({
                        'method': 'Bit-plane',
                        'compression_ratio': metrics['compression_ratio'],
                        'space_saving': metrics['space_saving_percent']
                    })
                except:
                    pass
                
                # DCT
                try:
                    dct_blocks, orig_shape, pad_shape, bs = transform_coding.dct_block_encode(img_array, 8)
                    compressed_size = sum(sum(block.nbytes for block in channel) for channel in dct_blocks)
                    metrics = transform_coding.calculate_compression_metrics(
                        img_array, compressed_size, img_array.nbytes
                    )
                    comparison_results.append({
                        'method': 'DCT Block',
                        'compression_ratio': metrics['compression_ratio'],
                        'space_saving': metrics['space_saving_percent']
                    })
                except:
                    pass
                
                # Predictive
                try:
                    errors = transform_coding.predictive_encode(img_array, 'previous')
                    compressed_size = errors.nbytes
                    metrics = transform_coding.calculate_compression_metrics(
                        img_array, compressed_size, img_array.nbytes
                    )
                    comparison_results.append({
                        'method': 'Predictive',
                        'compression_ratio': metrics['compression_ratio'],
                        'space_saving': metrics['space_saving_percent']
                    })
                except:
                    pass
                
                # Wavelet
                try:
                    coeffs, orig_shape, wv, level = transform_coding.wavelet_encode(img_array, 'haar', 3)
                    if isinstance(coeffs, list):
                        compressed_size = sum(sum(c.nbytes if isinstance(c, np.ndarray) else 0 
                                                  for c in channel if isinstance(c, np.ndarray)) 
                                              for channel in coeffs)
                    else:
                        compressed_size = sum(c.nbytes if isinstance(c, np.ndarray) else 0 
                                            for c in coeffs if isinstance(c, np.ndarray))
                    metrics = transform_coding.calculate_compression_metrics(
                        img_array, compressed_size, img_array.nbytes
                    )
                    comparison_results.append({
                        'method': 'Wavelet',
                        'compression_ratio': metrics['compression_ratio'],
                        'space_saving': metrics['space_saving_percent']
                    })
                except:
                    pass
                
                if comparison_results:
                    st.markdown("---")
                    st.subheader("Performance Comparison")
                    comp_df = pd.DataFrame(comparison_results)
                    st.dataframe(comp_df, use_container_width=True)
                    
                    # Visualization
                    fig, ax = plt.subplots()
                    x = np.arange(len(comp_df))
                    ax.bar(x, comp_df['space_saving'])
                    ax.set_ylabel('Space Saving (%)')
                    ax.set_title('Compression Performance Comparison')
                    ax.set_xticks(x)
                    ax.set_xticklabels(comp_df['method'], rotation=45)
                    st.pyplot(fig)
                    
                    st.session_state.transform_comparison = comparison_results
            except Exception as e:
                st.error("Comparison failed:")
                st.exception(e)
    else:
        st.info("Please upload an image to test transform coding techniques.")

# Footer
render_footer()

if __name__ == "__main__":
    pass
