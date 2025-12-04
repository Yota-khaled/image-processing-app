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
    translate, scale, rotate, shear_x, shear_y,
    nearest_neighbor_interpolation, bilinear_interpolation, bicubic_interpolation,
    show_histogram, histogram_equalization,
    gaussian_filter_func, median_filter_func,
    laplacian_filter, sobel_filter, gradient_filter
)

# Page config
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)

apply_custom_css()
initialize_session_state()

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
                    buf = io.BytesIO()
                    st.session_state.processed_image.save(buf, format='PNG')
                    st.download_button(
                        label="Download Image",
                        data=buf.getvalue(),
                        file_name="processed_image.png",
                        mime="image/png",
                        key="download_processed"
                    )
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
        st.image(st.session_state.original_image, use_container_width=True)
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
            st.image(st.session_state.processed_image, use_container_width=True)
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
            st.image(orig, use_container_width=True)
        else:
            st.info("No original")
    with comp_col2:
        st.markdown("**Grayscale**")
        if gray is not None:
            st.image(gray, use_container_width=True)
        else:
            st.info("Grayscale not computed (click Grayscale)")
    with comp_col3:
        st.markdown("**Binary**")
        if bin_img is not None:
            st.image(bin_img, use_container_width=True)
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
                    st.session_state.processed_image = translate(st.session_state.original_image, tx, ty)
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
                    st.session_state.processed_image = scale(st.session_state.original_image, scale_x, scale_y)
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
                    st.session_state.processed_image = rotate(st.session_state.original_image, angle)
                    st.success("Rotate applied.")
                except Exception as e:
                    st.error("Rotate failed:")
                    st.exception(e)
            else:
                st.warning("Please upload an image first")

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
    if c2.button("Bilinear", key="bilinear_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = bilinear_interpolation(st.session_state.original_image, (new_width, new_height))
                st.success("Bilinear applied.")
            except Exception as e:
                st.error("Bilinear failed:")
                st.exception(e)
    if c3.button("Bicubic", key="bicubic_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = bicubic_interpolation(st.session_state.original_image, (new_width, new_height))
                st.success("Bicubic applied.")
            except Exception as e:
                st.error("Bicubic failed:")
                st.exception(e)

with st.expander("Histogram"):
    c1, c2 = st.columns(2)
    if c1.button("Show Histogram", key="hist_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                hist_data = show_histogram(st.session_state.original_image)
                if hist_data:
                    st.info("Histogram calculated.")
                else:
                    st.warning("Could not calculate histogram")
            except Exception as e:
                st.error("Histogram failed:")
                st.exception(e)
    if c2.button("Equalization", key="equalize_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = histogram_equalization(st.session_state.original_image)
                st.success("Equalization applied.")
            except Exception as e:
                st.error("Equalization failed:")
                st.exception(e)

with st.expander("Filters"):
    st.write("**Low-Pass Filters**")
    c1, c2 = st.columns(2)
    sigma = st.slider("Gaussian Sigma", 0.1, 5.0, 1.0, 0.1, key="gaussian_sigma")
    median_size = st.slider("Median Filter Size", 3, 15, 3, 2, key="median_size")
    if c1.button("Gaussian", key="gaussian_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = gaussian_filter_func(st.session_state.original_image, sigma)
                st.success("Gaussian applied.")
            except Exception as e:
                st.error("Gaussian failed:")
                st.exception(e)
    if c2.button("Median", key="median_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = median_filter_func(st.session_state.original_image, median_size)
                st.success("Median applied.")
            except Exception as e:
                st.error("Median failed:")
                st.exception(e)

    st.write("**High-Pass Filters**")
    c3, c4, c5 = st.columns(3)
    if c3.button("Laplacian", key="laplacian_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = laplacian_filter(st.session_state.original_image)
                st.success("Laplacian applied.")
            except Exception as e:
                st.error("Laplacian failed:")
                st.exception(e)
    if c4.button("Sobel", key="sobel_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = sobel_filter(st.session_state.original_image)
                st.success("Sobel applied.")
            except Exception as e:
                st.error("Sobel failed:")
                st.exception(e)
    if c5.button("Gradient", key="gradient_btn"):
        if st.session_state.get("original_image") is not None:
            try:
                st.session_state.processed_image = gradient_filter(st.session_state.original_image)
                st.success("Gradient applied.")
            except Exception as e:
                st.error("Gradient failed:")
                st.exception(e)

# Footer
render_footer()

if __name__ == "__main__":
    pass
