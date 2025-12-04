"""
UI Components and styling for the Image Processing Application
Contains CSS styling and UI helper functions
"""

import streamlit as st
from config import COLORS


def apply_custom_css():
    st.markdown(f"""
        <style>
                
        [data-testid="stHeader"] {{
        background-color:  #00000 !important;
        }}
                
  
        /* ===== Background ===== */
        .stApp {{
            background-color: {COLORS['background']} !important;
        }}

        /* ===== Sidebar ===== */
        [data-testid="stSidebar"] {{
            background-color: {COLORS['sidebar']} !important;
        }}

        /* ===== Headers ===== */
        h1, h2, h3, h4 {{
            color: {COLORS['primary']} !important;
            font-weight: 700 !important;
        }}

        /* ===== Global Text ===== */
        p, div, label, span {{
            color: {COLORS['primary']} !important;
        }}

        /* ===== Muted Text ===== */
        .stCaption, .small {{
            color: {COLORS['text_muted']} !important;
        }}

        /* ===== Expander Header ===== */
        .streamlit-expanderHeader {{
            background-color: #1a2e3e !important;
            color: {COLORS['primary_light']} !important;
            border-left: 4px solid {COLORS['primary']} !important;
            border-radius: 8px !important;
            padding: 0.8rem !important;
            font-weight: 600 !important;
        }}

        .streamlit-expanderHeader:hover {{
            background-color: #2a3e4e !important;
        }}

        /* ===== Expander Content ===== */
        .streamlit-expanderContent {{
            background-color: {COLORS['card']} !important;
            border-radius: 0 0 8px 8px !important;
            padding: 1.5rem !important;
            border: 1px solid rgba(74, 112, 169, 0.12) !important;
            border-top: none !important;
        }}

        /* ===== PRIMARY BUTTONS ===== */
        .stButton > button {{
            background-color: {COLORS['primary']} !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            box-shadow: 0 2px 8px rgba(74, 112, 169, 0.2) !important;
            transition: all 0.3s ease !important;
            text-align: center !important;
        }}

        /* ===== FORCE WHITE TEXT INSIDE BUTTONS ===== */
        .stButton > button * {{
            color: #BED1E0 !important;
        }}

        .stButton > button:hover {{
            background-color: #EFECE3 !important;
            color: {COLORS['primary']} !important;
            box-shadow: 0 4px 12px rgba(74, 112, 169, 0.3) !important;
            transform: translateY(-2px) !important;
        }}

        .stButton > button:hover * {{
            color: {COLORS['primary']} !important;
        }}

        /* ===== Download Button ===== */
        .stDownloadButton > button {{
            background-color: {COLORS['primary']} !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
        }}

        .stDownloadButton > button * {{
            color: white !important;
        }}

        .stDownloadButton > button:hover {{
            background-color: #EFECE3 !important;
            color: {COLORS['primary']} !important;
        }}

        .stDownloadButton > button:hover * {{
            color: {COLORS['primary']} !important;
        }}


        /* ===== File Uploader ===== */
        .stFileUploader {{
            background-color: #BED1E0 !important;
            border: 2px dashed {COLORS['primary']} !important;
            border-radius: 12px !important;
            padding: 2rem !important;
        }}

        section[data-testid="stFileUploaderDropzone"] {{
            background-color: #BED1E0 !important;   /* ← your desired color */
            background: #BED1E0 !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.07) !important;
        }}

        /* Fallback for older Streamlit versions */
        div[data-testid="stFileUploaderDropzone"] {{
            background-color: #BED1E0 !important;
            background: #BED1E0 !important;
            border-radius: 16px !important;
        }}

        /* Change the background color of the "Browse files" button */
        button[data-testid="stBaseButton-secondary"][class*="st-emotion-cache-jszdd5"] {{
            background-color: #4A70A9 !important;
            border-color: #4A70A9 !important;
            color: #BED1E0  !important;
        }}

        /* Optional: change the text color if needed */
        button[data-testid="stBaseButton-secondary"][class*="st-emotion-cache-jszdd5"] span {{
            color: #BED1E0 !important; 
        }}

        /* Optional: hover effect */
        button[data-testid="stBaseButton-secondary"][class*="st-emotion-cache-jszdd5"]:hover {{
            background-color: #EFECE3 !important;
            color: #4A70A9 !important; 
        }}



        /* ===== Input Fields ===== */
        input, textarea, select {{
            background-color: white !important;
            color: {COLORS['primary']} !important;
            border: 1.5px solid {COLORS['primary_light']} !important;
            border-radius: 6px !important;
            padding: 0.5rem 0.8rem !important;
        }}

        input:focus, textarea:focus, select:focus {{
            border-color: {COLORS['primary']} !important;
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(74, 112, 169, 0.15) !important;
        }}

        /* ===== Slider Styling ===== */
        .rc-slider-track {{
            background: {COLORS['primary']} !important;
        }}

        .rc-slider-handle {{
            background-color: {COLORS['primary']} !important;
            border-color: {COLORS['primary']} !important;
        }}

        /* ===== Image Container ===== */
        .stImage > div {{
            background-color: {COLORS['card']} !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            box-shadow: 0 2px 8px rgba(74, 112, 169, 0.08) !important;
        }}

        /* ===== Divider ===== */
        hr {{
            border-color: {COLORS['primary_light']} !important;
        }}

        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar {{
            width: 10px;
        }}

        ::-webkit-scrollbar-track {{
            background: {COLORS['background']};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {COLORS['primary']};
            border-radius: 5px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS['primary_light']};
        }}

        </style>
    """, unsafe_allow_html=True)



def render_footer():
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: {COLORS['text_muted']}; padding: 1.5rem; font-size: 13px;'>
            <p>Image Processing Application | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def initialize_session_state():
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'image_history' not in st.session_state:
        st.session_state.image_history = []

# ---------- UI Helpers for Grayscale & Binary ----------
import io
from PIL import Image
# adjust this import to where your image processing functions live
# e.g. from image_processing import to_grayscale, to_binary_info, to_binary
from image_processing import to_grayscale, to_binary_info, to_binary

def pil_image_to_bytes(img: Image.Image, fmt: str = "PNG") -> io.BytesIO:
    """Return BytesIO for a PIL image (used by st.download_button)."""
    buf = io.BytesIO()
    # Ensure mode is appropriate for saving PNG
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

def show_grayscale_and_binary_ui(input_image: Image.Image = None):
    """
    Display Colored / Grayscale / Binary panels and controls.
    - input_image: PIL.Image (if None, attempt to use st.session_state.original_image)
    Conversions are performed ONLY when the user clicks the compute buttons.
    """
    # use session state image if input not provided
    if input_image is None:
        input_image = st.session_state.get("original_image", None)

    if input_image is None:
        st.info("Upload or select an image to view Grayscale & Binary conversions.")
        return

    # Sidebar controls for member 2
    st.sidebar.header("Member 2 — Grayscale & Binary")
    method_choice = st.sidebar.radio("Binary mode", ("auto", "avg", "otsu"), index=0)
    show_gray = st.sidebar.checkbox("Show Grayscale", value=True)
    show_compare = st.sidebar.checkbox("Show Avg/Otsu Comparison", value=False)

    # Buttons in main UI — user must click to compute
    st.markdown("### Member 2 — Controls")
    btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
    compute_gray = btn_col1.button("Compute Grayscale")
    compute_bin = btn_col2.button("Compute Binary")
    compute_both = btn_col3.button("Compute Both")

    # If user clicked compute, run conversions and store results in session state
    gray_img = None
    bin_img = None
    thresh = None
    chosen_method = None
    reason = ""

    # perform grayscale only when requested (or when both requested)
    if compute_gray or compute_both:
        try:
            gray_img = to_grayscale(input_image)
            st.session_state['member2_gray'] = gray_img
            st.success("Grayscale computed.")
        except Exception as e:
            st.error("Grayscale conversion failed:")
            st.exception(e)

    # perform binary only when requested (or when both requested)
    if compute_bin or compute_both:
        try:
            # to_binary_info supports method selection; respect sidebar choice when provided
            try:
                bin_img, thresh, chosen_method, reason = to_binary_info(input_image, method=method_choice)
            except TypeError:
                # fallback for older signature
                bin_img = to_binary(input_image)
                thresh = None
                chosen_method = "auto"
                reason = "Fallback used (to_binary)."

            st.session_state['member2_binary'] = bin_img
            st.session_state['member2_bin_thresh'] = thresh
            st.session_state['member2_bin_method'] = chosen_method
            st.session_state['member2_bin_reason'] = reason

            # Also update main processed_image so right-panel shows it if you want
            st.session_state.processed_image = bin_img

            if thresh is not None:
                st.success(f"Binary computed: {chosen_method.upper()} — Threshold: {float(thresh):.2f}")
            else:
                st.success(f"Binary computed: {chosen_method.upper()}")
            if reason:
                st.caption(reason)
        except Exception as e:
            st.error("Binary conversion failed:")
            st.exception(e)

    # Load results from session state if available (so they persist between reruns)
    if 'member2_gray' in st.session_state and gray_img is None:
        gray_img = st.session_state.get('member2_gray')
    if 'member2_binary' in st.session_state and bin_img is None:
        bin_img = st.session_state.get('member2_binary')
        thresh = st.session_state.get('member2_bin_thresh')
        chosen_method = st.session_state.get('member2_bin_method')
        reason = st.session_state.get('member2_bin_reason', "")

    # Layout display (three panels)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.subheader("Colored")
        st.image(input_image, use_column_width=True)
    with col2:
        st.subheader("Grayscale")
        if show_gray:
            if gray_img is not None:
                st.image(gray_img, use_column_width=True)
            else:
                st.info("Grayscale not computed yet. Click 'Compute Grayscale'.")
        else:
            st.write("Grayscale hidden — enable in sidebar.")
    with col3:
        st.subheader("Binary (chosen)")
        if bin_img is not None:
            st.image(bin_img, use_column_width=True)
            cap_text = f"Method: **{(chosen_method or 'AUTO').upper()}**"
            if thresh is not None:
                cap_text += f"  — Threshold: **{float(thresh):.2f}**"
            st.caption(cap_text)
        else:
            st.info("Binary not computed yet. Click 'Compute Binary'.")

    # Optional comparison (compute if user asked and results not present)
    if show_compare:
        st.markdown("---")
        st.subheader("Compare Average vs Otsu")
        try:
            # If not computed already, compute both variations on demand (do not save to session_state unless user clicked)
            avg_img, avg_t, _, _ = to_binary_info(input_image, method="avg")
            otsu_img, otsu_t, _, _ = to_binary_info(input_image, method="otsu")
            cmp_cols = st.columns(2)
            with cmp_cols[0]:
                st.markdown("**Average**")
                st.image(avg_img, use_column_width=True)
                st.caption(f"Threshold = {float(avg_t):.2f}")
            with cmp_cols[1]:
                st.markdown("**Otsu**")
                st.image(otsu_img, use_column_width=True)
                st.caption(f"Threshold = {float(otsu_t):.2f}")
        except Exception as e:
            st.warning("Comparison unavailable: " + str(e))

    # Downloads (only if binary exists)
    st.markdown("---")
    dl_cols = st.columns([1,1])
    with dl_cols[0]:
        if bin_img is not None:
            buf = pil_image_to_bytes(bin_img, fmt="PNG")
            st.download_button("Download Binary PNG", data=buf, file_name="binary_result.png", mime="image/png")
        else:
            st.write("Binary PNG download (compute first)")

    with dl_cols[1]:
        if 'member2_gray' in st.session_state or bin_img is not None:
            try:
                # build combined using what's available (use placeholders if missing)
                g = gray_img if gray_img is not None else input_image.convert("L").convert("RGB")
                b = bin_img if bin_img is not None else Image.new("RGB", input_image.size, (255,255,255))
                combined = _build_three_panel(input_image, g, b)
                buf2 = pil_image_to_bytes(combined, fmt="PNG")
                st.download_button("Download Side-by-side", data=buf2, file_name="three_panel.png", mime="image/png")
            except Exception:
                pass

    # Show decision reason (if available and binary computed)
    if method_choice == "auto" and reason:
        st.markdown("**Decision details:**")
        st.caption(reason)


def _build_three_panel(colored: Image.Image, gray: Image.Image, binary: Image.Image, padding: int = 20):
    """
    Build a horizontal combined image with labels (used for download).
    Returns a PIL.Image (RGB).
    """
    # ensure all three images are same height
    imgs = [colored.convert("RGB"), gray.convert("RGB"), binary.convert("RGB")]
    heights = [im.height for im in imgs]
    target_h = max(heights)
    resized = []
    for im in imgs:
        if im.height != target_h:
            new_w = int(im.width * (target_h / im.height))
            resized.append(im.resize((new_w, target_h), Image.LANCZOS))
        else:
            resized.append(im)

    widths = [im.width for im in resized]
    total_w = sum(widths) + padding * (len(resized) + 1)
    total_h = target_h + 60  # room for labels

    canvas = Image.new("RGB", (total_w, total_h), color=(255,255,255))
    x = padding
    draw = None
    for im, label in zip(resized, ("Colored", "Grayscale", "Binary")):
        canvas.paste(im, (x, padding))
        # draw label
        if draw is None:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(canvas)
            # try to load a small font, else default
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
            except Exception:
                font = None
        text_w, text_h = draw.textsize(label, font=font) if font else draw.textsize(label)
        text_x = x + (im.width - text_w) // 2
        text_y = padding + im.height + 8
        draw.text((text_x, text_y), label, fill=(0,0,0), font=font)
        x += im.width + padding

    return canvas
# ---------- End of UI Helpers for Grayscale & Binary ----------
