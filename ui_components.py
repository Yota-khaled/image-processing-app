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

