"""
Utility functions for the Image Processing Application
Contains helper functions for image loading and information display
"""

import streamlit as st
from PIL import Image


def load_image(uploaded_file):
    """
    Load image from uploaded file
    
    Args:
        uploaded_file: File object from Streamlit file uploader
        
    Returns:
        PIL Image object or None if error occurs
    """
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def display_image_info(image):
    """
    Generate formatted image information string
    
    Args:
        image: PIL Image object
        
    Returns:
        Formatted string with image information
    """
    if image:
        info = f"""
        **Resolution:** {image.size[0]} x {image.size[1]} pixels  
        **Format:** {image.format if hasattr(image, 'format') else 'RGB'}  
        **Mode:** {image.mode}
        """
        return info
    return "No image loaded"


def get_image_info_dict(image):
    """
    Get image information as a dictionary
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image information
    """
    if image:
        return {
            "width": image.size[0],
            "height": image.size[1],
            "format": image.format if hasattr(image, 'format') else 'RGB',
            "mode": image.mode
        }
    return None

