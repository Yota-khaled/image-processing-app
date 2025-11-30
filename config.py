"""
Configuration file for the Image Processing Application
Contains color palette and application settings
"""

# Updated palette — light background (95%) and stronger darker buttons (20%)
COLORS = {
    # Primary palette
    "primary": "#4A70A9",           # deep blue
    "primary_light": "#8FABD4",     # light blue
    "secondary": "#EFECE3",         # cream/beige
    "dark": "#000000",              # black
    
    # Derived colors
    "background": "#EFECE3",        # cream background
    "card": "#FFFFFF",              # white cards
    "sidebar": "#8FABD4",           # light blue sidebar
    "text": "#EFECE3",              # black text
    "text_muted": "#4A70A9",        # blue muted text
    "text_light": "#8FABD4",        # light blue text
    
    # Accents
    "accent": "#4A70A9",            # use primary as accent
    "accent_light": "#8FABD4",      # light accent
    
    # Status colors (using palette)
    "success": "#4A70A9",           # blue for success
    "warning": "#8FABD4",           # light blue for warning
    "danger": "#000000",            # black for danger
    "info": "#4A70A9"               # blue for info
}

# Application settings
APP_CONFIG = {
    "page_title": "Image Processing Application",
    "page_icon": "",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Supported image formats
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']

