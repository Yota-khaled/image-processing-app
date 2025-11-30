# Project Structure

This document describes the organization of the Image Processing Application project.

## File Organization

```
Image Processing Project/
│
├── app.py                 # Main Streamlit application (entry point)
├── config.py              # Configuration and color palette
├── utils.py               # Utility functions (image loading, info display)
├── image_processing.py    # All image processing functions
├── ui_components.py       # UI styling and components
├── __init__.py            # Package initialization
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── PROJECT_STRUCTURE.md   # This file
│
└── .streamlit/
    └── config.toml        # Streamlit theme configuration
```

## Module Descriptions

### `app.py`
- **Purpose**: Main application entry point
- **Contains**: 
  - Streamlit UI layout
  - User interaction handlers
  - Integration of all modules
- **Imports**: All other modules

### `config.py`
- **Purpose**: Centralized configuration
- **Contains**:
  - Color palette definitions
  - Application settings
  - Supported file formats
- **Used by**: All modules that need configuration

### `utils.py`
- **Purpose**: Helper and utility functions
- **Contains**:
  - `load_image()` - Load images from uploaded files
  - `display_image_info()` - Format image information
  - `get_image_info_dict()` - Get image info as dictionary
- **Used by**: `app.py`

### `image_processing.py`
- **Purpose**: All image processing operations
- **Contains**:
  - Basic operations (grayscale, binary)
  - Affine transformations (translate, scale, rotate, shear)
  - Interpolation methods (nearest, bilinear, bicubic)
  - Histogram operations
  - Filter operations (Gaussian, Median, Laplacian, Sobel, Gradient)
- **Used by**: `app.py`

### `ui_components.py`
- **Purpose**: UI styling and reusable components
- **Contains**:
  - `apply_custom_css()` - Apply custom styling
  - `render_footer()` - Render application footer
  - `initialize_session_state()` - Initialize Streamlit session state
- **Used by**: `app.py`

## Benefits of This Structure

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Maintainability**: Easy to find and modify specific functionality
3. **Reusability**: Functions can be imported and used in other projects
4. **Testability**: Each module can be tested independently
5. **Scalability**: Easy to add new features without cluttering the main file

## Adding New Features

### To add a new image processing function:
1. Add the function to `image_processing.py`
2. Import it in `app.py`
3. Add a button/control in the appropriate section of `app.py`

### To change colors/styling:
1. Update `COLORS` dictionary in `config.py`
2. CSS will automatically use the new colors

### To add a new utility function:
1. Add the function to `utils.py`
2. Import and use it where needed

## Dependencies

All dependencies are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

