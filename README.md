# Image Processing Application

A professional image processing application built with Streamlit, featuring a beautiful color scheme and comprehensive image manipulation tools.

## Color Palette

The application uses a custom color scheme:
- **Primary**: `#1B3C53` (Dark Blue)
- **Secondary**: `#234C6A` (Medium Blue)
- **Tertiary**: `#456882` (Lighter Blue)
- **Light**: `#E3E3E3` (Light Gray)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Features

### Basic Operations
- Image Info
- Grayscale Conversion
- Binary Conversion

### Affine Transformations
- Translate
- Scale
- Rotate
- Shear X/Y

### Image Interpolation
- Nearest Neighbor
- Bilinear
- Bicubic

### Histogram Operations
- Show Histogram
- Histogram Equalization

### Filters
- **Low-Pass**: Gaussian, Median
- **High-Pass**: Laplacian, Sobel, Gradient

## Usage

1. Upload an image using the sidebar file uploader
2. Select an operation from the processing tools
3. View the processed image in the right panel
4. Save the result using the "Save Result" button

## Requirements

- Python 3.7+
- Streamlit >= 1.28.0
- Pillow >= 10.0.0
- NumPy >= 1.24.0

