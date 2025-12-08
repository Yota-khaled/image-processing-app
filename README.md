# Image Processing Application (Gradio)

A feature-rich image processing app with a Gradio UI. It includes classic image transforms, filters, histogram tools, transform coding, and lossless compression experiments.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Gradio App

```bash
python app_gradio.py
```

The app starts at `http://127.0.0.1:7860` (shown in the console). If you’re inside an IDE panel, open that URL in a browser for fullscreen.

## Features

- **Basic**: Grayscale, Binary (auto/manual)
- **Affine**: Translate, Scale, Rotate, Shear X/Y
- **Interpolation**: Nearest, Bilinear, Bicubic
- **Filters**: Gaussian, Median, Laplacian, Sobel, Gradient
- **Histogram**: View (RGB/Gray), Equalization (RGB/Gray)
- **Operations**: Crop, Zoom, Flip, Brightness, Contrast
- **Transform Coding**: Symbol-based, Bit-plane, Block DCT, Predictive, Wavelet (PyWavelets required)
- **Compression**: Huffman, Golomb-Rice, Arithmetic, LZW, RLE (with correctness check and timing)

## Usage

1. Upload an image.
2. Choose operations from the tabs.
3. View the processed result.
4. Use “Download Processed” to save the current image.

## Development Notes

- Pure processing code lives in `image_processing.py`, `compression.py`, and `transform_coding.py`.
- Gradio UI wiring and handlers live in `app_gradio.py` and `app_handlers.py`.
- Shared Gradio CSS/theme lives in `ui_components.py`.

## Requirements

See `requirements.txt` for full dependency versions. Python 3.9+ is recommended.

