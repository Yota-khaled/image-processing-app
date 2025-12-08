# Project Structure

This document describes the organization of the Image Processing Application project (Gradio UI).

## File Organization

```
image-processing-app-main/
│
├── app_gradio.py        # Gradio entry point (main app)
├── app_handlers.py      # Gradio UI handlers, shared state, orchestration
├── image_processing.py  # Pure image-processing functions (no UI)
├── compression.py       # Lossless compression algorithms + reporter
├── transform_coding.py  # Transform/predictive/wavelet coding utilities
├── ui_components.py     # Shared Gradio CSS/theme snippet
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── PROJECT_STRUCTURE.md # This file

## Module Descriptions

### `app_gradio.py`
- **Purpose**: Main Gradio application (UI layout and wiring).
- **Imports**: Handlers from `app_handlers.py`, CSS from `ui_components.py`.

### `app_handlers.py`
- **Purpose**: UI orchestration layer.
- **Contains**:
  - Shared `global_state` for original/processed images.
  - Event handlers (`handle_*`) for all operations.
  - Load/reset/download helpers.
- **Uses**: `image_processing.py`, `compression.py`, `transform_coding.py`.

### `image_processing.py`
- **Purpose**: Pure image-processing utilities (no UI/state).
- **Contains**:
  - Grayscale/binary conversions
  - Affine transforms (translate/scale/rotate/shear)
  - Interpolation (nearest/bilinear/bicubic)
  - Histogram ops (compute/equalize)
  - Filters (Gaussian/Median/Laplacian/Sobel/Gradient)
  - Basic image ops (crop/zoom/flip/brightness/contrast)

### `compression.py`
- **Purpose**: Lossless compression algorithms and reporting helper.
- **Contains**: Huffman, Golomb-Rice, Arithmetic, LZW, RLE, `compress_and_report`.

### `transform_coding.py`
- **Purpose**: Transform/predictive/wavelet coding utilities.
- **Contains**: Symbol-based, bit-plane, block DCT, predictive coding, wavelet encode/decode.

### `ui_components.py`
- **Purpose**: Shared Gradio CSS theme (used by `app_gradio.py`).


## Adding New Features

1. Add pure processing code to `image_processing.py` (or `compression.py` / `transform_coding.py` as appropriate).
2. Expose it via a handler in `app_handlers.py`.
3. Wire the handler to UI controls in `app_gradio.py`.
4. Style updates go in `ui_components.py`.

## Dependencies

All dependencies are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

