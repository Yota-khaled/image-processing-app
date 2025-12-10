
import numpy as np
import os
from PIL import Image
import compression
import transform_coding



def test_algorithm(name, encode_func, decode_func, data, **kwargs):
    print(f"Testing {name}...")
    try:
        if name == "Predictive":
            encoded = encode_func(data, **kwargs)
            # Test Visualization
            vis = transform_coding.visualize_predictive_error(encoded)
            print(f"  [PASS] Visualization shape: {vis.shape}")
            decoded = decode_func(encoded, **kwargs)
        elif name == "Wavelet":
            encoded, shape, wavelet, level = encode_func(data, **kwargs)
            # Test Visualization
            vis = transform_coding.visualize_wavelet_coeffs(encoded)
            if vis is not None:
                print(f"  [PASS] Visualization shape: {vis.shape}")
            else:
                print(f"  [FAIL] Visualization returned None")
            decoded = decode_func(encoded, shape, wavelet, level)
        elif name == "Bit-plane":
            encoded, shape = encode_func(data, **kwargs)
            decoded = decode_func(encoded, shape)
        elif name == "DCT":
            encoded, original_shape, padded_shape, block_size = encode_func(data, **kwargs)
            decoded = decode_func(encoded, original_shape, padded_shape, block_size)
        elif name == "Huffman":
            # New Signature: encode(data) -> bytes
            # Flatten if numpy
            flat_data = data.flatten() if isinstance(data, np.ndarray) else data
            encoded = encode_func(flat_data)
            decoded_vals = decode_func(encoded)
            # Decoder returns ints/chars. If input was array of ints, decoded is list of ints.
            if isinstance(data, np.ndarray):
                decoded = np.array(decoded_vals, dtype=data.dtype)
                decoded = decoded.reshape(data.shape) # Re-shape if flattened
            else:
                decoded = decoded_vals 
        elif name == "Arithmetic":
            # New Signature: encode(data) -> bytes
            # Flatten if numpy
            flat_data = data.flatten() if isinstance(data, np.ndarray) else data
            encoded = encode_func(flat_data)
            decoded_vals = decode_func(encoded)
            if isinstance(data, np.ndarray):
                decoded = np.array(decoded_vals, dtype=data.dtype)
                decoded = decoded.reshape(data.shape)
            else:
                decoded = decoded_vals
        elif name == "LZW":
            encoded = encode_func(data)
            decoded = decode_func(encoded)
        else:
            # Default for legacy
            return 
            
        # Check correctness
        if name == "Wavelet" or name == "DCT":
             # Allow small float error
             diff = np.abs(data.astype(np.float32) - decoded.astype(np.float32))
             max_diff = np.max(diff)
             if max_diff < 5: # Tolerance
                 print(f"  [PASS] {name} (Max diff: {max_diff})")
             else:
                 print(f"  [FAIL] {name} (Max diff: {max_diff})")
        else:
            # For Huffman/Arithmetic on image data
            if isinstance(data, np.ndarray):
                 # Flatten for comparison if needed, or compare shapes
                 if data.shape != decoded.shape:
                      # If huffman/arithmetic didn't preserve shape (they don't by default), reshape
                      if len(decoded) == data.size:
                          decoded = decoded.reshape(data.shape)
            
            if np.array_equal(data, decoded):
                print(f"  [PASS] {name}")
            else:
                print(f"  [FAIL] {name}")
                # Show diff
                try:
                    diff = np.abs(data.astype(np.int16) - decoded.astype(np.int16))
                    print(f"  Max diff: {np.max(diff)}")
                except:
                    print("  Could not calc diff")

    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Load image
    img_path = "beautiful-natural-image-1844362_1280.jpg"
    if not os.path.exists(img_path):
        print("Image not found, creating random noise")
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    else:
        pil_img = Image.open(img_path).convert('L') # Grayscale for simplicity
        pil_img = pil_img.resize((256, 256))
        img = np.array(pil_img)

    print(f"Image shape: {img.shape}")

    # Test Transform Coding
    test_algorithm("Predictive", transform_coding.predictive_encode, transform_coding.predictive_decode, img, predictor='previous')
    test_algorithm("Wavelet", transform_coding.wavelet_encode, transform_coding.wavelet_decode, img, wavelet='haar', level=2)
    test_algorithm("Bit-plane", transform_coding.bit_plane_encode, transform_coding.bit_plane_decode, img)
    test_algorithm("DCT", transform_coding.dct_block_encode, transform_coding.dct_block_decode, img)

    # Test Lossless Compression (New Improvements)
    print("\n--- Testing Lossless Compression Improvements ---")
    
    # Huffman
    test_algorithm("Huffman", compression.huffman_encode, compression.huffman_decode, img)
    
    # Arithmetic
    test_algorithm("Arithmetic", compression.arithmetic_encode, compression.arithmetic_decode, img)
    
    # LZW (Note: LZW usually works on sequences of symbols. We can test with flat image bytes or string)
    # Using small string for LZW test to avoid huge dictionary growth in naive implementation on image
    lzw_text = "TOBEORNOTTOBEORTOBEORNOT"
    # LZW in compression.py expects string or iterable of chars for dictionary keys
    test_algorithm("LZW", compression.lzw_encode, compression.lzw_decode, lzw_text) 

    # Verify LZW Size Reporting
    print("\n--- Verifying LZW Size Reporting ---")
    report = compression.compress_and_report(lzw_text, "LZW", compression.lzw_encode, compression.lzw_decode)
    print(f"LZW Report: {report}")

if __name__ == "__main__":
    main()
