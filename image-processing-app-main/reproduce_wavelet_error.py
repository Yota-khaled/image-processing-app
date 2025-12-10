
import numpy as np
import transform_coding

def test_wavelet_viz_error():
    # Simulate the user's scenario
    # User said: dimension 0, array 0 has 192, array 1 has 185
    # This implies a fairly large image or multiple levels.
    
    # Let's try with a standard size and db4
    img = np.zeros((512, 512), dtype=np.uint8)
    
    print("Testing db4...")
    try:
        # Using level 3 similar to default
        coeffs, _, _, _ = transform_coding.wavelet_encode(img, 'db4', level=3)
        vis = transform_coding.visualize_wavelet_coeffs(coeffs)
        print("db4 Success")
    except Exception as e:
        print(f"db4 Failed: {e}")

    print("\nTesting bior2.2...")
    try:
        coeffs, _, _, _ = transform_coding.wavelet_encode(img, 'bior2.2', level=3)
        vis = transform_coding.visualize_wavelet_coeffs(coeffs)
        print("bior2.2 Success")
    except Exception as e:
        print(f"bior2.2 Failed: {e}")

if __name__ == "__main__":
    test_wavelet_viz_error()
