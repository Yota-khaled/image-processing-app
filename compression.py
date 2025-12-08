"""
Lossless Compression Algorithms
Implementation of various compression algorithms
"""

import heapq
from collections import Counter, defaultdict
import struct
import time
import numpy as np
from typing import Tuple, Dict, List, Optional


# -------------------------
# Huffman Coding
# -------------------------
class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data):
    """Build Huffman tree from data"""
    freq = Counter(data)
    if len(freq) == 0:
        return None
    
    if len(freq) == 1:
        # Special case: only one unique character
        char = list(freq.keys())[0]
        return HuffmanNode(char=char, freq=freq[char])
    
    heap = []
    for char, count in freq.items():
        heapq.heappush(heap, HuffmanNode(char=char, freq=count))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0]


def build_huffman_codes(node, code="", codes=None):
    """Build Huffman codes from tree"""
    if codes is None:
        codes = {}
    
    if node.char is not None:
        codes[node.char] = code if code else "0"
    else:
        if node.left:
            build_huffman_codes(node.left, code + "0", codes)
        if node.right:
            build_huffman_codes(node.right, code + "1", codes)
    
    return codes


def huffman_encode(data):
    """Encode data using Huffman coding"""
    if len(data) == 0:
        return b"", {}
    
    tree = build_huffman_tree(data)
    codes = build_huffman_codes(tree)
    
    encoded_bits = ''.join(codes[char] for char in data)
    
    # Convert bit string to bytes
    padding = 8 - (len(encoded_bits) % 8)
    encoded_bits += '0' * padding
    
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i+8]
        encoded_bytes.append(int(byte, 2))
    
    return bytes(encoded_bytes), codes, padding


def huffman_decode(encoded_data, codes, padding):
    """Decode Huffman encoded data"""
    if len(encoded_data) == 0:
        return []
    
    # Build reverse mapping
    reverse_codes = {v: k for k, v in codes.items()}
    
    # Convert bytes to bit string
    bit_string = ''.join(format(byte, '08b') for byte in encoded_data)
    bit_string = bit_string[:-padding] if padding > 0 else bit_string
    
    decoded = []
    current_code = ""
    for bit in bit_string:
        current_code += bit
        if current_code in reverse_codes:
            decoded.append(reverse_codes[current_code])
            current_code = ""
    
    return decoded


# -------------------------
# Golomb-Rice Coding
# -------------------------
def golomb_rice_encode(data, m):
    """Encode data using Golomb-Rice coding (m must be power of 2)"""
    if m == 0 or (m & (m - 1)) != 0:
        raise ValueError("m must be a power of 2")
    
    k = m.bit_length() - 1
    encoded = []
    
    for value in data:
        q = value // m
        r = value % m
        
        # Unary code for quotient
        unary = '1' * q + '0'
        
        # Binary code for remainder (k bits)
        binary = format(r, f'0{k}b')
        
        encoded.append(unary + binary)
    
    bit_string = ''.join(encoded)
    padding = 8 - (len(bit_string) % 8)
    bit_string += '0' * padding
    
    encoded_bytes = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        encoded_bytes.append(int(byte, 2))
    
    return bytes(encoded_bytes), padding


def golomb_rice_decode(encoded_data, m, padding, data_length):
    """Decode Golomb-Rice encoded data"""
    k = m.bit_length() - 1
    bit_string = ''.join(format(byte, '08b') for byte in encoded_data)
    bit_string = bit_string[:-padding] if padding > 0 else bit_string
    
    decoded = []
    i = 0
    count = 0
    
    while len(decoded) < data_length and i < len(bit_string):
        # Read unary code (quotient)
        q = 0
        while i < len(bit_string) and bit_string[i] == '1':
            q += 1
            i += 1
        
        if i >= len(bit_string):
            break
        
        i += 1  # Skip the '0' that ends unary code
        
        # Read binary code (remainder)
        if i + k > len(bit_string):
            break
        
        r = int(bit_string[i:i+k], 2)
        i += k
        
        value = q * m + r
        decoded.append(value)
    
    return decoded


# -------------------------
# Arithmetic Coding
# -------------------------
def arithmetic_encode(data):
    """Encode data using arithmetic coding"""
    if len(data) == 0:
        return b"", {}
    
    freq = Counter(data)
    total = len(data)
    
    # Build cumulative frequency table
    cum_freq = {}
    cum = 0
    for char in sorted(freq.keys()):
        cum_freq[char] = (cum, cum + freq[char])
        cum += freq[char]
    
    low = 0.0
    high = 1.0
    range_val = 1.0
    
    for char in data:
        char_low, char_high = cum_freq[char]
        low_new = low + range_val * (char_low / total)
        high_new = low + range_val * (char_high / total)
        low = low_new
        high = high_new
        range_val = high - low
    
    # Return the value in the middle of the final range
    encoded_value = (low + high) / 2
    
    # Convert to bytes (using 8 bytes for double precision)
    encoded_bytes = struct.pack('d', encoded_value)
    
    return encoded_bytes, cum_freq, total


def arithmetic_decode(encoded_bytes, cum_freq, total, data_length):
    """Decode arithmetic encoded data"""
    if data_length == 0:
        return []
    
    encoded_value = struct.unpack('d', encoded_bytes)[0]
    
    # Build reverse mapping
    char_list = sorted(cum_freq.keys())
    
    decoded = []
    low = 0.0
    high = 1.0
    
    for _ in range(data_length):
        range_val = high - low
        value = (encoded_value - low) / range_val
        
        # Find which character this value corresponds to
        for char in char_list:
            char_low, char_high = cum_freq[char]
            char_range_low = char_low / total
            char_range_high = char_high / total
            
            if char_range_low <= value < char_range_high:
                decoded.append(char)
                low_new = low + range_val * char_range_low
                high_new = low + range_val * char_range_high
                low = low_new
                high = high_new
                break
    
    return decoded


# -------------------------
# LZW Coding
# -------------------------
def lzw_encode(data):
    """Encode data using LZW algorithm"""
    if len(data) == 0:
        return []
    
    # Initialize dictionary with all possible single characters
    dictionary = {chr(i): i for i in range(256)}
    dict_size = 256
    
    encoded = []
    current = ""
    
    for char in data:
        combined = current + char
        if combined in dictionary:
            current = combined
        else:
            encoded.append(dictionary[current])
            dictionary[combined] = dict_size
            dict_size += 1
            current = char
    
    if current:
        encoded.append(dictionary[current])
    
    return encoded


def lzw_decode(encoded):
    """Decode LZW encoded data"""
    if len(encoded) == 0:
        return ""
    
    # Initialize dictionary
    dictionary = {i: chr(i) for i in range(256)}
    dict_size = 256
    
    decoded = ""
    prev = chr(encoded[0])
    decoded += prev
    
    for code in encoded[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = prev + prev[0]
        else:
            raise ValueError(f"Invalid code: {code}")
        
        decoded += entry
        
        dictionary[dict_size] = prev + entry[0]
        dict_size += 1
        prev = entry
    
    return decoded


# -------------------------
# Run-Length Encoding (RLE)
# -------------------------
def rle_encode(data):
    """Encode data using Run-Length Encoding"""
    if len(data) == 0:
        return []
    
    encoded = []
    i = 0
    
    while i < len(data):
        count = 1
        char = data[i]
        i += 1
        
        while i < len(data) and data[i] == char:
            count += 1
            i += 1
        
        encoded.append((char, count))
    
    return encoded


def rle_decode(encoded):
    """Decode RLE encoded data"""
    decoded = []
    for char, count in encoded:
        decoded.extend([char] * count)
    return decoded


# -------------------------
# Compression Report Helper
# -------------------------
def compress_and_report(data, algorithm_name, encode_func, decode_func, *args):
    """Compress data and generate report"""
    start_time = time.time()
    
    try:
        if algorithm_name == "Huffman":
            encoded, codes, padding = encode_func(data)
            original_size = len(data)
            compressed_size = len(encoded) + len(str(codes))  # Approximate
        elif algorithm_name == "Golomb-Rice":
            m = args[0] if args else 4
            encoded, padding = encode_func(data, m)
            original_size = len(data)
            compressed_size = len(encoded)
        elif algorithm_name == "Arithmetic":
            encoded, cum_freq, total = encode_func(data)
            original_size = len(data)
            compressed_size = len(encoded) + len(str(cum_freq))  # Approximate
        elif algorithm_name == "LZW":
            encoded = encode_func(data)
            original_size = len(data)
            # Convert to bytes for size calculation
            compressed_size = len(encoded) * 2  # Assuming 2 bytes per code
        elif algorithm_name == "RLE":
            encoded = encode_func(data)
            original_size = len(data)
            compressed_size = len(encoded) * 2  # Assuming 2 bytes per (char, count)
        else:
            return None
        
        encode_time = time.time() - start_time
        
        # Decode to verify
        decode_start = time.time()
        if algorithm_name == "Huffman":
            decoded = decode_func(encoded, codes, padding)
        elif algorithm_name == "Golomb-Rice":
            decoded = decode_func(encoded, args[0] if args else 4, padding, len(data))
        elif algorithm_name == "Arithmetic":
            decoded = decode_func(encoded, cum_freq, total, len(data))
        elif algorithm_name == "LZW":
            decoded = decode_func(encoded)
        elif algorithm_name == "RLE":
            decoded = decode_func(encoded)
        
        decode_time = time.time() - decode_start
        
        # Verify correctness
        if isinstance(data, str):
            is_correct = ''.join(map(str, decoded)) == data
        else:
            is_correct = list(decoded) == list(data)
        
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        return {
            'algorithm': algorithm_name,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'encode_time': encode_time,
            'decode_time': decode_time,
            'is_correct': is_correct
        }
    except Exception as e:
        return {
            'algorithm': algorithm_name,
            'error': str(e)
        }

