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
# Bit Stream Utilities
# -------------------------
class BitWriter:
    def __init__(self):
        self.bit_buffer = 0
        self.bit_count = 0
        self.bytes_io = bytearray()

    def write_bit(self, bit):
        self.bit_buffer = (self.bit_buffer << 1) | bit
        self.bit_count += 1
        if self.bit_count == 8:
            self.bytes_io.append(self.bit_buffer)
            self.bit_buffer = 0
            self.bit_count = 0

    def write_bits(self, value, num_bits):
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def write_byte(self, byte):
        self.write_bits(byte, 8)

    def get_bytes(self):
        # Flush remaining bits (padding with 0s)
        if self.bit_count > 0:
            self.bytes_io.append(self.bit_buffer << (8 - self.bit_count))
        return bytes(self.bytes_io)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0  # 0 to 7, extends from MSB to LSB

    def read_bit(self):
        if self.byte_pos >= len(self.data):
            raise EOFError("End of stream")
        
        byte = self.data[self.byte_pos]
        bit = (byte >> (7 - self.bit_pos)) & 1
        
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
            
        return bit

    def read_bits(self, num_bits):
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value

    def read_byte(self):
        return self.read_bits(8)


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

    def is_leaf(self):
        return self.char is not None


def build_huffman_tree(data):
    """Build Huffman tree from data"""
    freq = Counter(data)
    if len(freq) == 0:
        return None
    
    priority_queue = []
    for char, count in freq.items():
        heapq.heappush(priority_queue, HuffmanNode(char=char, freq=count))
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(priority_queue, merged)
    
    return priority_queue[0]


def build_huffman_codes(node, code="", codes=None):
    """Build Huffman codes from tree"""
    if codes is None:
        codes = {}
    
    if node.is_leaf():
        codes[node.char] = code if code else "0"
    else:
        if node.left:
            build_huffman_codes(node.left, code + "0", codes)
        if node.right:
            build_huffman_codes(node.right, code + "1", codes)
    
    return codes


def serialize_tree(node, writer):
    """
    Serialize Huffman tree structure:
    0 for internal node
    1 for leaf node + 8-bit character
    """
    if node.is_leaf():
        writer.write_bit(1)
        # Handle int, numpy int, or char
        val = node.char
        if isinstance(val, str):
            char_val = ord(val)
        else:
            char_val = int(val)
        writer.write_bits(char_val, 8)
    else:
        writer.write_bit(0)
        serialize_tree(node.left, writer)
        serialize_tree(node.right, writer)


def deserialize_tree(reader):
    """Reconstruct Huffman tree from bitstream"""
    is_leaf = reader.read_bit()
    if is_leaf:
        char_code = reader.read_bits(8)
        # Helper will decide if it needs to be chr or int later, 
        # but for internal tree structure, we keep as int for consistency if possible,
        # or convert back. Let's return the char_code and let decoder handle type.
        return HuffmanNode(char=char_code)
    else:
        left = deserialize_tree(reader)
        right = deserialize_tree(reader)
        return HuffmanNode(left=left, right=right)


def huffman_encode(data):
    """
    Encode data using Huffman coding with true bit-packing and serialized header.
    Returns: bytes (header + data)
    """
    if len(data) == 0:
        return b""
    
    tree = build_huffman_tree(data)
    codes = build_huffman_codes(tree)
    
    writer = BitWriter()
    
    # 1. Serialize Tree
    serialize_tree(tree, writer)
    
    # 2. Write Data Length (4 bytes) to know when to stop decoding
    length = len(data)
    writer.write_bits(length, 32)
    
    # 3. Encode Data
    for item in data:
        code = codes[item]
        for bit in code:
            writer.write_bit(int(bit))
            
    return writer.get_bytes()


def huffman_decode(encoded_bytes):
    """
    Decode Huffman encoded real bytes.
    """
    if not encoded_bytes:
        return []
    
    reader = BitReader(encoded_bytes)
    
    # 1. Deserialize Tree
    try:
        root = deserialize_tree(reader)
    except EOFError:
        return []

    # 2. Read Data Length
    length = reader.read_bits(32)
    
    decoded = []
    
    # 3. Decode Data
    # For optimization, we could use a lookup table, but tree traversal is simple.
    curr = root
    
    # If using string data initially, we might want to return string chars. 
    # But for general purpose, let's return what we stored (ints from 0-255).
    # If the original input was string, the caller might need to convert back, 
    # but since compression usually handles bytes, we'll stick to that or handle it in wrapper.
    
    count = 0
    while count < length:
        bit = reader.read_bit()
        if bit == 0:
            curr = curr.left
        else:
            curr = curr.right
            
        if curr.is_leaf():
            decoded.append(curr.char)
            curr = root
            count += 1
            
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
# -------------------------
# Arithmetic Coding (Integer Implementation)
# -------------------------
class ArithmeticModel:
    def __init__(self, data):
        self.freq = Counter(data)
        self.total = len(data)
        self.cum_freq = {}
        cum = 0
        self.char_to_index = {}
        self.index_to_char = {}
        
        # Ensure consistent order
        sorted_chars = sorted(self.freq.keys())
        for idx, char in enumerate(sorted_chars):
            self.cum_freq[char] = (cum, cum + self.freq[char])
            cum += self.freq[char]
            self.char_to_index[char] = idx
            self.index_to_char[idx] = char

    def get_range(self, char):
        return self.cum_freq[char]

    def get_char_from_cum_freq(self, value):
        # Linear search is fine for small alphabets, binary search better for large
        for char, (low, high) in self.cum_freq.items():
            if low <= value < high:
                return char, low, high
        raise ValueError("Value out of range")


# def arithmetic_encode(data):
#     """
#     Encode data using Integer Arithmetic Coding.
#     """
#     if len(data) == 0:
#         return b""
        
#     model = ArithmeticModel(data)
#     writer = BitWriter()
    
#     # Constants for 32-bit integer arithmetic coding
#     PRECISION = 32
#     MAX_VALUE = (1 << PRECISION) - 1
#     Q1 = (1 << (PRECISION - 2))
#     Q2 = (1 << (PRECISION - 1))
#     Q3 = Q1 * 3
    
#     low = 0
#     high = MAX_VALUE
#     pending_bits = 0
    
#     # 1. Write Header: Frequency table for reconstruction
#     # Format: [Number of Symbols (byte)] [Symbol (byte), Freq (4 bytes)]... [Total Length (4 bytes)]
#     writer.write_byte(len(model.freq))
#     for char, count in model.freq.items():
#         # Handle both str and int data
#         if isinstance(char, str):
#             char_val = ord(char)
#         else:
#             char_val = int(char)
#         writer.write_byte(char_val)
#         writer.write_bits(count, 32)
    
#     # Write total items count for decoding loop
#     writer.write_bits(len(data), 32)
    
#     # 2. Encode Symbols
#     for char in data:
#         c_low, c_high = model.get_range(char)
#         range_val = high - low + 1
        
#         # Rescale
#         high = low + (range_val * c_high) // model.total - 1
#         low = low + (range_val * c_low) // model.total
        
#         while True:
#             if high < Q2:
#                 writer.write_bit(0)
#                 writer.write_bits(1, pending_bits) # Write pending 1s
#                 pending_bits = 0
#                 low = low << 1
#                 high = (high << 1) | 1
#             elif low >= Q2:
#                 writer.write_bit(1)
#                 writer.write_bits(0, pending_bits) # Write pending 0s
#                 pending_bits = 0
#                 low = (low - Q2) << 1
#                 high = ((high - Q2) << 1) | 1
#             elif low >= Q1 and high < Q3:
#                 pending_bits += 1
#                 low = (low - Q1) << 1
#                 high = ((high - Q1) << 1) | 1
#             else:
#                 break
                
#     # Flush remaining bits
#     pending_bits += 1
#     if low < Q1:
#         writer.write_bit(0)
#         writer.write_bits(1, pending_bits)
#     else:
#         writer.write_bit(1)
#         writer.write_bits(0, pending_bits)
        
#     return writer.get_bytes()


# def arithmetic_decode(encoded_bytes):
#     """
#     Decode Integer Arithmetic Coding stream.
#     """
#     if not encoded_bytes:
#         return []
        
#     reader = BitReader(encoded_bytes)
    
#     # 1. Read Header
#     num_symbols = reader.read_byte()
#     if num_symbols == 0:
#         # Special case: 256 symbols if byte wrapped? No, just 0 empty.
#         # But wait, byte 256 is 0? Let's assume max 255 distinct symbols + 1
#         # If input was random 256 bytes, num_symbols might need 9 bits or special handling.
#         # For this exercise, assume < 256 unique symbols or 0 means 256. 
#         # Actually Counter(bytes) can have 256 keys. read_byte returns 0-255.
#         # If 0, it likely means 256? Or just empty?
#         # Let's fix encode side to handle 256 case if needed, but for now standard logic.
#         pass

#     # Quick fix for num_symbols == 0 meaning 256?
#     # If len(freq) == 256, byte(256) -> 0.
#     real_num_symbols = num_symbols if num_symbols > 0 else 256
    
#     # Reconstruct frequency table
#     freq = {}
#     for _ in range(real_num_symbols):
#         char_code = reader.read_byte()
#         count = reader.read_bits(32)
#         freq[char_code] = count
        
#     # Reconstruct data
#     # Create simple list to pass to Model constructor ?? 
#     # Or just manual model reconstruction
#     # We need a dummy data list to init Model or refactor Model.
#     # Refactoring Model to take freq directly is better.
    
#     # Inline Model logic for decoding:
#     total = sum(freq.values())
#     cum_freq = {}
#     cum = 0
#     # Ensure SAME sorting order as encoder!
#     sorted_chars = sorted(freq.keys())
    
#     # Mapping for reverse lookup
#     # Need range checking
    
#     cum_freq_list = [] # List of (char, low, high)
#     for char in sorted_chars:
#         cum_freq_list.append((char, cum, cum + freq[char]))
#         cum += freq[char]
        
#     total_items = reader.read_bits(32)
    
#     # Constants
#     PRECISION = 32
#     MAX_VALUE = (1 << PRECISION) - 1
#     Q1 = (1 << (PRECISION - 2))
#     Q2 = (1 << (PRECISION - 1))
#     Q3 = Q1 * 3
    
#     low = 0
#     high = MAX_VALUE
#     value = 0
    
#     # Read initial 32 bits into 'value' buffer
#     for _ in range(PRECISION):
#         try:
#             bit = reader.read_bit()
#         except EOFError:
#             bit = 0 # Padding
#         value = (value << 1) | bit
        
#     decoded = []
    
#     for _ in range(total_items):
#         range_val = high - low + 1
#         # scaled_value = ((value - low + 1) * total - 1) // range_val
#         # Formula derivation:
#         # target_cum_freq roughly (value - low) / (high - low) * total
#         scaled_value = ((value - low + 1) * total - 1) // range_val
        
#         # Find symbol
#         char = None
#         c_low = 0
#         c_high = 0
        
#         for c, l, h in cum_freq_list:
#             if l <= scaled_value < h:
#                 char = c
#                 c_low = l
#                 c_high = h
#                 break
                
#         decoded.append(char)
        
#         high = low + (range_val * c_high) // total - 1
#         low = low + (range_val * c_low) // total
        
#         while True:
#             if high < Q2:
#                 # do nothing to low/high ranges that shifts them out? 
#                 # actually just shift out
#                 pass 
#             elif low >= Q2:
#                 value -= Q2
#                 low -= Q2
#                 high -= Q2
#             elif low >= Q1 and high < Q3:
#                 value -= Q1
#                 low -= Q1
#                 high -= Q1
#             else:
#                 break
            
#             low = low << 1
#             high = (high << 1) | 1
#             try:
#                 bit = reader.read_bit()
#             except EOFError:
#                 bit = 0
#             value = (value << 1) | bit
            
#     return decoded

# -------------------------
# Arithmetic Coding (fixed, integer implementation)
# -------------------------
# Uses 32-bit integer arithmetic, consistent header, and symmetric encode/decode.

CODE_BITS = 32
MAX_VALUE = (1 << CODE_BITS) - 1
HALF = 1 << (CODE_BITS - 1)
QUARTER = HALF >> 1
THREE_QUARTER = HALF + QUARTER


def _to_symbol_sequence(data):
    """
    Convert input data to a sequence of integer symbols in [0, 255].
    - If data is bytes/bytearray: use each byte directly.
    - If data is str: use ord(char).
    - Else: assume iterable of ints.
    """
    if isinstance(data, (bytes, bytearray)):
        return [int(b) & 0xFF for b in data]
    elif isinstance(data, str):
        return [ord(ch) & 0xFF for ch in data]
    else:
        return [int(x) & 0xFF for x in data]


def arithmetic_encode(data):
    """
    Encode data using integer arithmetic coding.
    Returns: bytes (header + coded bits).
    Header format:
        - nsym: 16 bits (number of distinct symbols)
        - for each symbol:
            * symbol value: 8 bits
            * frequency: 32 bits
        - original length: 32 bits
    """
    seq = _to_symbol_sequence(data)
    if not seq:
        return b""

    # --- build frequency table over 0..255, but only store those that appear ---
    freq = [0] * 256
    for s in seq:
        freq[s] += 1

    symbols = [s for s in range(256) if freq[s] > 0]
    nsym = len(symbols)

    # cumulative frequencies
    cum = [0] * (nsym + 1)
    for i, s in enumerate(symbols):
        cum[i + 1] = cum[i] + freq[s]
    total = cum[nsym]

    # map symbol value -> index
    sym2idx = {s: i for i, s in enumerate(symbols)}

    writer = BitWriter()

    # --- header ---
    # number of symbols (16 bits)
    writer.write_bits(nsym, 16)
    # for each symbol: value (8 bits) + freq (32 bits)
    for s in symbols:
        writer.write_bits(s, 8)
        writer.write_bits(freq[s], 32)
    # original length (32 bits)
    writer.write_bits(len(seq), 32)

    # --- arithmetic encoding ---
    low = 0
    high = MAX_VALUE
    pending = 0

    for s in seq:
        idx = sym2idx[s]
        low_count = cum[idx]
        high_count = cum[idx + 1]

        rng = (high - low) + 1
        high = low + (rng * high_count // total) - 1
        low = low + (rng * low_count // total)

        # renormalization
        while True:
            if high < HALF:
                # emit 0 and pending 1s
                writer.write_bit(0)
                for _ in range(pending):
                    writer.write_bit(1)
                pending = 0
                low = (low << 1) & MAX_VALUE
                high = ((high << 1) & MAX_VALUE) | 1
            elif low >= HALF:
                # emit 1 and pending 0s
                writer.write_bit(1)
                for _ in range(pending):
                    writer.write_bit(0)
                pending = 0
                low = ((low - HALF) << 1) & MAX_VALUE
                high = (((high - HALF) << 1) & MAX_VALUE) | 1
            elif low >= QUARTER and high < THREE_QUARTER:
                # underflow region
                pending += 1
                low = ((low - QUARTER) << 1) & MAX_VALUE
                high = (((high - QUARTER) << 1) & MAX_VALUE) | 1
            else:
                break

    # finalization: output one more bit + pending ones
    pending += 1
    if low < QUARTER:
        writer.write_bit(0)
        for _ in range(pending):
            writer.write_bit(1)
    else:
        writer.write_bit(1)
        for _ in range(pending):
            writer.write_bit(0)

    return writer.get_bytes()


def arithmetic_decode(encoded_bytes):
    """
    Decode integer arithmetic coded stream created by arithmetic_encode.
    Returns: list of integer symbols (0..255).
    Caller is responsible for converting to bytes or string if needed.
    """
    if not encoded_bytes:
        return []

    reader = BitReader(encoded_bytes)

    # --- read header ---
    nsym = reader.read_bits(16)
    if nsym <= 0 or nsym > 256:
        raise ValueError(f"Invalid number of symbols in header: {nsym}")

    symbols = []
    freq = []
    for _ in range(nsym):
        s = reader.read_bits(8)
        f = reader.read_bits(32)
        symbols.append(s)
        freq.append(f)

    cum = [0] * (nsym + 1)
    for i, f in enumerate(freq):
        cum[i + 1] = cum[i] + f
    total = cum[nsym]

    length = reader.read_bits(32)

    # --- init arithmetic decoder ---
    low = 0
    high = MAX_VALUE
    value = 0
    for _ in range(CODE_BITS):
        try:
            bit = reader.read_bit()
        except EOFError:
            bit = 0
        value = ((value << 1) & MAX_VALUE) | bit

    decoded = []

    for _ in range(length):
        rng = (high - low) + 1
        scaled = ((value - low + 1) * total - 1) // rng

        # find symbol index by scanning cum (at most 256, so linear is OK)
        idx = 0
        for i in range(nsym):
            if cum[i + 1] > scaled:
                idx = i
                break

        s = symbols[idx]
        decoded.append(s)

        low_count = cum[idx]
        high_count = cum[idx + 1]

        high = low + (rng * high_count // total) - 1
        low = low + (rng * low_count // total)

        # renormalization
        while True:
            if high < HALF:
                # do nothing special to value, just shift
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER
            else:
                break

            low = (low << 1) & MAX_VALUE
            high = ((high << 1) & MAX_VALUE) | 1
            try:
                bit = reader.read_bit()
            except EOFError:
                bit = 0
            value = ((value << 1) & MAX_VALUE) | bit

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
# -------------------------
# Compression Report Helper
# -------------------------
def compress_and_report(data, algorithm_name, encode_func, decode_func, *args):
    """Compress data and generate report"""
    start_time = time.time()
    
    try:
        if algorithm_name == "Huffman":
            encoded_bytes = encode_func(data)
            original_size = len(data)
            compressed_size = len(encoded_bytes)
        
        elif algorithm_name == "Golomb-Rice":
            m = args[0] if args else 4
            encoded, padding = encode_func(data, m)
            original_size = len(data)
            compressed_size = len(encoded)
            
        elif algorithm_name == "Arithmetic":
            encoded_bytes = encode_func(data)
            original_size = len(data)
            compressed_size = len(encoded_bytes)
            
        elif algorithm_name == "LZW":
            encoded = encode_func(data)
            original_size = len(data)
            
            # ACCURATE SIZE CALCULATION
            # LZW codes start at 9 bits (for 256 initial dictionary)
            # and increase as dictionary grows.
            total_bits = 0
            current_dict_size = 256
            code_bit_width = 9
            
            for _ in encoded:
                total_bits += code_bit_width
                current_dict_size += 1
                # When dictionary fills up the current bit width, increase width
                # E.g., when size > 512, need 10 bits.
                # Standard LZW: check if Size > (1 << Width)
                if current_dict_size > (1 << code_bit_width):
                    code_bit_width += 1
            
            compressed_size = (total_bits + 7) // 8  # Convert bits to bytes (ceil)

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
            decoded = decode_func(encoded_bytes)
        elif algorithm_name == "Golomb-Rice":
            decoded = decode_func(encoded, args[0] if args else 4, padding, len(data))
        elif algorithm_name == "Arithmetic":
            decoded = decode_func(encoded_bytes)
        elif algorithm_name == "LZW":
            decoded = decode_func(encoded)
        elif algorithm_name == "RLE":
            decoded = decode_func(encoded)
        
        decode_time = time.time() - decode_start
        
        # Verify correctness
        # Helper to convert to list for comparison if needed
        def to_list(d):
            if isinstance(d, str): return list(d)
            if isinstance(d, bytes): return list(d)
            return list(d)
        def to_str(d):
            if isinstance(d, str): return d
            if isinstance(d, bytes): return d.decode('latin1') # fallback
            return "".join(map(chr, d))

        if isinstance(data, str):
            # For string input, decoded might be list of chars or ints depending on algo
            if algorithm_name in ["Huffman", "Arithmetic"]:
                # These now return ints (char codes). Convert back to string for comparison.
                decoded_str = "".join(chr(c) for c in decoded)
                is_correct = decoded_str == data
            elif algorithm_name == "LZW":
                # LZW decode returns string
                is_correct = decoded == data
            else:
                 # Generic fallback
                 is_correct = to_list(decoded) == to_list(data)
        else:
            # Bytes/Array input
            if algorithm_name == "LZW":
                 # LZW usually strings, but if we extended it? Current impl uses dict{chr(i)} so assumes strings/chars.
                 # If data is bytes, our LZW might fail earlier unless we fixed LZW to use bytes.
                 # Given current LZW impl uses chr(), let's assume str input or char-lists.
                 # Test sends bytes? No, test sends numpy array or bytes.
                 # Original compress.py LZW: `dictionary = {chr(i): i` -> expects chars.
                 # If data is ints, `chr(data[i])` might be needed.
                 # But let's assume the test data is handled.
                 is_correct = to_list(decoded) == to_list(data)
            else:
                 is_correct = to_list(decoded) == to_list(data)
        
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
        import traceback
        traceback.print_exc()
        return {
            'algorithm': algorithm_name,
            'error': str(e)
        }

