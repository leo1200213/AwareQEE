# sqnr_calculation.py

import numpy as np
import torch

def quantize(tensor, bits=8):
    """
    簡單的量化函數
    - tensor: 需要量化的PyTorch張量
    - bits: 量化位元，預設為8位
    """
    qmin, qmax = -(2**(bits - 1)), 2**(bits - 1) - 1
    scale = tensor.abs().max() / qmax
    quantized = (tensor / scale).round().clamp(qmin, qmax)
    return quantized * scale

def calculate_sqnr(original_output, quantized_output):
    """
    計算SQNR（Signal-to-Quantization-Noise Ratio）
    - original_output: 原始輸出（numpy array）
    - quantized_output: 量化後的輸出（numpy array）
    """
    signal_power = np.mean(np.square(original_output))
    error_power = np.mean(np.square(original_output - quantized_output))
    sqnr = 10 * np.log10(signal_power / error_power)
    return sqnr
