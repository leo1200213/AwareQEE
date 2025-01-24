# sqnr_calculation.py

import numpy as np
import torch

def quantize(tensor, bits=8):
    """
    Simple quantization function.
    - tensor: torch.Tensor or numpy.ndarray to quantize
    - bits: Number of quantization bits, default is 8
    """
    qmin, qmax = -(2**(bits - 1)), 2**(bits - 1) - 1
    if isinstance(tensor, torch.Tensor):
        # For PyTorch tensors
        scale = tensor.abs().max().item() / qmax
        quantized = torch.clamp((tensor / scale).round(), qmin, qmax)
        return quantized * scale
    elif isinstance(tensor, np.ndarray):
        # For NumPy arrays
        scale = np.abs(tensor).max() / qmax
        quantized = np.clip(np.round(tensor / scale), qmin, qmax)
        return quantized * scale
    else:
        raise TypeError("Input must be a torch.Tensor or a numpy.ndarray")

def calculate_sqnr(original_output, quantized_output):
    """
    Calculates SQNR (Signal-to-Quantization-Noise Ratio).
    - original_output: Original output (numpy.ndarray)
    - quantized_output: Quantized output (numpy.ndarray)
    """
    signal_power = np.mean(np.square(original_output))
    error_power = np.mean(np.square(original_output - quantized_output))
    if error_power == 0:
        return float('inf')  # Infinite SQNR if no error
    sqnr = 10 * np.log10(signal_power / error_power)
    return sqnr
