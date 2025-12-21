"""
Core utilities for Deep ANC system.

Functions:
- soft_clip: Tanh-based limiter preventing amplifier clipping
- watchdog_check: Anti-howling monitor for 2-5kHz instability
- fractional_delay: Sinc interpolation for sub-sample delays
- modified_c_weight: Bass-priority weighting for ANC loss
"""

import numpy as np
import torch
from scipy import signal
from typing import Union, Tuple


def soft_clip(x: Union[np.ndarray, torch.Tensor], threshold: float = 0.95) -> Union[np.ndarray, torch.Tensor]:
    """
    Tanh-based soft limiter to prevent amplifier clipping.
    
    Smoothly limits signal amplitude to prevent hard clipping artifacts
    while maintaining signal shape below threshold.
    
    Args:
        x: Input signal (numpy array or torch tensor)
        threshold: Soft clipping threshold (0-1 range normalized)
    
    Returns:
        Soft-clipped signal with same type as input
    """
    if isinstance(x, torch.Tensor):
        return threshold * torch.tanh(x / threshold)
    else:
        return threshold * np.tanh(x / threshold)


def watchdog_check(
    error_buffer: np.ndarray,
    fs: int = 48000,
    threshold_db: float = -3.0,
    sustained_samples: int = 50
) -> Tuple[bool, float]:
    """
    Anti-howling watchdog: Monitor 2-5kHz energy for instability.
    
    If sustained high energy is detected in the 2-5kHz band,
    the system should bypass ANC to prevent speaker/ear damage.
    
    Args:
        error_buffer: Recent error signal samples
        fs: Sample rate
        threshold_db: Energy threshold in dB (relative to full scale)
        sustained_samples: Number of samples above threshold to trigger
    
    Returns:
        triggered: True if watchdog should activate bypass
        energy_db: Current energy level in dB
    """
    if len(error_buffer) < 256:
        return False, -100.0
    
    # Design bandpass filter for 2-5kHz (howling detection band)
    nyq = fs / 2
    low = 2000 / nyq
    high = min(5000 / nyq, 0.99)
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.lfilter(b, a, error_buffer)
    except ValueError:
        return False, -100.0
    
    # Calculate energy in dB
    energy = np.mean(filtered[-sustained_samples:] ** 2)
    energy_db = 10 * np.log10(energy + 1e-10)
    
    # Check if sustained high energy
    triggered = energy_db > threshold_db
    
    return triggered, energy_db


def fractional_delay(sig: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Apply fractional sample delay using sinc interpolation.
    
    Used for physics-based augmentation to simulate small
    distance changes (sub-sample precision).
    
    Args:
        sig: Input signal
        delay_samples: Delay in samples (can be fractional)
    
    Returns:
        Delayed signal (same length, zero-padded)
    """
    if abs(delay_samples) < 1e-6:
        return sig.copy()
    
    n_taps = 31  # Sinc interpolation filter length
    half_taps = n_taps // 2
    
    # Create sinc kernel with Hamming window
    n = np.arange(n_taps) - half_taps
    
    # Sinc function centered at fractional delay
    frac_delay = delay_samples - int(delay_samples)
    sinc_kernel = np.sinc(n - frac_delay)
    
    # Apply Hamming window
    window = np.hamming(n_taps)
    kernel = sinc_kernel * window
    kernel /= np.sum(kernel)  # Normalize
    
    # Apply integer delay by padding
    int_delay = int(delay_samples)
    if int_delay > 0:
        sig_padded = np.concatenate([np.zeros(int_delay), sig[:-int_delay]])
    elif int_delay < 0:
        sig_padded = np.concatenate([sig[-int_delay:], np.zeros(-int_delay)])
    else:
        sig_padded = sig.copy()
    
    # Apply fractional delay via convolution
    delayed = np.convolve(sig_padded, kernel, mode='same')
    
    return delayed


def modified_c_weight(freqs: Union[np.ndarray, torch.Tensor], fs: int = 48000) -> Union[np.ndarray, torch.Tensor]:
    """
    Modified C-weighting curve for ANC loss function.
    
    Flat (0 dB) from 20Hz-1kHz, then -12dB/octave rolloff above 1kHz.
    This prioritizes bass frequencies where passive isolation fails
    and ANC is most needed.
    
    Why not A-weighting? A-weighting de-emphasizes low frequencies
    where ANC is most effective. C-weighting keeps bass priority.
    
    Args:
        freqs: Frequency array in Hz
        fs: Sample rate (for normalization)
    
    Returns:
        Weight array (linear scale, same shape as freqs)
    """
    is_tensor = isinstance(freqs, torch.Tensor)
    
    if is_tensor:
        freqs_np = freqs.cpu().numpy()
    else:
        freqs_np = np.asarray(freqs)
    
    # Flat below 1kHz, -12dB/octave rolloff above
    weights = np.ones_like(freqs_np, dtype=np.float32)
    
    f_corner = 1000.0  # Corner frequency
    high_mask = freqs_np > f_corner
    
    if np.any(high_mask):
        # -12dB/octave = factor of 4 per octave = (f/f_corner)^(-2)
        weights[high_mask] = (f_corner / freqs_np[high_mask]) ** 2
    
    # Apply low frequency rolloff below 20Hz (inaudible)
    low_mask = freqs_np < 20.0
    if np.any(low_mask):
        weights[low_mask] = (freqs_np[low_mask] / 20.0) ** 2
    
    # Handle DC (0 Hz)
    weights[freqs_np == 0] = 0.0
    
    if is_tensor:
        return torch.from_numpy(weights).to(freqs.device)
    return weights


def a_weight(freqs: np.ndarray) -> np.ndarray:
    """
    Standard A-weighting curve (for comparison/reference).
    
    Note: A-weighting de-emphasizes low frequencies, which is
    NOT ideal for ANC loss. Use modified_c_weight instead.
    
    Args:
        freqs: Frequency array in Hz
    
    Returns:
        Weight array in dB
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    freqs = np.clip(freqs, 1e-6, None)  # Avoid division by zero
    
    # A-weighting formula (IEC 61672-1)
    f2 = freqs ** 2
    
    numerator = 12194**2 * f2**2
    denominator = ((f2 + 20.6**2) * 
                   np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * 
                   (f2 + 12194**2))
    
    Ra = numerator / (denominator + 1e-10)
    A_db = 20 * np.log10(Ra + 1e-10) + 2.0  # +2dB normalization at 1kHz
    
    return A_db


def generate_leakage_tf(n_taps: int = 64, fs: int = 48000) -> np.ndarray:
    """
    Generate a "leaky fit" transfer function for augmentation.
    
    Simulates imperfect acoustic seal in headphones - some
    low-frequency energy leaks in from the environment.
    
    Args:
        n_taps: Filter length
        fs: Sample rate
    
    Returns:
        FIR filter coefficients
    """
    # High-pass characteristic (leakage affects lows)
    # Cutoff around 100-200Hz
    cutoff = np.random.uniform(80, 200)
    
    b = signal.firwin(n_taps, cutoff / (fs / 2), pass_zero=False)
    
    # Add some gain variation
    gain = np.random.uniform(0.7, 1.0)
    
    return b * gain


def db_to_linear(db: float) -> float:
    """Convert decibels to linear scale."""
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """Convert linear scale to decibels."""
    return 20 * np.log10(linear + 1e-10)


def rms(x: np.ndarray) -> float:
    """Calculate RMS (Root Mean Square) of signal."""
    return np.sqrt(np.mean(x ** 2))


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Input audio signal
        target_db: Target peak level in dB (e.g., -3.0)
    
    Returns:
        Normalized audio
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    
    target_linear = db_to_linear(target_db)
    return audio * (target_linear / peak)

