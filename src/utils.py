"""
Core utilities for Deep ANC system.

Includes safety limiters, watchdog monitoring, signal processing helpers,
and psychoacoustic weighting curves.
"""

import numpy as np
import torch
import scipy.signal as signal
from typing import Union, Tuple


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20)


def linear_to_db(linear: float, eps: float = 1e-10) -> float:
    """Convert linear amplitude to decibels."""
    return 20 * np.log10(np.maximum(linear, eps))


def soft_clip(x: Union[np.ndarray, torch.Tensor], threshold: float = 0.95) -> Union[np.ndarray, torch.Tensor]:
    """
    Tanh-based soft limiter to prevent amplifier clipping.
    
    Smoothly limits signal amplitude to prevent hard clipping artifacts
    while maintaining signal shape below threshold.
    
    Args:
        x: Input signal (numpy array or torch tensor)
        threshold: Maximum output amplitude (0-1)
    
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
    
    Howling/feedback typically manifests as high energy in the 2-5kHz band.
    If sustained high energy is detected, system should bypass to prevent damage.
    
    Args:
        error_buffer: Recent error signal samples (at least sustained_samples long)
        fs: Sample rate in Hz
        threshold_db: Energy threshold in dB (relative to full scale)
        sustained_samples: Number of consecutive samples above threshold to trigger
    
    Returns:
        Tuple of (triggered: bool, current_energy_db: float)
    """
    # Design bandpass filter for 2-5kHz
    nyq = fs / 2
    low = 2000 / nyq
    high = min(5000 / nyq, 0.99)  # Ensure we don't exceed Nyquist
    
    if low >= high:
        # Sample rate too low for this band
        return False, -100.0
    
    try:
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        filtered = signal.sosfilt(sos, error_buffer)
    except ValueError:
        return False, -100.0
    
    # Calculate energy in dB
    energy = np.mean(filtered ** 2)
    energy_db = 10 * np.log10(energy + 1e-10)
    
    threshold_linear = db_to_linear(threshold_db) ** 2
    
    # Check if energy exceeds threshold
    triggered = energy > threshold_linear
    
    return triggered, energy_db


def fractional_delay(
    signal_in: np.ndarray,
    delay_samples: float,
    filter_length: int = 31
) -> np.ndarray:
    """
    Apply fractional sample delay using windowed sinc interpolation.
    
    Physics-accurate sub-sample delay for simulating acoustic path
    length changes without phase distortion artifacts.
    
    Args:
        signal_in: Input signal
        delay_samples: Delay in samples (can be fractional)
        filter_length: Length of sinc filter (odd number, longer = more accurate)
    
    Returns:
        Delayed signal (same length as input)
    """
    if abs(delay_samples) < 1e-6:
        return signal_in.copy()
    
    # Ensure odd filter length
    if filter_length % 2 == 0:
        filter_length += 1
    
    # Create sinc filter centered at fractional delay
    half_len = filter_length // 2
    n = np.arange(-half_len, half_len + 1)
    
    # Fractional part of delay
    frac_delay = delay_samples - int(delay_samples)
    int_delay = int(delay_samples)
    
    # Windowed sinc filter
    sinc_filter = np.sinc(n - frac_delay)
    
    # Apply Hann window to reduce ringing
    window = np.hanning(filter_length)
    sinc_filter = sinc_filter * window
    sinc_filter = sinc_filter / np.sum(sinc_filter)  # Normalize
    
    # Apply fractional delay via convolution
    delayed = signal.convolve(signal_in, sinc_filter, mode='same')
    
    # Apply integer delay via shifting
    if int_delay > 0:
        delayed = np.pad(delayed, (int_delay, 0))[:len(signal_in)]
    elif int_delay < 0:
        delayed = np.pad(delayed, (0, -int_delay))[-int_delay:]
        if len(delayed) < len(signal_in):
            delayed = np.pad(delayed, (0, len(signal_in) - len(delayed)))
    
    return delayed


def modified_c_weight(freqs: Union[np.ndarray, torch.Tensor], fs: int = 48000) -> Union[np.ndarray, torch.Tensor]:
    """
    Modified C-weighting curve for ANC loss function.
    
    Flat (0 dB) from 20Hz-1kHz, then -12dB/octave rolloff above 1kHz.
    This prioritizes bass frequencies where passive isolation fails
    and ANC is most needed.
    
    Args:
        freqs: Frequency values in Hz
        fs: Sample rate (for normalization context)
    
    Returns:
        Weighting values (linear scale, not dB)
    """
    if isinstance(freqs, torch.Tensor):
        weights = torch.ones_like(freqs)
        # -12dB/oct above 1kHz = multiply by (1000/f)^2
        high_freq_mask = freqs > 1000
        weights[high_freq_mask] = (1000.0 / freqs[high_freq_mask]) ** 2
        # Zero out DC and very low frequencies
        weights[freqs < 20] = 0.0
    else:
        weights = np.ones_like(freqs)
        high_freq_mask = freqs > 1000
        weights[high_freq_mask] = (1000.0 / freqs[high_freq_mask]) ** 2
        weights[freqs < 20] = 0.0
    
    return weights


def a_weight(freqs: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Standard A-weighting curve (IEC 61672) for hiss level measurement.
    
    Note: Do NOT use for training loss - A-weighting ignores bass
    where ANC is most needed. Use modified_c_weight instead.
    
    Args:
        freqs: Frequency values in Hz
    
    Returns:
        A-weighting values (linear scale)
    """
    if isinstance(freqs, torch.Tensor):
        f2 = freqs ** 2
        # A-weighting formula
        num = 12194 ** 2 * f2 ** 2
        denom = (f2 + 20.6 ** 2) * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2)
        ra = num / (denom + 1e-10)
        # Normalize to 0dB at 1kHz
        ra = ra / (ra[torch.argmin(torch.abs(freqs - 1000))] + 1e-10)
    else:
        f2 = freqs ** 2
        num = 12194 ** 2 * f2 ** 2
        denom = (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2)
        ra = num / (denom + 1e-10)
        idx_1k = np.argmin(np.abs(freqs - 1000))
        ra = ra / (ra[idx_1k] + 1e-10)
    
    return ra


def generate_leakage_tf(fs: int = 48000, leak_type: str = "light") -> np.ndarray:
    """
    Generate a 'leaky fit' transfer function for augmentation.
    
    Simulates acoustic leakage when headphones don't seal properly
    (glasses, hair, loose fit). Creates characteristic low-frequency
    notch and phase roll-off.
    
    Args:
        fs: Sample rate
        leak_type: "light", "medium", or "heavy" leakage
    
    Returns:
        Impulse response of leakage transfer function
    """
    # Leakage creates a high-pass like effect with resonances
    leak_params = {
        "light": {"fc": 80, "q": 0.7, "gain_db": -3},
        "medium": {"fc": 150, "q": 0.5, "gain_db": -6},
        "heavy": {"fc": 300, "q": 0.4, "gain_db": -12}
    }
    
    params = leak_params.get(leak_type, leak_params["light"])
    
    # Create high-shelf filter to simulate bass loss
    nyq = fs / 2
    fc_norm = params["fc"] / nyq
    
    if fc_norm >= 1:
        fc_norm = 0.9
    
    # Design a simple high-pass to simulate seal leakage
    b, a = signal.butter(2, fc_norm, btype='high')
    
    # Get impulse response
    impulse = np.zeros(256)
    impulse[0] = 1.0
    ir = signal.lfilter(b, a, impulse)
    
    # Apply gain
    ir = ir * db_to_linear(params["gain_db"])
    
    # Blend with direct path (leakage doesn't remove all bass)
    blend = 0.7 if leak_type == "light" else 0.5 if leak_type == "medium" else 0.3
    ir[0] += blend
    
    return ir

