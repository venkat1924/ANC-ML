"""
Audio processing utilities.
"""

import numpy as np
import librosa
from IPython.display import Audio, display

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_RATE


def load_audio(path, sr=None, duration=None):
    """
    Load and normalize an audio file.

    Args:
        path: Path to audio file.
        sr: Sample rate. If None, uses config SAMPLE_RATE.
        duration: Optional duration in seconds to load.

    Returns:
        Normalized audio array.
    """
    if sr is None:
        sr = SAMPLE_RATE

    if duration:
        audio, _ = librosa.load(path, sr=sr, duration=duration)
    else:
        audio, _ = librosa.load(path, sr=sr)

    return normalize_audio(audio)


def normalize_audio(audio):
    """
    Normalize audio to [-1, 1] range.

    Args:
        audio: Audio array.

    Returns:
        Normalized audio array.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def play_audio(audio, sr=None, label=None):
    """
    Display audio player in Jupyter/IPython.

    Args:
        audio: Audio array.
        sr: Sample rate. If None, uses config SAMPLE_RATE.
        label: Optional label to print before audio.
    """
    if sr is None:
        sr = SAMPLE_RATE

    if label:
        print(label)
    display(Audio(audio, rate=sr))

