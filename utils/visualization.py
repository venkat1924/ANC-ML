"""
Visualization utilities for ANC results.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_RATE, DURATION, SEQUENCE_LENGTH, ZOOM_DURATION


def plot_results(test_waveform, predicted_waveform, combined_waveform,
                 sample_rate=None, duration=None, sequence_length=None,
                 zoom_duration=None, save_path=None):
    """
    Plot ANC results with 5 subplots.

    Args:
        test_waveform: Original input waveform.
        predicted_waveform: Model-predicted anti-phase waveform.
        combined_waveform: Input + predicted (should approach silence).
        sample_rate: Sample rate. If None, uses config.
        duration: Duration in seconds. If None, uses config.
        sequence_length: Sequence length. If None, uses config.
        zoom_duration: Zoom window duration. If None, uses config.
        save_path: Optional path to save the figure.
    """
    # Use config defaults if not provided
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    if duration is None:
        duration = DURATION
    if sequence_length is None:
        sequence_length = SEQUENCE_LENGTH
    if zoom_duration is None:
        zoom_duration = ZOOM_DURATION

    time_axis = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    zoom_samples = int(sample_rate * zoom_duration)

    plt.figure(figsize=(15, 10))

    # Plot 1: Input Waveform
    plt.subplot(5, 1, 1)
    plt.title('Input Waveform')
    plt.plot(time_axis[:zoom_samples], test_waveform[:zoom_samples], color='blue')
    plt.xlim(0, zoom_duration)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot 2: Predicted Waveform
    plt.subplot(5, 1, 2)
    plt.title('Predicted Waveform (Anti-phase)')
    plt.plot(
        time_axis[sequence_length:sequence_length + zoom_samples],
        predicted_waveform[:zoom_samples],
        color='orange'
    )
    plt.xlim(0, zoom_duration)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot 3: Combined Waveform
    plt.subplot(5, 1, 3)
    plt.title('Combined Waveform (Input + Predicted)')
    plt.plot(
        time_axis[sequence_length:sequence_length + zoom_samples],
        combined_waveform[:zoom_samples],
        color='green'
    )
    plt.xlim(0, zoom_duration)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot 4: Combined vs Input comparison
    plt.subplot(5, 1, 4)
    plt.title('Combined vs Input')
    plt.plot(
        time_axis[sequence_length:sequence_length + zoom_samples],
        combined_waveform[:zoom_samples],
        color='orange',
        label='Combined'
    )
    plt.plot(
        time_axis[sequence_length:sequence_length + zoom_samples],
        test_waveform[sequence_length:sequence_length + zoom_samples],
        color='red',
        label='Input'
    )
    plt.xlim(0, zoom_duration)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot 5: Residual Noise in dB
    residuals = combined_waveform[:zoom_samples]
    residuals_db = 20 * np.log10(np.abs(residuals) + 1e-10)

    plt.subplot(5, 1, 5)
    plt.title('Residual Noise (dB)')
    plt.plot(
        time_axis[sequence_length:sequence_length + zoom_samples],
        residuals_db,
        color='red'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Residual (dB)')
    plt.ylim(-100, 0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

