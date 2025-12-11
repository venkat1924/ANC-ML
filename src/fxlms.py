"""
Filtered-x Least Mean Squares (FxLMS) Adaptive Filter.

The linear component of the hybrid ANC system, handling broadband
noise cancellation in the 20-800Hz range.
"""

import numpy as np
from typing import Optional


class FxLMS:
    """
    Filtered-x LMS adaptive filter for ANC.
    
    Standard FxLMS algorithm with safety features:
    - Weight normalization to prevent divergence
    - Emergency reset capability
    - Gradient clipping
    """
    
    def __init__(
        self,
        n_taps: int = 64,
        learning_rate: float = 0.005,
        secondary_path_estimate: Optional[np.ndarray] = None,
        leakage: float = 0.9999
    ):
        """
        Initialize FxLMS filter.
        
        Args:
            n_taps: Number of filter taps (length of adaptive filter)
            learning_rate: Step size mu (tune: increase until unstable, back off 50%)
            secondary_path_estimate: S_hat(z) for filtering reference signal
            leakage: Leaky LMS coefficient (prevents DC drift, 0.999-0.9999)
        """
        self.n_taps = n_taps
        self.mu = learning_rate
        self.leakage = leakage
        
        # Adaptive filter weights
        self.w = np.zeros(n_taps)
        
        # Reference signal buffer for prediction
        self.x_buffer = np.zeros(n_taps)
        
        # Filtered-x buffer for weight update
        self.fx_buffer = np.zeros(n_taps)
        
        # Secondary path estimate and history buffer
        if secondary_path_estimate is not None:
            self.s_hat = secondary_path_estimate
        else:
            # Default: simple delay approximation
            self.s_hat = np.zeros(64)
            self.s_hat[3] = 1.0  # ~60µs delay at 48kHz
        
        self.s_hat_len = len(self.s_hat)
        self.ref_history = np.zeros(self.s_hat_len)
        
        # Power normalization for NLMS variant
        self.power_estimate = 0.001
        self.power_alpha = 0.99
    
    def update(self, x_n: float, e_n: float) -> None:
        """
        Update adaptive filter weights using FxLMS algorithm.
        
        Weight update: w(n+1) = leakage * w(n) + mu * e(n) * x'(n)
        where x'(n) is the reference signal filtered through S_hat.
        
        Args:
            x_n: Current reference microphone sample
            e_n: Current error microphone sample
        """
        # Update reference history for S_hat filtering
        self.ref_history = np.roll(self.ref_history, 1)
        self.ref_history[0] = x_n
        
        # Compute filtered reference: x'(n) = s_hat * x(n)
        x_filtered = np.dot(self.ref_history, self.s_hat)
        
        # Update filtered-x buffer
        self.fx_buffer = np.roll(self.fx_buffer, 1)
        self.fx_buffer[0] = x_filtered
        
        # Power estimate for NLMS normalization
        self.power_estimate = (
            self.power_alpha * self.power_estimate +
            (1 - self.power_alpha) * (x_filtered ** 2)
        )
        
        # Normalized step size
        mu_normalized = self.mu / (self.power_estimate + 1e-8)
        
        # Gradient for minimizing e² in ANC: w = w - mu * e * x'
        # The negative sign is critical - we want y*S(z) to approach -d(n)
        gradient = e_n * self.fx_buffer
        
        # Gradient clipping for stability
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1.0:
            gradient = gradient / grad_norm
        
        # LMS weight update with leakage (prevents DC drift)
        # NEGATIVE sign for gradient descent to minimize error
        self.w = self.leakage * self.w - mu_normalized * gradient
    
    def predict(self, x_n: float) -> float:
        """
        Generate anti-noise prediction.
        
        Args:
            x_n: Current reference microphone sample
        
        Returns:
            Anti-noise output sample
        """
        # Update reference buffer
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x_n
        
        # FIR filter output
        y = np.dot(self.w, self.x_buffer)
        
        return y
    
    def step(self, x_n: float, e_n: float) -> float:
        """
        Combined predict and update in one call.
        
        Note: Uses PREVIOUS error for update (causality).
        
        Args:
            x_n: Current reference sample
            e_n: Previous error sample
        
        Returns:
            Current anti-noise output
        """
        # First update weights with previous error
        self.update(x_n, e_n)
        
        # Then generate prediction
        return self.predict(x_n)
    
    def reset_weights(self) -> None:
        """
        Emergency reset: zero all weights.
        
        Called by watchdog when instability is detected.
        """
        self.w = np.zeros(self.n_taps)
        self.x_buffer = np.zeros(self.n_taps)
        self.fx_buffer = np.zeros(self.n_taps)
        self.ref_history = np.zeros(self.s_hat_len)
        self.power_estimate = 0.001
    
    def set_learning_rate(self, mu: float) -> None:
        """Dynamically adjust learning rate."""
        self.mu = mu
    
    def get_weights(self) -> np.ndarray:
        """Get current filter weights."""
        return self.w.copy()
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set filter weights (for initialization or transfer)."""
        if len(weights) != self.n_taps:
            raise ValueError(f"Weight length {len(weights)} != n_taps {self.n_taps}")
        self.w = weights.copy()
    
    def get_filter_response(self, n_fft: int = 1024) -> tuple:
        """
        Compute frequency response of current filter.
        
        Args:
            n_fft: FFT size
        
        Returns:
            Tuple of (frequencies, magnitude_db, phase_rad)
        """
        # Zero-pad weights for FFT
        w_padded = np.zeros(n_fft)
        w_padded[:len(self.w)] = self.w
        
        # FFT
        W = np.fft.rfft(w_padded)
        
        # Frequency axis (assuming 48kHz)
        freqs = np.fft.rfftfreq(n_fft, d=1/48000)
        
        # Magnitude in dB
        mag_db = 20 * np.log10(np.abs(W) + 1e-10)
        
        # Phase in radians
        phase_rad = np.angle(W)
        
        return freqs, mag_db, phase_rad

