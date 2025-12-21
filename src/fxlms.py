"""
FxLMS (Filtered-x Least Mean Squares) Adaptive Filter.

The workhorse of ANC - handles broadband linear noise effectively.
Works in parallel with Mamba (NOT serial!) to avoid decorrelation.
"""

import numpy as np
from typing import Optional


class FxLMS:
    """
    Filtered-x LMS adaptive filter for ANC.
    
    Key features:
    - Filtered-x algorithm: Uses estimate of secondary path S(z)
    - Leaky integration: Prevents weight explosion
    - Gradient clipping: Stabilizes adaptation
    """
    
    def __init__(
        self,
        n_taps: int = 64,
        mu: float = 0.005,
        leakage: float = 0.9999,
        s_hat: Optional[np.ndarray] = None,
        grad_clip: float = 1.0
    ):
        """
        Initialize FxLMS filter.
        
        Args:
            n_taps: Number of filter taps (64-128 typical for ANC)
            mu: Step size / learning rate (tune until unstable, back off 50%)
            leakage: Leaky LMS coefficient (prevents weight growth)
            s_hat: Estimate of secondary path S(z), or None for identity
            grad_clip: Gradient clipping threshold
        """
        self.n_taps = n_taps
        self.mu = mu
        self.leakage = leakage
        self.grad_clip = grad_clip
        
        # Filter weights
        self.w = np.zeros(n_taps, dtype=np.float32)
        
        # Input buffer (reference signal history)
        self.x_buffer = np.zeros(n_taps, dtype=np.float32)
        
        # Filtered-x buffer (reference filtered through S_hat)
        self.fx_buffer = np.zeros(n_taps, dtype=np.float32)
        
        # Secondary path estimate
        if s_hat is not None:
            self.s_hat = s_hat.astype(np.float32)
        else:
            # Identity (assumes perfect secondary path knowledge)
            self.s_hat = np.zeros(32, dtype=np.float32)
            self.s_hat[0] = 1.0
        
        # Buffer for S_hat filtering
        self.s_buffer = np.zeros(len(self.s_hat), dtype=np.float32)
    
    def predict(self, x_n: float) -> float:
        """
        Generate anti-noise prediction for current sample.
        
        Args:
            x_n: Current reference microphone sample
        
        Returns:
            y_n: Anti-noise output sample
        """
        # Shift input buffer and add new sample
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x_n
        
        # Compute output: y = w^T * x
        y_n = np.dot(self.w, self.x_buffer)
        
        return y_n
    
    def update(self, x_n: float, e_n: float) -> None:
        """
        Update filter weights based on error signal.
        
        Uses Filtered-x LMS:
        w(n+1) = leakage * w(n) - mu * e(n) * x'(n)
        
        where x'(n) is x(n) filtered through S_hat.
        
        IMPORTANT: The negative sign is critical for minimizing error.
        
        Args:
            x_n: Current reference sample (already in buffer from predict)
            e_n: Error microphone sample
        """
        # Update S_hat buffer for filtered-x computation
        self.s_buffer = np.roll(self.s_buffer, 1)
        self.s_buffer[0] = x_n
        
        # Compute filtered reference: x' = S_hat * x
        x_filtered = np.dot(self.s_hat, self.s_buffer[:len(self.s_hat)])
        
        # Update filtered-x buffer
        self.fx_buffer = np.roll(self.fx_buffer, 1)
        self.fx_buffer[0] = x_filtered
        
        # Normalized step size (NLMS-style)
        power = np.dot(self.fx_buffer, self.fx_buffer) + 1e-8
        mu_normalized = self.mu / power
        
        # Compute gradient: e * x'
        gradient = e_n * self.fx_buffer
        
        # Gradient clipping for stability
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.grad_clip:
            gradient = gradient * (self.grad_clip / grad_norm)
        
        # Weight update: w = leakage*w - mu*e*x' (NEGATIVE for gradient descent!)
        self.w = self.leakage * self.w - mu_normalized * gradient
    
    def reset_weights(self):
        """Reset filter weights to zero (emergency bypass)."""
        self.w.fill(0)
        self.x_buffer.fill(0)
        self.fx_buffer.fill(0)
        self.s_buffer.fill(0)
    
    def get_weights(self) -> np.ndarray:
        """Get current filter weights."""
        return self.w.copy()
    
    def set_weights(self, w: np.ndarray) -> None:
        """Set filter weights (for loading checkpoints)."""
        assert len(w) == self.n_taps
        self.w = w.astype(np.float32)
    
    def set_s_hat(self, s_hat: np.ndarray) -> None:
        """
        Update secondary path estimate.
        
        In real systems, S_hat is identified online or offline.
        """
        self.s_hat = s_hat.astype(np.float32)
        self.s_buffer = np.zeros(len(self.s_hat), dtype=np.float32)


class FxLMSBlock:
    """
    Block-based FxLMS for efficient processing.
    
    Processes multiple samples at once using matrix operations.
    More efficient than sample-by-sample for training/simulation.
    """
    
    def __init__(
        self,
        n_taps: int = 64,
        block_size: int = 64,
        mu: float = 0.005,
        leakage: float = 0.9999,
        s_hat: Optional[np.ndarray] = None
    ):
        """
        Initialize block FxLMS.
        
        Args:
            n_taps: Number of filter taps
            block_size: Processing block size
            mu: Step size
            leakage: Leaky coefficient
            s_hat: Secondary path estimate
        """
        self.n_taps = n_taps
        self.block_size = block_size
        self.mu = mu
        self.leakage = leakage
        
        self.w = np.zeros(n_taps, dtype=np.float32)
        
        if s_hat is not None:
            self.s_hat = s_hat.astype(np.float32)
        else:
            self.s_hat = np.zeros(32, dtype=np.float32)
            self.s_hat[0] = 1.0
        
        # State buffers
        self.x_state = np.zeros(n_taps + block_size - 1, dtype=np.float32)
    
    def process_block(
        self,
        x_block: np.ndarray,
        e_block: np.ndarray
    ) -> np.ndarray:
        """
        Process a block of samples.
        
        Args:
            x_block: Reference signal block (block_size,)
            e_block: Error signal block (block_size,)
        
        Returns:
            y_block: Anti-noise output block (block_size,)
        """
        block_size = len(x_block)
        
        # Update state buffer
        self.x_state = np.roll(self.x_state, -block_size)
        self.x_state[-block_size:] = x_block
        
        # Build Toeplitz-like matrix for convolution
        X = np.zeros((block_size, self.n_taps), dtype=np.float32)
        for i in range(block_size):
            X[i] = self.x_state[self.n_taps - 1 + i::-1][:self.n_taps]
        
        # Output: y = X @ w
        y_block = X @ self.w
        
        # Filtered-x (simplified - using X directly)
        Xf = np.convolve(x_block, self.s_hat, mode='full')[:block_size]
        
        # Gradient accumulation
        gradient = np.zeros(self.n_taps, dtype=np.float32)
        for i in range(block_size):
            gradient += e_block[i] * X[i] * Xf[i]
        
        gradient /= block_size
        
        # Weight update
        power = np.mean(Xf ** 2) + 1e-8
        self.w = self.leakage * self.w - (self.mu / power) * gradient
        
        return y_block
    
    def reset(self):
        """Reset filter state."""
        self.w.fill(0)
        self.x_state.fill(0)

