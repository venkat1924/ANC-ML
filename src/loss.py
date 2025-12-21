"""
Composite Loss Function for ANC Training.

Components:
1. L_time: MSE in time domain
2. L_spec: C-weighted spectral magnitude loss
3. L_phase: Phase cosine similarity loss
4. L_uncertainty: Penalty for high output on stochastic input

Total: L = λ_t*L_time + λ_s*L_spec + λ_p*L_phase + λ_u*L_uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


def modified_c_weight_torch(freqs: torch.Tensor) -> torch.Tensor:
    """
    Modified C-weighting for loss function (PyTorch version).
    
    Flat 20Hz-1kHz, -12dB/octave rolloff above.
    Prioritizes bass frequencies where ANC is most effective.
    """
    weights = torch.ones_like(freqs)
    
    f_corner = 1000.0
    
    # High frequency rolloff
    high_mask = freqs > f_corner
    weights = torch.where(
        high_mask,
        (f_corner / freqs) ** 2,
        weights
    )
    
    # Low frequency rolloff (below 20Hz)
    low_mask = freqs < 20.0
    weights = torch.where(
        low_mask,
        (freqs / 20.0) ** 2,
        weights
    )
    
    # Handle DC
    weights = torch.where(freqs == 0, torch.zeros_like(weights), weights)
    
    return weights


def time_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Time-domain MSE loss.
    
    Args:
        pred: Predicted signal (B, 1, T) or (B, T)
        target: Target signal (B, 1, T) or (B, T)
    
    Returns:
        MSE loss (scalar)
    """
    return F.mse_loss(pred, target)


def spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 512,
    fs: int = 48000,
    c_weight: bool = True
) -> torch.Tensor:
    """
    C-weighted spectral magnitude loss.
    
    Args:
        pred: Predicted signal (B, 1, T)
        target: Target signal (B, 1, T)
        n_fft: FFT size
        fs: Sample rate
        c_weight: Apply C-weighting
    
    Returns:
        Spectral loss (scalar)
    """
    # Ensure 2D for STFT
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # STFT
    window = torch.hann_window(n_fft, device=pred.device)
    
    pred_stft = torch.stft(
        pred, n_fft, hop_length=n_fft//4, win_length=n_fft,
        window=window, return_complex=True
    )
    target_stft = torch.stft(
        target, n_fft, hop_length=n_fft//4, win_length=n_fft,
        window=window, return_complex=True
    )
    
    # Magnitude
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)
    
    # C-weighting
    if c_weight:
        freqs = torch.linspace(0, fs/2, pred_mag.shape[1], device=pred.device)
        weights = modified_c_weight_torch(freqs)
        weights = weights.view(1, -1, 1)  # (1, F, 1)
        
        pred_mag = pred_mag * weights
        target_mag = target_mag * weights
    
    # Log magnitude for perceptual relevance
    pred_log = torch.log(pred_mag + 1e-8)
    target_log = torch.log(target_mag + 1e-8)
    
    return F.mse_loss(pred_log, target_log)


def phase_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 512
) -> torch.Tensor:
    """
    Phase cosine similarity loss.
    
    L_phase = 1 - cos(θ_pred - θ_target)
    
    Critical for ANC: phase errors > 60° cause noise amplification!
    
    Args:
        pred: Predicted signal (B, 1, T)
        target: Target signal (B, 1, T)
        n_fft: FFT size
    
    Returns:
        Phase loss (scalar, 0 = perfect alignment)
    """
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    window = torch.hann_window(n_fft, device=pred.device)
    
    pred_stft = torch.stft(
        pred, n_fft, hop_length=n_fft//4, win_length=n_fft,
        window=window, return_complex=True
    )
    target_stft = torch.stft(
        target, n_fft, hop_length=n_fft//4, win_length=n_fft,
        window=window, return_complex=True
    )
    
    # Phase difference via complex multiplication
    # angle(pred * conj(target)) = angle(pred) - angle(target)
    phase_diff = pred_stft * torch.conj(target_stft)
    
    # Cosine of phase difference = real part of normalized product
    magnitude = torch.abs(pred_stft) * torch.abs(target_stft) + 1e-8
    cos_diff = phase_diff.real / magnitude
    
    # Loss: 1 - cos(Δθ), averaged
    loss = 1.0 - cos_diff.mean()
    
    return loss


def uncertainty_penalty(
    pred: torch.Tensor,
    input_signal: Optional[torch.Tensor] = None,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Uncertainty penalty to prevent "phantom anti-noise".
    
    When input is highly stochastic/unpredictable, the model
    should output conservative (near-zero) predictions rather
    than guessing and potentially amplifying noise.
    
    Args:
        pred: Predicted signal (B, 1, T)
        input_signal: Input signal for predictability estimation
        threshold: Predictability threshold
    
    Returns:
        Penalty loss (scalar)
    """
    pred_energy = torch.mean(pred ** 2)
    
    if input_signal is not None:
        # Estimate input "predictability" via autocorrelation
        if input_signal.dim() == 3:
            input_flat = input_signal.squeeze(1)
        else:
            input_flat = input_signal
        
        # Simple autocorrelation estimate
        # High autocorrelation = predictable, low = stochastic
        T = input_flat.shape[-1]
        lag = min(100, T // 4)
        
        x1 = input_flat[..., :-lag]
        x2 = input_flat[..., lag:]
        
        autocorr = torch.mean(x1 * x2) / (torch.std(x1) * torch.std(x2) + 1e-8)
        
        # If autocorrelation is low, penalize high output
        unpredictability = 1.0 - torch.abs(autocorr)
        penalty = unpredictability * pred_energy
    else:
        # Without input, just penalize very high outputs
        penalty = F.relu(pred_energy - threshold)
    
    return penalty


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: int = 48000,
    lambda_time: float = 1.0,
    lambda_spec: float = 0.5,
    lambda_phase: float = 0.5,
    lambda_uncertainty: float = 0.1,
    input_signal: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Composite loss combining all components.
    
    L_total = λ_t*L_time + λ_s*L_spec + λ_p*L_phase + λ_u*L_uncertainty
    
    Args:
        pred: Predicted signal (B, 1, T)
        target: Target signal (B, 1, T)
        fs: Sample rate
        lambda_time: Weight for time-domain MSE
        lambda_spec: Weight for spectral loss
        lambda_phase: Weight for phase loss
        lambda_uncertainty: Weight for uncertainty penalty
        input_signal: Input signal (for uncertainty estimation)
    
    Returns:
        total_loss: Combined loss (scalar)
        components: Dict of individual loss components
    """
    # Time domain
    L_time = time_loss(pred, target)
    
    # Spectral (C-weighted)
    L_spec = spectral_loss(pred, target, fs=fs, c_weight=True)
    
    # Phase
    L_phase = phase_cosine_loss(pred, target)
    
    # Uncertainty
    L_uncertainty = uncertainty_penalty(pred, input_signal)
    
    # Combine
    L_total = (
        lambda_time * L_time +
        lambda_spec * L_spec +
        lambda_phase * L_phase +
        lambda_uncertainty * L_uncertainty
    )
    
    components = {
        "time": L_time.item(),
        "spectral": L_spec.item(),
        "phase": L_phase.item(),
        "uncertainty": L_uncertainty.item(),
        "total": L_total.item()
    }
    
    return L_total, components


class CompositeLoss(nn.Module):
    """
    Composite loss as nn.Module for easy integration.
    """
    
    def __init__(
        self,
        fs: int = 48000,
        lambda_time: float = 1.0,
        lambda_spec: float = 0.5,
        lambda_phase: float = 0.5,
        lambda_uncertainty: float = 0.1
    ):
        super().__init__()
        self.fs = fs
        self.lambda_time = lambda_time
        self.lambda_spec = lambda_spec
        self.lambda_phase = lambda_phase
        self.lambda_uncertainty = lambda_uncertainty
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_signal: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return composite_loss(
            pred, target,
            fs=self.fs,
            lambda_time=self.lambda_time,
            lambda_spec=self.lambda_spec,
            lambda_phase=self.lambda_phase,
            lambda_uncertainty=self.lambda_uncertainty,
            input_signal=input_signal
        )

