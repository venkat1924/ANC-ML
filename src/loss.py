"""
Composite Loss Function for Deep ANC Training.

Combines multiple objectives:
- L_time: Time-domain MSE
- L_spec: C-weighted spectral magnitude loss (prioritizes bass)
- L_phase: Phase cosine similarity loss
- L_uncertainty: Penalizes high output on stochastic signals

Research-compliant: Uses C-weighting (NOT A-weighting) to focus on
frequencies where passive isolation fails (20-800Hz).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def modified_c_weight_torch(freqs: torch.Tensor) -> torch.Tensor:
    """
    Modified C-weighting curve for loss function.
    
    Flat (0 dB) from 20Hz-1kHz, then -12dB/octave rolloff above 1kHz.
    Prioritizes bass frequencies where ANC is most needed.
    
    Args:
        freqs: Frequency bins in Hz
    
    Returns:
        Linear-scale weighting values
    """
    weights = torch.ones_like(freqs)
    
    # -12dB/oct above 1kHz = (1000/f)^2 in linear scale
    high_freq_mask = freqs > 1000
    weights[high_freq_mask] = (1000.0 / freqs[high_freq_mask]) ** 2
    
    # Zero out DC and sub-bass (avoid division issues)
    weights[freqs < 20] = 0.0
    
    return weights


def time_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Time-domain MSE loss.
    
    Basic waveform matching - necessary but not sufficient for ANC.
    
    Args:
        pred: Predicted anti-noise (B, 1, T) or (B, T)
        target: Target waveform
    
    Returns:
        Scalar MSE loss
    """
    return F.mse_loss(pred, target)


def spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: int = 48000,
    use_log: bool = True
) -> torch.Tensor:
    """
    C-weighted spectral magnitude loss.
    
    Forces model to prioritize 20-800Hz band where passive isolation fails.
    This is critical - A-weighting would ignore bass and produce useless ANC.
    
    Args:
        pred: Predicted signal (B, 1, T) or (B, T)
        target: Target signal
        fs: Sample rate
        use_log: Use log-magnitude distance (more perceptual)
    
    Returns:
        Weighted spectral loss
    """
    # Ensure 2D for FFT
    if pred.dim() == 3:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    
    # Compute FFT
    n_fft = pred.shape[-1]
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    
    # Frequency bins
    freqs = torch.fft.rfftfreq(n_fft, d=1/fs).to(pred.device)
    
    # C-weighting
    weights = modified_c_weight_torch(freqs)
    
    # Magnitude
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    if use_log:
        # Log-magnitude distance (more perceptual)
        pred_mag = torch.log10(pred_mag + 1e-8)
        target_mag = torch.log10(target_mag + 1e-8)
    
    # Weighted magnitude error
    mag_error = (pred_mag - target_mag) ** 2
    weighted_error = weights.unsqueeze(0) * mag_error
    
    return torch.mean(weighted_error)


def phase_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: int = 48000,
    magnitude_gate: float = 0.01
) -> torch.Tensor:
    """
    Phase cosine similarity loss.
    
    Critical for ANC: phase errors cause amplification, not cancellation.
    - 0° error = perfect cancellation
    - 60° error = no effect
    - >60° error = noise AMPLIFICATION
    
    Args:
        pred: Predicted signal (B, 1, T) or (B, T)
        target: Target signal
        fs: Sample rate
        magnitude_gate: Only penalize phase where magnitude is significant
    
    Returns:
        Phase loss (1 - cos(phase_error)), weighted by C-curve
    """
    if pred.dim() == 3:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    
    n_fft = pred.shape[-1]
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    
    # Frequency bins
    freqs = torch.fft.rfftfreq(n_fft, d=1/fs).to(pred.device)
    weights = modified_c_weight_torch(freqs)
    
    # Cosine similarity in complex domain
    # cos(θ) = Re(pred * conj(target)) / (|pred| * |target|)
    numerator = (pred_fft * target_fft.conj()).real
    denominator = torch.abs(pred_fft) * torch.abs(target_fft) + 1e-8
    
    cos_sim = numerator / denominator
    
    # Magnitude gate: only care about phase where there's energy
    mag_mask = torch.abs(target_fft) > magnitude_gate
    
    # Loss = 1 - cos(θ) (0 when perfectly aligned, 2 when opposite)
    phase_error = 1.0 - cos_sim
    
    # Apply magnitude gate and weighting
    phase_error = phase_error * mag_mask.float()
    weighted_phase_error = weights.unsqueeze(0) * phase_error
    
    return torch.mean(weighted_phase_error)


def uncertainty_penalty(
    pred: torch.Tensor,
    input_energy: Optional[torch.Tensor] = None,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Uncertainty penalty: output zero on stochastic signals.
    
    Prevents "phantom anti-noise" artifacts (clicks/pops) when model
    tries to predict unpredictable stochastic noise like wind gusts.
    
    Conservative strategy: better to output nothing than guess wrong.
    
    Args:
        pred: Predicted signal
        input_energy: Energy of input (if low, prediction should be conservative)
        threshold: Threshold for considering input "low energy"
    
    Returns:
        Penalty for high output when prediction confidence should be low
    """
    pred_energy = torch.mean(pred ** 2, dim=-1, keepdim=True)
    
    if input_energy is not None:
        # Penalize high output when input is low (likely stochastic residual)
        low_input_mask = input_energy < threshold
        penalty = pred_energy * low_input_mask.float()
    else:
        # Simple energy penalty (regularization)
        penalty = F.relu(pred_energy - threshold)
    
    return torch.mean(penalty)


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: int = 48000,
    lambda_time: float = 1.0,
    lambda_spec: float = 0.5,
    lambda_phase: float = 0.5,
    lambda_uncertainty: float = 0.1,
    input_signal: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Complete composite loss for ANC training.
    
    L_total = λ1*L_time + λ2*L_spec + λ3*L_phase + λ4*L_uncertainty
    
    Args:
        pred: Model prediction (B, 1, T) or (B, T)
        target: Target anti-noise signal
        fs: Sample rate
        lambda_time: Weight for time-domain MSE
        lambda_spec: Weight for C-weighted spectral loss
        lambda_phase: Weight for phase cosine loss
        lambda_uncertainty: Weight for uncertainty penalty
        input_signal: Original input (for uncertainty gating)
    
    Returns:
        Tuple of (total_loss, dict of component losses)
    """
    L_time = time_loss(pred, target)
    L_spec = spectral_loss(pred, target, fs)
    L_phase = phase_cosine_loss(pred, target, fs)
    
    if input_signal is not None:
        input_energy = torch.mean(input_signal ** 2, dim=-1, keepdim=True)
        L_uncertainty = uncertainty_penalty(pred, input_energy)
    else:
        L_uncertainty = uncertainty_penalty(pred)
    
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
    PyTorch Module wrapper for composite loss.
    
    Convenient for use in training loops with configurable weights.
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
    ) -> Tuple[torch.Tensor, dict]:
        return composite_loss(
            pred, target,
            fs=self.fs,
            lambda_time=self.lambda_time,
            lambda_spec=self.lambda_spec,
            lambda_phase=self.lambda_phase,
            lambda_uncertainty=self.lambda_uncertainty,
            input_signal=input_signal
        )

