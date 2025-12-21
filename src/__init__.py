"""
Deep ANC: Parallel Predictive Feedforward Active Noise Cancellation

A hybrid system combining:
- FxLMS: Linear adaptive filter for broadband noise
- TinyMamba: Non-linear predictor for residuals

Key principle: Parallel topology (NOT serial) - both operate on raw reference signal.
"""

from .utils import soft_clip, watchdog_check, fractional_delay, modified_c_weight
from .physics import AcousticPhysics
from .dataset import DNSDataset
from .mamba_anc import TinyMambaANC
from .fxlms import FxLMS
from .loss import composite_loss, time_loss, spectral_loss, phase_cosine_loss

__version__ = "1.0.0"
__all__ = [
    "soft_clip",
    "watchdog_check",
    "fractional_delay",
    "modified_c_weight",
    "AcousticPhysics",
    "DNSDataset",
    "TinyMambaANC",
    "FxLMS",
    "composite_loss",
    "time_loss",
    "spectral_loss",
    "phase_cosine_loss",
]

