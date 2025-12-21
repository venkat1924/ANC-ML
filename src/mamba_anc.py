"""
TinyMamba ANC Model: Neural predictor for non-linear noise components.

Architecture:
- Encoder: Strided Conv1d for phase-linear downsampling
- Core: 2x Mamba blocks with skip-connections
- Decoder: ConvTranspose1d for upsampling

The model predicts anti-noise that cancels residuals FxLMS misses.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

# Try to import mamba-ssm, provide fallback if not available
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("mamba-ssm not installed. Using SimpleSSM fallback.")


class SimpleSSM(nn.Module):
    """
    Simple State Space Model fallback when mamba-ssm is not available.
    
    Implements a basic SSM: y = Cx where x evolves via dx/dt = Ax + Bu
    Discretized for sequential processing.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)
        self.B = nn.Linear(self.d_inner, d_state, bias=False)
        self.C = nn.Linear(d_state, self.d_inner, bias=False)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Activation
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, inference_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, T, d_model)
            inference_params: Optional inference state
        
        Returns:
            Output tensor (B, T, d_model)
        """
        B, T, D = x.shape
        
        # Input projection and gate
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Simple recurrent SSM
        x_proj = self.act(x_proj)
        
        # State evolution (simplified)
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(T):
            # Discretized state update: h = A*h + B*x
            b_t = self.B(x_proj[:, t])  # (B, d_state)
            h = 0.9 * h + 0.1 * b_t  # Leaky integration
            
            # Output: y = C*h + D*x
            c_t = self.C(h)  # (B, d_inner)
            y_t = c_t + self.D * x_proj[:, t]
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, T, d_inner)
        
        # Gate and output projection
        y = y * self.act(z)
        y = self.out_proj(y)
        
        return y


class TinyMambaANC(nn.Module):
    """
    TinyMamba model for ANC residual prediction.
    
    Designed for:
    - Low latency (strided conv encoder, not attention)
    - Phase preservation (no IIR filters in signal path)
    - Non-linear modeling (captures what FxLMS can't)
    """
    
    def __init__(
        self,
        d_model: int = 32,
        d_state: int = 16,
        n_layers: int = 2,
        expand: int = 1,
        dropout: float = 0.1,
        leaky_alpha: float = 0.95
    ):
        """
        Initialize TinyMamba model.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            n_layers: Number of Mamba blocks
            expand: Expansion factor for Mamba inner dimension
            dropout: Dropout rate
            leaky_alpha: Leaky state coefficient (prevents drift)
        """
        super().__init__()
        
        self.d_model = d_model
        self.leaky_alpha = leaky_alpha
        
        # Encoder: Strided Conv1d for phase-linear downsampling
        # stride=2 gives 2x compression
        self.encoder = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=4, stride=2, padding=1),
            nn.GELU()
        )
        
        # Mamba blocks with skip-connections
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(n_layers):
            self.norms.append(nn.LayerNorm(d_model))
            if HAS_MAMBA:
                self.layers.append(
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        expand=expand
                    )
                )
            else:
                self.layers.append(
                    SimpleSSM(
                        d_model=d_model,
                        d_state=d_state,
                        expand=expand
                    )
                )
        
        self.dropout = nn.Dropout(dropout)
        
        # Decoder: ConvTranspose1d for upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, 1, kernel_size=4, stride=2, padding=1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 1, T)
            inference_params: Optional inference state for streaming
        
        Returns:
            Output tensor (B, 1, T)
        """
        # Store original length for output matching
        orig_len = x.shape[-1]
        
        # Encoder: (B, 1, T) -> (B, d_model, T//2)
        x = self.encoder(x)
        
        # Transpose for Mamba: (B, d_model, T//2) -> (B, T//2, d_model)
        x = x.transpose(1, 2)
        
        # Mamba blocks with skip connections
        for norm, layer in zip(self.norms, self.layers):
            residual = x
            x = norm(x)
            x = layer(x, inference_params=inference_params)
            x = self.dropout(x)
            x = x + residual  # Skip connection
        
        # Transpose back: (B, T//2, d_model) -> (B, d_model, T//2)
        x = x.transpose(1, 2)
        
        # Decoder: (B, d_model, T//2) -> (B, 1, T)
        x = self.decoder(x)
        
        # Match output length to input length
        if x.shape[-1] != orig_len:
            if x.shape[-1] > orig_len:
                x = x[..., :orig_len]
            else:
                x = nn.functional.pad(x, (0, orig_len - x.shape[-1]))
        
        return x
    
    def apply_leaky_state(self, alpha: Optional[float] = None):
        """
        Apply leaky state reset to prevent drift.
        
        This is called periodically during inference to prevent
        the SSM internal state from drifting over long sequences.
        
        Args:
            alpha: Leaky factor (default: self.leaky_alpha)
        """
        alpha = alpha or self.leaky_alpha
        
        # For SimpleSSM, the state is managed internally with leaky integration
        # For real Mamba, we'd need to access internal states
        # This is a no-op for now but provides the interface
        pass
    
    def reset_state(self):
        """Reset internal state for new sequence."""
        # Interface for streaming inference
        pass
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    model_size: str = "tiny",
    **kwargs
) -> TinyMambaANC:
    """
    Factory function to create model with preset configurations.
    
    Args:
        model_size: "tiny", "small", "medium"
        **kwargs: Override default parameters
    
    Returns:
        TinyMambaANC model
    """
    configs = {
        "tiny": {"d_model": 32, "d_state": 16, "n_layers": 2, "expand": 1},
        "small": {"d_model": 64, "d_state": 32, "n_layers": 3, "expand": 2},
        "medium": {"d_model": 128, "d_state": 64, "n_layers": 4, "expand": 2},
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    return TinyMambaANC(**config)

