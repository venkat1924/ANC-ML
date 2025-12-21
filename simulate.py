#!/usr/bin/env python3
"""
ANC Simulation: Real-time feedforward active noise cancellation.

Simulates the complete ANC system:
1. Acoustic physics (noise paths, speaker nonlinearity)
2. FxLMS adaptive filter (linear broadband cancellation)
3. TinyMamba neural predictor (non-linear residual prediction)

The system uses PARALLEL topology: both FxLMS and Mamba operate
on the raw reference signal, and their outputs are summed.

Usage:
    python simulate.py --duration 5.0
    python simulate.py --model_path ./checkpoints/mamba_anc_best.pth --mamba_gain 0.3
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

from src.physics import AcousticPhysics
from src.fxlms import FxLMS
from src.mamba_anc import TinyMambaANC, create_model
from src.utils import soft_clip, watchdog_check, rms, linear_to_db


def generate_test_noise(duration: float, fs: int = 48000, noise_type: str = "engine") -> np.ndarray:
    """Generate test noise signal."""
    t = np.arange(int(duration * fs)) / fs
    
    if noise_type == "engine":
        # Engine harmonics
        noise = np.zeros(len(t))
        base_freq = 50
        for harm in [1, 2, 3, 4, 5, 6]:
            amp = 0.3 / harm
            noise += amp * np.sin(2 * np.pi * base_freq * harm * t)
        # RPM variation
        noise *= (1 + 0.15 * np.sin(2 * np.pi * 0.3 * t))
        
    elif noise_type == "broadband":
        # Low-frequency broadband
        noise = np.random.randn(len(t))
        b, a = signal.butter(4, 300 / (fs/2), btype='low')
        noise = signal.filtfilt(b, a, noise)
        
    elif noise_type == "mixed":
        # Engine + broadband
        engine = np.zeros(len(t))
        for harm in [1, 2, 3, 4]:
            engine += (0.3 / harm) * np.sin(2 * np.pi * 50 * harm * t)
        
        broadband = np.random.randn(len(t))
        b, a = signal.butter(3, 200 / (fs/2), btype='low')
        broadband = signal.filtfilt(b, a, broadband) * 0.3
        
        noise = engine + broadband
        
    elif noise_type == "sweep":
        # Frequency sweep for testing
        noise = signal.chirp(t, 20, duration, 500) * 0.5
        
    else:
        noise = np.random.randn(len(t)) * 0.3
    
    # Normalize
    noise = noise / (np.max(np.abs(noise)) + 1e-8) * 0.8
    
    return noise.astype(np.float32)


def load_model(model_path: str, device: torch.device) -> TinyMambaANC:
    """Load trained Mamba model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    args = checkpoint.get("args", {})
    model_size = args.get("model_size", "tiny")
    
    model = create_model(model_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Val loss: {checkpoint.get('val_loss', '?'):.4f}")
    
    return model


def run_simulation(
    duration: float = 5.0,
    fs: int = 48000,
    noise_type: str = "engine",
    model_path: str = None,
    mamba_gain: float = 0.0,
    chunk_size: int = 64,
    fxlms_taps: int = 64,
    fxlms_mu: float = 0.0005,
    device: str = "cpu"
):
    """
    Run ANC simulation.
    
    Args:
        duration: Simulation duration in seconds
        fs: Sample rate
        noise_type: Type of test noise
        model_path: Path to trained Mamba model
        mamba_gain: Gain for Mamba output (0 = FxLMS only)
        chunk_size: Processing chunk size
        fxlms_taps: FxLMS filter length
        fxlms_mu: FxLMS learning rate
        device: Torch device
    """
    print("\n" + "=" * 60)
    print("  ANC Simulation")
    print("=" * 60)
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Noise type: {noise_type}")
    print(f"  Mamba gain: {mamba_gain}")
    print(f"  FxLMS: {fxlms_taps} taps, µ={fxlms_mu}")
    print("=" * 60 + "\n")
    
    device = torch.device(device)
    
    # Generate noise
    print("Generating test noise...")
    noise_source = generate_test_noise(duration, fs, noise_type)
    n_samples = len(noise_source)
    
    # Initialize physics
    print("Initializing acoustic physics...")
    physics = AcousticPhysics(fs=fs, seed=42)
    
    # Pre-compute full paths (more accurate than sample-by-sample)
    ref_full = physics.apply_path(noise_source, physics.RIR_ref)
    noise_at_ear_full = physics.apply_path(noise_source, physics.P_z)
    
    # Initialize FxLMS
    fxlms = FxLMS(
        n_taps=fxlms_taps,
        mu=fxlms_mu,
        leakage=0.9999,
        s_hat=physics.S_z[:32]  # Use truncated S(z) estimate
    )
    
    # Initialize Mamba model
    model = None
    g = mamba_gain
    if model_path and mamba_gain > 0:
        try:
            model = load_model(model_path, device)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Running with FxLMS only.")
            g = 0
    elif mamba_gain > 0:
        print("No model path provided, using untrained model (for testing)")
        model = create_model("tiny").to(device)
        model.eval()
    
    # Buffers for real-time simulation
    drive_buffer = np.zeros(len(physics.S_z))
    error_buffer = np.zeros(1024)  # For watchdog
    
    # Output arrays
    reference = np.zeros(n_samples)
    error = np.zeros(n_samples)
    noise_at_ear = np.zeros(n_samples)
    anti_noise = np.zeros(n_samples)
    y_lin_out = np.zeros(n_samples)
    y_deep_out = np.zeros(n_samples)
    
    # Watchdog state
    watchdog_triggered = False
    watchdog_count = 0
    
    # Process in chunks
    n_chunks = n_samples // chunk_size
    
    print("Running simulation...")
    for chunk_idx in tqdm(range(n_chunks), desc="Processing"):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        
        # Get pre-computed signals for this chunk
        ref_chunk = ref_full[start:end]
        noise_at_ear_chunk = noise_at_ear_full[start:end]
        
        # Mamba prediction (chunk-based for efficiency)
        if model is not None and g > 0 and not watchdog_triggered:
            with torch.no_grad():
                x_tensor = torch.from_numpy(ref_chunk).float().unsqueeze(0).unsqueeze(0).to(device)
                y_deep_chunk = model(x_tensor).squeeze().cpu().numpy()
                
                # Match length
                if len(y_deep_chunk) != chunk_size:
                    y_deep_chunk = np.interp(
                        np.linspace(0, 1, chunk_size),
                        np.linspace(0, 1, len(y_deep_chunk)),
                        y_deep_chunk
                    )
        else:
            y_deep_chunk = np.zeros(chunk_size)
        
        # Sample-by-sample processing
        for i in range(chunk_size):
            n = start + i
            ref_sample = ref_chunk[i]
            
            # FxLMS prediction and update
            y_lin = fxlms.predict(ref_sample)
            
            # Mamba contribution
            y_deep = g * y_deep_chunk[i]
            
            # Combine (parallel topology!)
            y_total = soft_clip(np.array([y_lin + y_deep]), threshold=2.0)[0]
            
            # Speaker nonlinearity + secondary path
            speaker_out = physics.speaker_nonlinearity(np.array([y_total]))[0]
            
            # Apply secondary path
            drive_buffer = np.roll(drive_buffer, 1)
            drive_buffer[0] = speaker_out
            anti_at_ear = np.dot(drive_buffer[:len(physics.S_z)], physics.S_z)
            
            # Error signal (noise + anti-noise at ear)
            e = noise_at_ear_chunk[i] + anti_at_ear
            
            # Update FxLMS
            fxlms.update(ref_sample, e)
            
            # Store outputs
            reference[n] = ref_sample
            noise_at_ear[n] = noise_at_ear_chunk[i]
            anti_noise[n] = anti_at_ear
            error[n] = e
            y_lin_out[n] = y_lin
            y_deep_out[n] = y_deep
            
            # Update watchdog buffer
            error_buffer = np.roll(error_buffer, 1)
            error_buffer[0] = e
        
        # Watchdog check (after warmup)
        if chunk_idx > 100:  # Allow FxLMS to converge first
            triggered, energy_db = watchdog_check(error_buffer, fs, threshold_db=6.0)
            if triggered:
                watchdog_count += 1
                if watchdog_count > 5:  # Sustained triggering
                    print(f"\n  WATCHDOG: High-band energy detected ({energy_db:.1f} dB)")
                    watchdog_triggered = True
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # NMSE (Normalized Mean Square Error)
    noise_power = np.mean(noise_at_ear ** 2)
    error_power = np.mean(error ** 2)
    nmse_db = 10 * np.log10(error_power / (noise_power + 1e-10))
    
    # Active Insertion Loss
    ail_db = 10 * np.log10(noise_power / (error_power + 1e-10))
    
    # Boost probability (percentage of time ANC makes it worse)
    boost_mask = np.abs(error) > np.abs(noise_at_ear)
    boost_prob = np.mean(boost_mask) * 100
    
    # RMS reduction
    noise_rms = rms(noise_at_ear)
    error_rms = rms(error)
    rms_reduction = noise_rms / (error_rms + 1e-10)
    
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  NMSE: {nmse_db:.2f} dB")
    print(f"  Active Insertion Loss: {ail_db:.2f} dB")
    print(f"  Boost Probability: {boost_prob:.2f}%")
    print(f"  RMS Reduction: {rms_reduction:.2f}x ({linear_to_db(rms_reduction):.1f} dB)")
    print(f"  Watchdog triggered: {watchdog_triggered}")
    print("=" * 60)
    
    # Plot results
    print("\nGenerating plots...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    t = np.arange(n_samples) / fs
    
    # Time window for detailed view
    t_start = 0.5
    t_end = min(2.5, duration)
    idx_start = int(t_start * fs)
    idx_end = int(t_end * fs)
    
    # Plot 1: Noise vs Error (zoomed)
    axes[0].plot(t[idx_start:idx_end], noise_at_ear[idx_start:idx_end], 
                 label="Noise at ear", alpha=0.7, linewidth=0.8)
    axes[0].plot(t[idx_start:idx_end], error[idx_start:idx_end], 
                 label="Error (residual)", alpha=0.7, linewidth=0.8)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"ANC Performance (AIL: {ail_db:.1f} dB)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Control signals
    axes[1].plot(t[idx_start:idx_end], y_lin_out[idx_start:idx_end], 
                 label="FxLMS output", alpha=0.8, linewidth=0.8)
    axes[1].plot(t[idx_start:idx_end], y_deep_out[idx_start:idx_end], 
                 label=f"Mamba output (g={g})", alpha=0.8, linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Control Signals")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Spectrogram comparison
    ax3 = axes[2]
    f, t_spec, Sxx_noise = signal.spectrogram(noise_at_ear, fs, nperseg=512)
    f, t_spec, Sxx_error = signal.spectrogram(error, fs, nperseg=512)
    
    # Plot noise spectrogram
    ax3.pcolormesh(t_spec, f[:100], 10*np.log10(Sxx_noise[:100] + 1e-10), 
                   shading='gouraud', cmap='inferno', vmin=-60, vmax=0)
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Noise at Ear (Spectrogram)")
    ax3.set_ylim(0, 1000)
    
    # Plot 4: Error spectrogram
    ax4 = axes[3]
    im = ax4.pcolormesh(t_spec, f[:100], 10*np.log10(Sxx_error[:100] + 1e-10), 
                        shading='gouraud', cmap='inferno', vmin=-60, vmax=0)
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Error Signal (Spectrogram)")
    ax4.set_ylim(0, 1000)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("anc_simulation_results.png")
    plt.savefig(output_path, dpi=150)
    print(f"Results saved to {output_path}")
    plt.close()
    
    # Save audio files
    print("\nSaving audio files...")
    audio_dir = Path("audio_output")
    audio_dir.mkdir(exist_ok=True)
    
    def save_audio(name, data):
        data_norm = data / (np.max(np.abs(data)) + 1e-8) * 0.9
        data_int = (data_norm * 32767).astype(np.int16)
        wavfile.write(audio_dir / f"{name}.wav", fs, data_int)
    
    save_audio("noise_at_ear", noise_at_ear)
    save_audio("error_residual", error)
    save_audio("reference", reference)
    print(f"Audio files saved to {audio_dir}/")
    
    return {
        "nmse_db": nmse_db,
        "ail_db": ail_db,
        "boost_prob": boost_prob,
        "rms_reduction": rms_reduction,
        "watchdog_triggered": watchdog_triggered
    }


def main():
    parser = argparse.ArgumentParser(description="Run ANC simulation")
    
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--fs", type=int, default=48000,
                        help="Sample rate")
    parser.add_argument("--noise_type", type=str, default="engine",
                        choices=["engine", "broadband", "mixed", "sweep"],
                        help="Type of test noise")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained Mamba model")
    parser.add_argument("--mamba_gain", type=float, default=0.0,
                        help="Gain for Mamba output (0 = FxLMS only)")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Processing chunk size")
    parser.add_argument("--fxlms_taps", type=int, default=64,
                        help="FxLMS filter length")
    parser.add_argument("--fxlms_mu", type=float, default=0.0005,
                        help="FxLMS learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    
    args = parser.parse_args()
    
    results = run_simulation(
        duration=args.duration,
        fs=args.fs,
        noise_type=args.noise_type,
        model_path=args.model_path,
        mamba_gain=args.mamba_gain,
        chunk_size=args.chunk_size,
        fxlms_taps=args.fxlms_taps,
        fxlms_mu=args.fxlms_mu,
        device=args.device
    )
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()

