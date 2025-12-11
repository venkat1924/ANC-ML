"""
Real-Time ANC Simulation Loop.

Simulates the complete hybrid ANC system:
- FxLMS linear controller on raw reference
- TinyMamba neural predictor in parallel
- Physical acoustic paths with speaker nonlinearity
- Safety watchdog for instability detection

Usage:
    python simulate.py --model_path ./checkpoints/mamba_anc_best.pth --duration 10
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.physics import AcousticPhysics
from src.mamba_anc import TinyMambaANC
from src.fxlms import FxLMS
from src.utils import soft_clip, watchdog_check


def parse_args():
    parser = argparse.ArgumentParser(description="Run ANC simulation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--sample_rate", type=int, default=48000,
                        help="Sample rate in Hz")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Processing chunk size")
    parser.add_argument("--mamba_gain", type=float, default=0.0,
                        help="Initial Mamba output gain (0=disabled, tune up slowly)")
    parser.add_argument("--noise_type", type=str, default="engine",
                        choices=["engine", "wind", "mixed", "sweep", "sine"],
                        help="Type of noise to generate")
    parser.add_argument("--output_plot", type=str, default="./anc_results.png",
                        help="Path to save results plot")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for Mamba inference")
    return parser.parse_args()


def generate_test_noise(
    duration: float,
    fs: int,
    noise_type: str = "engine"
) -> np.ndarray:
    """Generate synthetic test noise."""
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    if noise_type == "engine":
        # Engine-like periodic noise (harmonics of 50Hz)
        noise = np.zeros(n_samples)
        for freq in [50, 100, 150, 200, 250]:
            phase = np.random.uniform(0, 2 * np.pi)
            amp = 0.3 / (freq / 50)  # Decreasing harmonics
            noise += amp * np.sin(2 * np.pi * freq * t + phase)
        # Add slight variation
        noise *= 1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
        
    elif noise_type == "wind":
        # Broadband stochastic noise (wind-like)
        noise = np.random.randn(n_samples)
        # Low-pass filter for wind characteristics
        from scipy.signal import butter, lfilter
        b, a = butter(3, 500 / (fs / 2), btype='low')
        noise = lfilter(b, a, noise)
        
    elif noise_type == "mixed":
        # Combination of engine and wind
        engine = generate_test_noise(duration, fs, "engine")
        wind = generate_test_noise(duration, fs, "wind")
        noise = 0.7 * engine + 0.3 * wind
        
    elif noise_type == "sweep":
        # Frequency sweep for testing
        from scipy.signal import chirp
        noise = chirp(t, f0=20, f1=2000, t1=duration, method='logarithmic')
    
    elif noise_type == "sine":
        # Simple 100 Hz sine wave for testing
        noise = 0.5 * np.sin(2 * np.pi * 100 * t)
    
    else:
        noise = np.random.randn(n_samples)
    
    # Normalize
    noise = noise / np.max(np.abs(noise)) * 0.8
    
    return noise.astype(np.float32)


def compute_metrics(
    noise_at_ear: np.ndarray,
    error: np.ndarray,
    fs: int = 48000
) -> dict:
    """Compute ANC performance metrics."""
    # Trim startup transient
    skip = fs // 4  # Skip first 0.25s
    noise_at_ear = noise_at_ear[skip:]
    error = error[skip:]
    
    # NMSE (Normalized Mean Square Error)
    nmse = 10 * np.log10(np.mean(error ** 2) / (np.mean(noise_at_ear ** 2) + 1e-10))
    
    # Active Insertion Loss
    ail = -nmse  # dB of reduction
    
    # Boost probability (% of time ANC makes noise worse)
    boost_prob = np.mean(np.abs(error) > np.abs(noise_at_ear)) * 100
    
    # RMS levels
    rms_noise = np.sqrt(np.mean(noise_at_ear ** 2))
    rms_error = np.sqrt(np.mean(error ** 2))
    
    return {
        "nmse_db": nmse,
        "ail_db": ail,
        "boost_probability": boost_prob,
        "rms_noise": rms_noise,
        "rms_error": rms_error,
        "reduction_linear": rms_noise / (rms_error + 1e-10)
    }


def run_simulation(args):
    """Run the complete ANC simulation."""
    print("=" * 60)
    print("Deep ANC Simulation")
    print("=" * 60)
    
    fs = args.sample_rate
    duration = args.duration
    chunk_size = args.chunk_size
    device = torch.device(args.device)
    
    # 1. Initialize physics simulation
    print("\n1. Initializing acoustic physics...")
    physics = AcousticPhysics(fs=fs, hardware_delay_samples=3, use_pyroomacoustics=False)
    S_hat = physics.get_secondary_path_estimate()
    
    # 2. Initialize FxLMS
    print("2. Initializing FxLMS controller...")
    fxlms = FxLMS(
        n_taps=64,
        learning_rate=0.0005,  # Conservative for stability
        secondary_path_estimate=S_hat
    )
    
    # 3. Initialize Mamba model
    print("3. Initializing TinyMamba model...")
    model = TinyMambaANC(d_model=32, d_state=16, n_layers=2)
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"   Loading weights from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("   Using untrained model (FxLMS will dominate)")
    
    model = model.to(device)
    model.eval()
    
    g = args.mamba_gain  # Mamba output gain
    print(f"   Mamba gain: {g}")
    
    # 4. Generate test noise
    print(f"4. Generating {duration}s of '{args.noise_type}' noise...")
    noise_source = generate_test_noise(duration, fs, args.noise_type)
    n_samples = len(noise_source)
    
    # 5. Preallocate output buffers
    reference_log = np.zeros(n_samples)
    error_log = np.zeros(n_samples)
    noise_at_ear_log = np.zeros(n_samples)
    anti_noise_log = np.zeros(n_samples)
    y_lin_log = np.zeros(n_samples)
    y_deep_log = np.zeros(n_samples)
    
    # 6. Simulation parameters
    n_chunks = n_samples // chunk_size
    watchdog_buffer = np.zeros(100)
    watchdog_triggered = False
    bypass_count = 0
    
    print(f"\n5. Running simulation ({n_chunks} chunks of {chunk_size} samples)...")
    print("-" * 60)
    
    # 7. Main simulation loop - use block-based processing for efficiency
    # Pre-compute full paths first (more accurate than sample-by-sample approx)
    print("   Pre-computing acoustic paths...")
    
    # Full path convolutions
    ref_full = physics.apply_path(noise_source, physics.RIR_ref)
    noise_at_ear_full = physics.apply_path(noise_source, physics.P_z)
    
    # Initialize drive signal buffer for secondary path
    drive_buffer = np.zeros(len(physics.S_z))
    
    for chunk_idx in tqdm(range(n_chunks), desc="Simulating"):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        
        # Get reference signal for this chunk
        ref_chunk = ref_full[start:end]
        noise_at_ear_chunk = noise_at_ear_full[start:end]
        
        # --- A. Get Mamba prediction for this chunk ---
        if g > 0 and not watchdog_triggered:
            x_tensor = torch.from_numpy(ref_chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                y_deep_chunk = model(x_tensor)
                y_deep_chunk = y_deep_chunk.squeeze().cpu().numpy()
            
            if len(y_deep_chunk) != chunk_size:
                if len(y_deep_chunk) < chunk_size:
                    y_deep_chunk = np.pad(y_deep_chunk, (0, chunk_size - len(y_deep_chunk)))
                else:
                    y_deep_chunk = y_deep_chunk[:chunk_size]
        else:
            y_deep_chunk = np.zeros(chunk_size)
        
        # --- B. Process sample-by-sample within chunk (for FxLMS adaptation) ---
        for i in range(chunk_size):
            n = start + i
            ref_sample = ref_chunk[i]
            
            reference_log[n] = ref_sample
            noise_at_ear_log[n] = noise_at_ear_chunk[i]
            
            # FxLMS prediction (on raw reference)
            y_lin = fxlms.predict(ref_sample)
            y_lin_log[n] = y_lin
            
            # Get Mamba prediction for this sample
            y_deep = g * y_deep_chunk[i] if g > 0 else 0
            y_deep_log[n] = y_deep
            
            # Combine outputs (PARALLEL topology)
            y_total = y_lin + y_deep
            
            # Safety limiter (threshold tuned for simulation signal levels)
            y_total = soft_clip(np.array([y_total]), threshold=2.0)[0]
            
            # Apply speaker nonlinearity
            speaker_out = physics.speaker_nonlinearity(np.array([y_total]))[0]
            
            # Update drive buffer and compute anti-noise through S(z)
            drive_buffer = np.roll(drive_buffer, 1)
            drive_buffer[0] = speaker_out
            anti_at_ear = np.dot(drive_buffer, physics.S_z[:len(drive_buffer)])
            anti_noise_log[n] = anti_at_ear
            
            # Error = parallel summation at ear
            error = noise_at_ear_chunk[i] + anti_at_ear
            error_log[n] = error
            
            # Update FxLMS with current error
            fxlms.update(ref_sample, error)
            
            # Update watchdog buffer
            watchdog_buffer = np.roll(watchdog_buffer, 1)
            watchdog_buffer[0] = error
        
        # Check watchdog at end of chunk (only after warmup, with relaxed threshold)
        if chunk_idx > 100:  # Skip first 100 chunks for FxLMS convergence
            triggered, energy_db = watchdog_check(watchdog_buffer, fs=fs, threshold_db=6.0)
            if triggered and not watchdog_triggered:
                print(f"\n  WATCHDOG TRIGGERED at chunk {chunk_idx} (energy: {energy_db:.1f} dB)")
                watchdog_triggered = True
                bypass_count += 1
                g = 0
                fxlms.reset_weights()
    
    # 8. Compute and display metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    metrics = compute_metrics(noise_at_ear_log, error_log, fs)
    
    print(f"\nPerformance Metrics:")
    print(f"  NMSE:              {metrics['nmse_db']:.2f} dB")
    print(f"  Active Insertion:  {metrics['ail_db']:.2f} dB")
    print(f"  Boost Probability: {metrics['boost_probability']:.2f}%")
    print(f"  RMS Reduction:     {metrics['reduction_linear']:.2f}x")
    
    if bypass_count > 0:
        print(f"\n  Watchdog Bypasses: {bypass_count}")
    
    # 9. Plot results
    print(f"\n6. Saving plot to {args.output_plot}...")
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Time axis (show first 0.5 seconds for detail)
    show_samples = min(int(0.5 * fs), n_samples)
    t = np.arange(show_samples) / fs * 1000  # ms
    
    # Plot 1: Noise at ear vs Error
    axes[0].plot(t, noise_at_ear_log[:show_samples], 'b-', alpha=0.7, label='Noise at Ear (ANC OFF)')
    axes[0].plot(t, error_log[:show_samples], 'r-', alpha=0.7, label='Error (ANC ON)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'ANC Performance: {metrics["ail_db"]:.1f} dB Insertion Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Control signals
    axes[1].plot(t, y_lin_log[:show_samples], 'g-', alpha=0.7, label='FxLMS Output')
    axes[1].plot(t, y_deep_log[:show_samples], 'm-', alpha=0.7, label='Mamba Output')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Control Signals (Parallel Topology)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Full duration error
    t_full = np.arange(n_samples) / fs
    axes[2].plot(t_full, noise_at_ear_log, 'b-', alpha=0.5, linewidth=0.5, label='Noise')
    axes[2].plot(t_full, error_log, 'r-', alpha=0.5, linewidth=0.5, label='Error')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Full Duration')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Spectrogram
    from scipy.signal import spectrogram
    f, t_spec, Sxx = spectrogram(error_log, fs=fs, nperseg=1024, noverlap=512)
    axes[3].pcolormesh(t_spec, f[:100], 10*np.log10(Sxx[:100] + 1e-10), shading='gouraud', cmap='viridis')
    axes[3].set_ylabel('Frequency (Hz)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Error Spectrogram')
    
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f"   Plot saved to {args.output_plot}")
    
    plt.close()
    
    return metrics


def main():
    args = parse_args()
    
    # Run simulation
    metrics = run_simulation(args)
    
    print("\nSimulation complete!")
    
    # Check if targets met
    print("\nTarget Check:")
    print(f"  NMSE < -15 dB:  {'✓ PASS' if metrics['nmse_db'] < -15 else '✗ FAIL'} ({metrics['nmse_db']:.1f} dB)")
    print(f"  Boost < 0.1%:   {'✓ PASS' if metrics['boost_probability'] < 0.1 else '✗ FAIL'} ({metrics['boost_probability']:.2f}%)")


if __name__ == "__main__":
    main()

