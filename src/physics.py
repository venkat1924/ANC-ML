"""
Acoustic Digital Twin for ANC Simulation.

Simulates three parallel acoustic paths:
1. RIR_ref: Noise source → Reference microphone
2. P_z: Noise source → Ear (primary path)  
3. S_z: Speaker → Ear (secondary path)

Also includes speaker nonlinearity (Volterra series).
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


class AcousticPhysics:
    """
    Acoustic physics simulation for headphone ANC.
    
    Models the complete acoustic environment including:
    - Reference microphone pickup
    - Primary noise path to ear
    - Secondary (speaker) path to ear
    - Speaker nonlinearity at high drive levels
    """
    
    def __init__(
        self,
        fs: int = 48000,
        rir_length: int = 256,
        primary_delay_ms: float = 0.5,
        secondary_delay_ms: float = 0.2,
        use_realistic_paths: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize acoustic physics simulation.
        
        Args:
            fs: Sample rate
            rir_length: Length of impulse responses in samples
            primary_delay_ms: Primary path delay in milliseconds
            secondary_delay_ms: Secondary path delay in milliseconds
            use_realistic_paths: Use realistic frequency-dependent paths
            seed: Random seed for reproducibility
        """
        self.fs = fs
        self.rir_length = rir_length
        self.primary_delay_samples = int(primary_delay_ms * fs / 1000)
        self.secondary_delay_samples = int(secondary_delay_ms * fs / 1000)
        
        if seed is not None:
            np.random.seed(seed)
        
        if use_realistic_paths:
            self._generate_realistic_paths()
        else:
            self._generate_simple_paths()
    
    def _generate_realistic_paths(self):
        """Generate frequency-dependent realistic acoustic paths."""
        
        # Reference mic path: External noise → reference mic
        # Strong pickup, relatively flat response
        self.RIR_ref = self._create_path(
            gain=0.9,
            delay=2,
            lowpass_hz=8000,
            decay_time_ms=5
        )
        
        # Primary path: External noise → ear (through headphone seal)
        # Weaker (passive isolation), bass-heavy (low freq leaks through)
        self.P_z = self._create_path(
            gain=0.3,  # Passive isolation reduces this
            delay=self.primary_delay_samples,
            lowpass_hz=2000,  # High frequencies blocked by cups
            decay_time_ms=10
        )
        
        # Secondary path: Speaker → ear
        # Strong, relatively flat (speaker is close to ear)
        self.S_z = self._create_path(
            gain=0.8,
            delay=self.secondary_delay_samples,
            lowpass_hz=12000,
            decay_time_ms=3
        )
    
    def _create_path(
        self,
        gain: float,
        delay: int,
        lowpass_hz: float,
        decay_time_ms: float
    ) -> np.ndarray:
        """Create a single acoustic path impulse response."""
        
        # Start with impulse
        ir = np.zeros(self.rir_length)
        ir[min(delay, self.rir_length - 1)] = gain
        
        # Add exponential decay (room reflections)
        decay_samples = int(decay_time_ms * self.fs / 1000)
        decay = np.exp(-np.arange(self.rir_length) / max(decay_samples, 1))
        
        # Add some early reflections
        n_reflections = 3
        for i in range(n_reflections):
            ref_delay = delay + int((i + 1) * decay_samples * 0.3)
            if ref_delay < self.rir_length:
                ir[ref_delay] += gain * 0.2 * (0.5 ** i) * (1 if np.random.rand() > 0.5 else -1)
        
        ir *= decay
        
        # Apply lowpass characteristic
        nyq = self.fs / 2
        cutoff = min(lowpass_hz / nyq, 0.99)
        b, a = signal.butter(2, cutoff, btype='low')
        ir = signal.lfilter(b, a, ir)
        
        # Normalize
        ir = ir / (np.max(np.abs(ir)) + 1e-8) * gain
        
        return ir.astype(np.float32)
    
    def _generate_simple_paths(self):
        """Generate simple exponential decay paths."""
        
        decay = np.exp(-np.arange(self.rir_length) / 30)
        
        # Reference mic - strong, immediate
        self.RIR_ref = np.zeros(self.rir_length)
        self.RIR_ref[0] = 0.9
        self.RIR_ref *= decay
        
        # Primary path - delayed, weaker
        self.P_z = np.zeros(self.rir_length)
        self.P_z[self.primary_delay_samples] = 0.3
        self.P_z *= decay
        
        # Secondary path - speaker to ear
        self.S_z = np.zeros(self.rir_length)
        self.S_z[self.secondary_delay_samples] = 0.8
        self.S_z *= decay
    
    def apply_path(self, signal_in: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """
        Apply impulse response to signal.
        
        Args:
            signal_in: Input signal
            ir: Impulse response
        
        Returns:
            Convolved output (same length as input)
        """
        return np.convolve(signal_in, ir, mode='same')
    
    def speaker_nonlinearity(self, x: np.ndarray, drive_level: float = 0.8) -> np.ndarray:
        """
        Apply Volterra-style speaker nonlinearity.
        
        Models cone excursion limits and suspension nonlinearity.
        At high drive levels, speakers exhibit 3rd-order distortion.
        
        Args:
            x: Input signal (drive signal to speaker)
            drive_level: How hard the speaker is being driven (0-1)
        
        Returns:
            Nonlinear speaker output
        """
        # Volterra 3rd-order nonlinearity
        # y = x - a*x^3 where a increases with drive level
        a = 0.1 * drive_level
        return x - a * (x ** 3)
    
    def simulate_paths(
        self,
        noise_source: np.ndarray,
        drive_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate complete acoustic paths.
        
        This is the core physics simulation: parallel path summation
        at the error microphone.
        
        Args:
            noise_source: External noise source signal
            drive_signal: ANC output drive signal
        
        Returns:
            ref_mic: Reference microphone signal
            error_mic: Error microphone signal (noise + anti-noise)
            noise_at_ear: Noise component at ear
            anti_at_ear: Anti-noise component at ear
        """
        # Reference mic picks up external noise
        ref_mic = self.apply_path(noise_source, self.RIR_ref)
        
        # Noise leaks to ear through primary path
        noise_at_ear = self.apply_path(noise_source, self.P_z)
        
        # Anti-noise from speaker (with nonlinearity) through secondary path
        speaker_out = self.speaker_nonlinearity(drive_signal)
        anti_at_ear = self.apply_path(speaker_out, self.S_z)
        
        # Error mic sees sum of noise and anti-noise (parallel paths!)
        error_mic = noise_at_ear + anti_at_ear
        
        return ref_mic, error_mic, noise_at_ear, anti_at_ear
    
    def simulate_sample(
        self,
        noise_sample: float,
        drive_sample: float,
        noise_buffer: np.ndarray,
        drive_buffer: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Sample-by-sample simulation for real-time loop.
        
        Args:
            noise_sample: Current noise source sample
            drive_sample: Current drive signal sample
            noise_buffer: Buffer of recent noise samples
            drive_buffer: Buffer of recent drive samples
        
        Returns:
            ref_sample: Reference mic sample
            error_sample: Error mic sample
            noise_at_ear: Noise component at ear
        """
        # Update buffers
        noise_buffer = np.roll(noise_buffer, 1)
        noise_buffer[0] = noise_sample
        
        drive_buffer = np.roll(drive_buffer, 1)
        drive_buffer[0] = self.speaker_nonlinearity(np.array([drive_sample]))[0]
        
        # Convolve with truncated IRs
        ref_sample = np.dot(noise_buffer[:len(self.RIR_ref)], self.RIR_ref)
        noise_at_ear = np.dot(noise_buffer[:len(self.P_z)], self.P_z)
        anti_at_ear = np.dot(drive_buffer[:len(self.S_z)], self.S_z)
        
        error_sample = noise_at_ear + anti_at_ear
        
        return ref_sample, error_sample, noise_at_ear
    
    def get_path_info(self) -> dict:
        """Get information about acoustic paths."""
        return {
            "fs": self.fs,
            "rir_length": self.rir_length,
            "RIR_ref_energy": np.sum(self.RIR_ref ** 2),
            "P_z_energy": np.sum(self.P_z ** 2),
            "S_z_energy": np.sum(self.S_z ** 2),
            "primary_delay_samples": self.primary_delay_samples,
            "secondary_delay_samples": self.secondary_delay_samples,
        }

