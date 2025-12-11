"""
Acoustic Digital Twin: Physics simulation for ANC system.

Simulates the parallel acoustic paths in a headphone ANC system:
- RIR_ref: Noise source → Reference microphone
- P(z): Noise source → Ear (primary path)
- S(z): Speaker → Ear (secondary path)

Also includes speaker nonlinearity (Volterra) for realistic simulation.
"""

import numpy as np
import scipy.signal as signal

try:
    import pyroomacoustics as pra
    HAS_PRA = True
except ImportError:
    HAS_PRA = False


class AcousticPhysics:
    """
    Acoustic Digital Twin for headphone ANC simulation.
    
    Models the complete acoustic environment including:
    - Primary path (noise to ear)
    - Secondary path (speaker to ear) 
    - Reference microphone path
    - Speaker nonlinearity (Volterra series)
    - Hardware latency (ADC/DAC)
    """
    
    def __init__(
        self,
        fs: int = 48000,
        hardware_delay_samples: int = 3,
        ir_length: int = 256,
        use_pyroomacoustics: bool = True
    ):
        """
        Initialize acoustic physics simulation.
        
        Args:
            fs: Sample rate in Hz
            hardware_delay_samples: ADC+DAC latency (~60µs at 48kHz = 3 samples)
            ir_length: Length of impulse responses
            use_pyroomacoustics: Use pyroomacoustics for RIR generation if available
        """
        self.fs = fs
        self.hardware_delay = hardware_delay_samples
        self.ir_length = ir_length
        
        # Generate the three distinct acoustic paths
        if use_pyroomacoustics and HAS_PRA:
            self._generate_paths_pra()
        else:
            self._generate_paths_synthetic()
        
        # Create imperfect estimate of secondary path for FxLMS
        # In real systems, this is measured via system identification
        self.S_z_hat = self._create_path_estimate(self.S_z, mismatch=0.1)
        
        # Buffers for real-time convolution
        self.ref_buffer = np.zeros(len(self.RIR_ref))
        self.primary_buffer = np.zeros(len(self.P_z))
        self.secondary_buffer = np.zeros(len(self.S_z))
    
    def _generate_paths_pra(self):
        """Generate impulse responses using pyroomacoustics."""
        # Reference mic path: noise source to external mic
        # Simulates open-air propagation with some reflections
        room_ref = pra.ShoeBox(
            [3.0, 3.0, 2.5],  # Room dimensions (m)
            fs=self.fs,
            max_order=5,
            absorption=0.4
        )
        room_ref.add_source([1.5, 1.5, 1.5])  # Noise source
        room_ref.add_microphone([1.6, 1.5, 1.5])  # Ref mic (10cm from source)
        room_ref.compute_rir()
        self.RIR_ref = room_ref.rir[0][0][:self.ir_length]
        self.RIR_ref = self.RIR_ref / (np.max(np.abs(self.RIR_ref)) + 1e-10)
        
        # Primary path: noise source to ear (through headphone shell)
        # Longer path with more attenuation at high frequencies
        room_primary = pra.ShoeBox(
            [3.0, 3.0, 2.5],
            fs=self.fs,
            max_order=8,
            absorption=0.3
        )
        room_primary.add_source([1.5, 1.5, 1.5])
        room_primary.add_microphone([1.65, 1.5, 1.5])  # Ear position (15cm from source)
        room_primary.compute_rir()
        self.P_z = room_primary.rir[0][0][:self.ir_length]
        self.P_z = self.P_z / (np.max(np.abs(self.P_z)) + 1e-10)
        
        # Apply low-pass to simulate passive isolation of ear cup
        nyq = self.fs / 2
        b, a = signal.butter(2, 800 / nyq, btype='low')
        self.P_z = signal.lfilter(b, a, self.P_z)
        self.P_z = self.P_z / (np.max(np.abs(self.P_z)) + 1e-10)
        
        # Secondary path: speaker to ear (inside ear cup)
        # Short, near-field, relatively clean
        room_sec = pra.ShoeBox(
            [0.1, 0.1, 0.05],  # Ear cup dimensions
            fs=self.fs,
            max_order=10,
            absorption=0.3
        )
        room_sec.add_source([0.02, 0.05, 0.025])  # Speaker driver
        room_sec.add_microphone([0.08, 0.05, 0.025])  # Error mic near ear
        room_sec.compute_rir()
        self.S_z = room_sec.rir[0][0][:self.ir_length]
        self.S_z = self.S_z / (np.max(np.abs(self.S_z)) + 1e-10) * 0.8
        
        # Prepend hardware delay to secondary path
        self.S_z = np.pad(self.S_z, (self.hardware_delay, 0))[:self.ir_length]
    
    def _generate_paths_synthetic(self):
        """Generate synthetic impulse responses (fallback without pyroomacoustics)."""
        # Reference path: Ref mic is close to noise source
        # Direct path with minimal delay, high gain
        self.RIR_ref = np.zeros(self.ir_length)
        self.RIR_ref[1] = 0.9  # Direct
        self.RIR_ref[4] = 0.1  # Reflection
        
        # Primary path: Noise travels around headphone to ear
        # Delayed and ATTENUATED by passive isolation
        # Key: P_z should have similar shape but LOWER gain
        self.P_z = np.zeros(self.ir_length)
        self.P_z[8] = 0.4  # Main path (delayed, attenuated by ~7dB)
        self.P_z[12] = 0.1  # Reflection
        self.P_z[18] = 0.05
        
        # Secondary path: Speaker to ear (inside ear cup)
        # Short delay (hardware + acoustic), direct
        self.S_z = np.zeros(self.ir_length)
        self.S_z[self.hardware_delay] = 0.8  # Direct
        self.S_z[self.hardware_delay + 3] = 0.1  # Reflection
    
    def _create_path_estimate(self, true_path: np.ndarray, mismatch: float = 0.1) -> np.ndarray:
        """
        Create imperfect estimate of a path (for FxLMS).
        
        In real systems, this represents measurement error in system ID.
        
        Args:
            true_path: True impulse response
            mismatch: Amount of mismatch (0-1)
        
        Returns:
            Estimated path with added error
        """
        # Scale and add noise
        estimate = true_path * (1.0 - mismatch / 2)
        noise = np.random.normal(0, mismatch * 0.1, len(true_path))
        estimate = estimate + noise
        
        # Small timing error (shift by fraction of a sample)
        # This simulates temperature-dependent speed of sound changes
        return estimate
    
    def speaker_nonlinearity(self, x: np.ndarray, drive_level: float = 0.8) -> np.ndarray:
        """
        Apply Volterra-style speaker nonlinearity.
        
        Models cone excursion limits and suspension nonlinearity.
        At high drive levels, speakers exhibit 3rd-order distortion.
        
        Args:
            x: Input signal (drive voltage)
            drive_level: Normalized drive level (higher = more distortion)
        
        Returns:
            Distorted output signal
        """
        # 3rd-order polynomial approximation of speaker transfer function
        # y = x - a*x^3 (soft saturation)
        # Coefficient scales with drive level
        a = 0.1 * drive_level
        return x - a * (x ** 3)
    
    def apply_path(self, audio: np.ndarray, path: np.ndarray) -> np.ndarray:
        """
        Convolve audio with an impulse response.
        
        Args:
            audio: Input audio signal
            path: Impulse response
        
        Returns:
            Convolved signal (same length as input)
        """
        return signal.fftconvolve(audio, path, mode='full')[:len(audio)]
    
    def simulate_paths(
        self,
        noise_source: np.ndarray,
        drive_signal: np.ndarray
    ) -> tuple:
        """
        Simulate complete acoustic paths.
        
        This is the core physics simulation: parallel path summation
        at the error microphone.
        
        Args:
            noise_source: External noise signal
            drive_signal: Anti-noise drive signal
        
        Returns:
            Tuple of (reference_mic, error_mic, noise_at_ear, anti_at_ear)
        """
        # Path 1: Noise to reference microphone
        ref_mic = self.apply_path(noise_source, self.RIR_ref)
        
        # Path 2: Noise to ear (primary path)
        noise_at_ear = self.apply_path(noise_source, self.P_z)
        
        # Path 3: Drive through speaker nonlinearity then secondary path
        speaker_out = self.speaker_nonlinearity(drive_signal)
        anti_at_ear = self.apply_path(speaker_out, self.S_z)
        
        # Error microphone: PARALLEL summation
        error_mic = noise_at_ear + anti_at_ear
        
        return ref_mic, error_mic, noise_at_ear, anti_at_ear
    
    def step(
        self,
        noise_sample: float,
        drive_sample: float
    ) -> tuple:
        """
        Single-sample physics simulation for real-time loop.
        
        Uses internal buffers for efficient sample-by-sample convolution.
        
        Args:
            noise_sample: Current noise source sample
            drive_sample: Current drive signal sample
        
        Returns:
            Tuple of (ref_mic_sample, error_sample)
        """
        # Update reference buffer and compute output
        self.ref_buffer = np.roll(self.ref_buffer, 1)
        self.ref_buffer[0] = noise_sample
        ref_out = np.dot(self.ref_buffer, self.RIR_ref[:len(self.ref_buffer)])
        
        # Update primary buffer and compute noise at ear
        self.primary_buffer = np.roll(self.primary_buffer, 1)
        self.primary_buffer[0] = noise_sample
        noise_at_ear = np.dot(self.primary_buffer, self.P_z[:len(self.primary_buffer)])
        
        # Apply speaker nonlinearity
        speaker_out = self.speaker_nonlinearity(np.array([drive_sample]))[0]
        
        # Update secondary buffer and compute anti-noise at ear
        self.secondary_buffer = np.roll(self.secondary_buffer, 1)
        self.secondary_buffer[0] = speaker_out
        anti_at_ear = np.dot(self.secondary_buffer, self.S_z[:len(self.secondary_buffer)])
        
        # Error = parallel summation
        error = noise_at_ear + anti_at_ear
        
        return ref_out, error
    
    def reset_buffers(self):
        """Reset internal convolution buffers to zero."""
        self.ref_buffer = np.zeros(len(self.RIR_ref))
        self.primary_buffer = np.zeros(len(self.P_z))
        self.secondary_buffer = np.zeros(len(self.S_z))
    
    def get_secondary_path_estimate(self) -> np.ndarray:
        """Get the imperfect secondary path estimate for FxLMS."""
        return self.S_z_hat.copy()

