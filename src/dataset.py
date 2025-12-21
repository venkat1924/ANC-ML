"""
DNS Dataset loader for ANC training.

Loads audio from Microsoft DNS Challenge noise data and applies
physics-based augmentation for ANC training.

Key features:
- Loads .wav files from DNS dataset (or synthetic)
- Resamples to target sample rate (48kHz)
- Chunks audio into training segments
- Applies physics-based augmentation:
  - Gain variation (mic calibration)
  - Fractional delay (distance changes)
  - Leakage filter (imperfect seal)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
from typing import Optional, Tuple, List
import random

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class DNSDataset(Dataset):
    """
    Dataset for ANC training using DNS Challenge noise data.
    
    Generates (input, target) pairs where:
    - input: noise[0:T-K] (current noise)
    - target: -noise[K:T] (inverted future for cancellation)
    
    K is the delay compensation in samples.
    """
    
    def __init__(
        self,
        data_dir: str,
        chunk_length: int = 16384,
        K: int = 3,
        fs: int = 48000,
        augment: bool = True,
        augment_prob: float = 0.5,
        max_files: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize DNS dataset.
        
        Args:
            data_dir: Path to directory containing .wav files
            chunk_length: Length of audio chunks in samples
            K: Delay compensation in samples (~60µs at 48kHz)
            fs: Target sample rate
            augment: Whether to apply augmentation
            augment_prob: Probability of applying each augmentation
            max_files: Maximum number of files to load (for testing)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.chunk_length = chunk_length
        self.K = K
        self.fs = fs
        self.augment = augment
        self.augment_prob = augment_prob
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Find all audio files
        self.audio_files = self._find_audio_files(max_files)
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
        
        # Build index: (file_idx, chunk_start)
        self.chunks = self._build_chunk_index()
        
        print(f"DNSDataset: {len(self.audio_files)} files, {len(self.chunks)} chunks")
    
    def _find_audio_files(self, max_files: Optional[int]) -> List[Path]:
        """Find all .wav files in data directory."""
        extensions = ['.wav', '.WAV', '.flac', '.FLAC']
        files = []
        
        for ext in extensions:
            files.extend(self.data_dir.rglob(f'*{ext}'))
        
        files = sorted(files)
        
        if max_files is not None:
            files = files[:max_files]
        
        return files
    
    def _build_chunk_index(self) -> List[Tuple[int, int]]:
        """Build index of (file_idx, chunk_start) tuples."""
        chunks = []
        
        for file_idx, audio_path in enumerate(self.audio_files):
            try:
                # Get file duration without loading full audio
                if HAS_SOUNDFILE:
                    info = sf.info(audio_path)
                    n_samples = int(info.frames * self.fs / info.samplerate)
                else:
                    # Fallback: load and check
                    sr, audio = wavfile.read(audio_path)
                    n_samples = int(len(audio) * self.fs / sr)
                
                # Create chunks with 50% overlap
                stride = self.chunk_length // 2
                for start in range(0, max(1, n_samples - self.chunk_length), stride):
                    chunks.append((file_idx, start))
                    
            except Exception as e:
                print(f"Warning: Could not index {audio_path}: {e}")
        
        return chunks
    
    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load and resample audio file."""
        try:
            if HAS_LIBROSA:
                # Librosa handles resampling automatically
                audio, _ = librosa.load(file_path, sr=self.fs, mono=True)
            elif HAS_SOUNDFILE:
                audio, sr = sf.read(file_path, always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != self.fs:
                    # Simple resampling using scipy
                    n_samples = int(len(audio) * self.fs / sr)
                    audio = signal.resample(audio, n_samples)
            else:
                sr, audio = wavfile.read(file_path)
                # Convert to float
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != self.fs:
                    n_samples = int(len(audio) * self.fs / sr)
                    audio = signal.resample(audio, n_samples)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros(self.chunk_length, dtype=np.float32)
    
    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply physics-based augmentation.
        
        Augmentations simulate real-world variations:
        - Gain: Microphone calibration differences (±6dB)
        - Delay: Small distance changes (±5 samples)
        - Leakage: Imperfect headphone seal
        """
        if not self.augment:
            return audio
        
        # Gain variation (±6dB)
        if random.random() < self.augment_prob:
            gain_db = random.uniform(-6, 6)
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
        
        # Fractional delay (±5 samples)
        if random.random() < self.augment_prob:
            delay = random.uniform(-5, 5)
            audio = self._fractional_delay(audio, delay)
        
        # Leakage filter (30% probability)
        if random.random() < 0.3:
            audio = self._apply_leakage(audio)
        
        return audio
    
    def _fractional_delay(self, sig: np.ndarray, delay: float) -> np.ndarray:
        """Apply fractional sample delay via sinc interpolation."""
        if abs(delay) < 0.01:
            return sig
        
        n_taps = 31
        half_taps = n_taps // 2
        n = np.arange(n_taps) - half_taps
        
        frac = delay - int(delay)
        kernel = np.sinc(n - frac) * np.hamming(n_taps)
        kernel /= np.sum(kernel)
        
        int_delay = int(delay)
        if int_delay > 0:
            sig = np.concatenate([np.zeros(int_delay), sig[:-int_delay]])
        elif int_delay < 0:
            sig = np.concatenate([sig[-int_delay:], np.zeros(-int_delay)])
        
        return np.convolve(sig, kernel, mode='same')
    
    def _apply_leakage(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass leakage filter (imperfect seal simulation)."""
        cutoff = random.uniform(80, 200)
        nyq = self.fs / 2
        
        b, a = signal.butter(2, cutoff / nyq, btype='high')
        leaked = signal.lfilter(b, a, audio)
        
        # Mix leaked signal with original
        leak_amount = random.uniform(0.1, 0.3)
        return audio * (1 - leak_amount) + leaked * leak_amount
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            input_tensor: Shape (1, T-K) - current noise
            target_tensor: Shape (1, T-K) - inverted future noise
        """
        file_idx, chunk_start = self.chunks[idx]
        audio_path = self.audio_files[file_idx]
        
        # Load audio
        audio = self._load_audio(audio_path)
        
        # Extract chunk
        chunk_end = chunk_start + self.chunk_length + self.K
        if chunk_end > len(audio):
            # Pad if needed
            audio = np.pad(audio, (0, chunk_end - len(audio)))
        
        audio = audio[chunk_start:chunk_end]
        
        # Apply augmentation
        audio = self._apply_augmentation(audio)
        
        # Create input/target pairs with delay compensation
        # Input: noise[0:T-K]
        # Target: -noise[K:T] (inverted future for cancellation)
        K = self.K
        if K > 0:
            input_signal = audio[:-K]
            target_signal = -audio[K:]  # Inverted for cancellation
        else:
            input_signal = audio
            target_signal = -audio
        
        # Normalize
        max_val = max(np.max(np.abs(input_signal)), 1e-8)
        input_signal = input_signal / max_val
        target_signal = target_signal / max_val
        
        # Convert to tensors with channel dimension
        input_tensor = torch.from_numpy(input_signal).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_signal).float().unsqueeze(0)
        
        return input_tensor, target_tensor


class SyntheticNoiseDataset(Dataset):
    """
    Generate synthetic noise for testing without real data.
    
    Creates various noise types:
    - Engine harmonics
    - Broadband (traffic, HVAC)
    - Wind gusts
    - Mixed
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        chunk_length: int = 16384,
        K: int = 3,
        fs: int = 48000,
        seed: Optional[int] = None
    ):
        self.n_samples = n_samples
        self.chunk_length = chunk_length
        self.K = K
        self.fs = fs
        
        if seed is not None:
            np.random.seed(seed)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random noise type
        noise_type = idx % 5
        t = np.arange(self.chunk_length + self.K) / self.fs
        
        if noise_type == 0:
            # Engine harmonics
            audio = self._generate_engine(t)
        elif noise_type == 1:
            # Broadband traffic
            audio = self._generate_broadband(t, cutoff=200)
        elif noise_type == 2:
            # Fan/HVAC
            audio = self._generate_tonal(t, base_freq=120)
        elif noise_type == 3:
            # Wind
            audio = self._generate_broadband(t, cutoff=100) * self._generate_envelope(t)
        else:
            # Mixed
            audio = 0.5 * self._generate_engine(t) + 0.5 * self._generate_broadband(t, 300)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Create input/target
        K = self.K
        input_signal = audio[:-K] if K > 0 else audio
        target_signal = -audio[K:] if K > 0 else -audio
        
        input_tensor = torch.from_numpy(input_signal).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_signal).float().unsqueeze(0)
        
        return input_tensor, target_tensor
    
    def _generate_engine(self, t: np.ndarray) -> np.ndarray:
        base_freq = np.random.uniform(40, 60)
        audio = np.zeros(len(t))
        for h in range(1, 7):
            amp = 0.3 / h
            phase = np.random.uniform(0, 2 * np.pi)
            audio += amp * np.sin(2 * np.pi * base_freq * h * t + phase)
        return audio
    
    def _generate_broadband(self, t: np.ndarray, cutoff: float) -> np.ndarray:
        audio = np.random.randn(len(t))
        b, a = signal.butter(3, cutoff / (self.fs / 2), btype='low')
        return signal.filtfilt(b, a, audio)
    
    def _generate_tonal(self, t: np.ndarray, base_freq: float) -> np.ndarray:
        audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
        audio += 0.25 * np.sin(2 * np.pi * base_freq * 2 * t)
        audio += np.random.randn(len(t)) * 0.1
        return audio
    
    def _generate_envelope(self, t: np.ndarray) -> np.ndarray:
        # Wind gust envelope
        return 1 + 0.5 * np.sin(2 * np.pi * 0.2 * t) * np.sin(2 * np.pi * 0.05 * t)

