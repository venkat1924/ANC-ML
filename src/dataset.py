"""
ANC Dataset: Audio data pipeline with physics-based augmentation.

Loads environmental noise recordings and applies augmentations that
simulate real-world acoustic variations:
- Gain: Microphone calibration variance
- Delay: Acoustic path length changes
- Leakage: Imperfect headphone seal
"""

import os
import glob
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.signal as signal

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from .utils import fractional_delay, generate_leakage_tf, db_to_linear


class ANCDataset(Dataset):
    """
    Dataset for ANC training.
    
    Loads audio files and applies physics-based augmentations.
    Returns (input, target) pairs where target is inverted future noise
    for delay-compensated training.
    """
    
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 48000,
        chunk_size: int = 16384,
        delay_compensation: int = 3,
        augment: bool = True,
        gain_range_db: Tuple[float, float] = (-6.0, 6.0),
        delay_range_samples: Tuple[float, float] = (-5.0, 5.0),
        leakage_probability: float = 0.3
    ):
        """
        Initialize ANC dataset.
        
        Args:
            root_dir: Directory containing .wav files (searched recursively)
            sample_rate: Target sample rate (48kHz recommended)
            chunk_size: Length of audio chunks in samples
            delay_compensation: K samples to shift target (for latency compensation)
            augment: Whether to apply augmentations
            gain_range_db: Random gain range in dB
            delay_range_samples: Random delay range in samples
            leakage_probability: Probability of applying leakage TF
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.K = delay_compensation
        self.augment = augment
        self.gain_range_db = gain_range_db
        self.delay_range_samples = delay_range_samples
        self.leakage_probability = leakage_probability
        
        # Find all audio files
        self.files = self._find_audio_files(root_dir)
        
        if len(self.files) == 0:
            print(f"Warning: No audio files found in {root_dir}")
            print("Dataset will generate synthetic noise for testing.")
            self.synthetic_mode = True
        else:
            self.synthetic_mode = False
            print(f"Found {len(self.files)} audio files in {root_dir}")
        
        # Pre-generate leakage transfer functions
        self.leakage_tfs = {
            "light": generate_leakage_tf(sample_rate, "light"),
            "medium": generate_leakage_tf(sample_rate, "medium"),
            "heavy": generate_leakage_tf(sample_rate, "heavy")
        }
    
    def _find_audio_files(self, root_dir: str) -> List[str]:
        """Recursively find all .wav files."""
        patterns = ["**/*.wav", "**/*.WAV", "**/*.flac", "**/*.mp3"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
        return sorted(files)
    
    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return (audio, sample_rate)."""
        if HAS_TORCHAUDIO:
            audio, sr = torchaudio.load(path)
            audio = audio.numpy()
        elif HAS_SOUNDFILE:
            audio, sr = sf.read(path)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            else:
                audio = audio.T  # (channels, samples)
        else:
            raise RuntimeError("Neither torchaudio nor soundfile available")
        
        return audio, sr
    
    def _generate_synthetic_noise(self) -> np.ndarray:
        """Generate synthetic noise for testing when no files available."""
        t = np.arange(self.chunk_size) / self.sample_rate
        
        # Mix of tones and broadband noise (simulates engine + wind)
        noise = np.zeros(self.chunk_size)
        
        # Engine harmonics
        for freq in [50, 100, 150, 200]:
            phase = random.uniform(0, 2 * np.pi)
            amp = random.uniform(0.1, 0.3)
            noise += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Broadband component
        noise += 0.2 * np.random.randn(self.chunk_size)
        
        # Random low-pass (simulates varying distance)
        cutoff = random.uniform(500, 2000)
        nyq = self.sample_rate / 2
        b, a = signal.butter(2, cutoff / nyq, btype='low')
        noise = signal.lfilter(b, a, noise)
        
        # Normalize
        noise = noise / (np.max(np.abs(noise)) + 1e-8) * 0.8
        
        return noise.astype(np.float32)
    
    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply physics-based augmentations."""
        if not self.augment:
            return audio
        
        # 1. Random gain (±6dB)
        gain_db = random.uniform(*self.gain_range_db)
        audio = audio * db_to_linear(gain_db)
        
        # 2. Random fractional delay (±5 samples)
        delay = random.uniform(*self.delay_range_samples)
        if abs(delay) > 0.1:
            audio = fractional_delay(audio, delay)
        
        # 3. Leakage transfer function (30% probability)
        if random.random() < self.leakage_probability:
            leak_type = random.choice(["light", "medium", "heavy"])
            leak_tf = self.leakage_tfs[leak_type]
            audio = signal.convolve(audio, leak_tf, mode='same')
        
        return audio
    
    def __len__(self) -> int:
        if self.synthetic_mode:
            return 1000  # Virtual dataset size for synthetic mode
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training sample.
        
        Returns:
            Tuple of (input, target) tensors
            - input: noise signal (1, T-K)
            - target: inverted future noise (1, T-K) for cancellation
        """
        if self.synthetic_mode:
            audio = self._generate_synthetic_noise()
        else:
            # Load audio file
            path = self.files[idx % len(self.files)]
            try:
                audio, orig_sr = self._load_audio(path)
                
                # Convert to mono
                if audio.shape[0] > 1:
                    audio = np.mean(audio, axis=0)
                else:
                    audio = audio[0]
                
                # Resample if needed
                if orig_sr != self.sample_rate:
                    num_samples = int(len(audio) * self.sample_rate / orig_sr)
                    audio = signal.resample(audio, num_samples)
                
                # Pad or crop to chunk_size
                if len(audio) < self.chunk_size:
                    pad = self.chunk_size - len(audio)
                    audio = np.pad(audio, (0, pad), mode='constant')
                elif len(audio) > self.chunk_size:
                    start = random.randint(0, len(audio) - self.chunk_size)
                    audio = audio[start:start + self.chunk_size]
                
                audio = audio.astype(np.float32)
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
                audio = self._generate_synthetic_noise()
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        # Apply augmentations
        audio = self._apply_augmentation(audio)
        
        # Create input/target pair with delay compensation
        # Input: noise[0:T-K]
        # Target: -noise[K:T] (inverted future for cancellation)
        K = self.K
        input_signal = audio[:-K] if K > 0 else audio
        target_signal = -audio[K:] if K > 0 else -audio
        
        # Convert to tensors (1, T-K)
        input_tensor = torch.from_numpy(input_signal).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_signal).float().unsqueeze(0)
        
        return input_tensor, target_tensor


class ANCDatasetStreaming(Dataset):
    """
    Streaming variant that yields overlapping chunks from long files.
    
    More efficient for training on large audio files.
    """
    
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 48000,
        chunk_size: int = 16384,
        hop_size: int = 8192,
        delay_compensation: int = 3,
        augment: bool = True
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.K = delay_compensation
        self.augment = augment
        
        # Load and concatenate all audio
        files = glob.glob(os.path.join(root_dir, "**/*.wav"), recursive=True)
        
        self.audio_data = []
        for f in files[:100]:  # Limit for memory
            try:
                if HAS_TORCHAUDIO:
                    audio, sr = torchaudio.load(f)
                    audio = audio.mean(dim=0).numpy()
                elif HAS_SOUNDFILE:
                    audio, sr = sf.read(f)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                else:
                    continue
                
                if sr != sample_rate:
                    num_samples = int(len(audio) * sample_rate / sr)
                    audio = signal.resample(audio, num_samples)
                
                self.audio_data.append(audio.astype(np.float32))
            except Exception:
                continue
        
        if self.audio_data:
            self.audio_data = np.concatenate(self.audio_data)
            self.n_chunks = (len(self.audio_data) - chunk_size) // hop_size
        else:
            self.audio_data = np.random.randn(sample_rate * 60).astype(np.float32)
            self.n_chunks = (len(self.audio_data) - chunk_size) // hop_size
    
    def __len__(self) -> int:
        return max(1, self.n_chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = (idx * self.hop_size) % (len(self.audio_data) - self.chunk_size)
        chunk = self.audio_data[start:start + self.chunk_size].copy()
        
        # Normalize
        max_val = np.max(np.abs(chunk))
        if max_val > 0:
            chunk = chunk / max_val * 0.9
        
        K = self.K
        input_signal = chunk[:-K] if K > 0 else chunk
        target_signal = -chunk[K:] if K > 0 else -chunk
        
        return (
            torch.from_numpy(input_signal).float().unsqueeze(0),
            torch.from_numpy(target_signal).float().unsqueeze(0)
        )

