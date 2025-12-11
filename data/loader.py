"""
Data loading utilities for UrbanSound8K dataset.
"""

import random
import numpy as np
import librosa
import soundata

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SAMPLE_RATE,
    DURATION,
    SEQUENCE_LENGTH,
    ALLOWED_CLASSES,
    DATA_DIR,
)


class UrbanSoundLoader:
    """
    Handles loading and preprocessing of UrbanSound8K dataset.
    """

    def __init__(self, data_home=None):
        """
        Initialize the dataset loader.

        Args:
            data_home: Optional path to dataset. If None, uses soundata default.
        """
        self.sample_rate = SAMPLE_RATE
        self.duration = DURATION
        self.sequence_length = SEQUENCE_LENGTH
        self.allowed_classes = ALLOWED_CLASSES

        # Initialize dataset
        if data_home:
            self.dataset = soundata.initialize('urbansound8k', data_home=data_home)
        else:
            self.dataset = soundata.initialize('urbansound8k')

        # Load and filter clips
        clips = self.dataset.load_clips()
        self.filtered_clips = {
            clip_id: clip
            for clip_id, clip in clips.items()
            if clip.class_id in self.allowed_classes
        }

        print(f"Loaded {len(self.filtered_clips)} clips from classes {self.allowed_classes}")

        # Track used waveforms to avoid duplicates
        self.used_wave_hashes = set()

    def download_dataset(self):
        """Download the dataset if not present."""
        self.dataset.download()
        self.dataset.validate()

    def generate_waveform(self, avoid_duplicates=True):
        """
        Generate a normalized waveform from a random clip.

        Args:
            avoid_duplicates: If True, avoid returning previously returned waveforms.

        Returns:
            Normalized waveform array of shape (num_samples,)
        """
        # Select a random clip
        random_clip_id = random.choice(list(self.filtered_clips.keys()))
        random_clip = self.filtered_clips[random_clip_id]

        # Load audio
        audio_signal, _ = librosa.load(random_clip.audio_path, sr=self.sample_rate)

        # Extract desired duration
        num_samples = int(self.duration * self.sample_rate)
        extracted_audio = audio_signal[:num_samples]

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(extracted_audio))
        if max_val > 0:
            normalized_wave = extracted_audio / max_val
        else:
            normalized_wave = extracted_audio

        # Avoid duplicates if requested
        if avoid_duplicates:
            wave_hash = hash(tuple(normalized_wave))
            if wave_hash in self.used_wave_hashes:
                return self.generate_waveform(avoid_duplicates=True)
            self.used_wave_hashes.add(wave_hash)

        return normalized_wave

    def prepare_sequences(self, num_waves, invert_target=True):
        """
        Prepare training sequences from multiple waveforms.

        Args:
            num_waves: Number of waveforms to generate.
            invert_target: If True, target is inverted (for ANC). Default True.

        Returns:
            X: Input sequences of shape (num_samples, sequence_length, 1)
            y: Target values of shape (num_samples,)
        """
        X, y = [], []

        for _ in range(num_waves):
            waveform = self.generate_waveform()

            for i in range(len(waveform) - self.sequence_length):
                X.append(waveform[i:i + self.sequence_length])
                target = waveform[i + self.sequence_length]
                if invert_target:
                    target = -target  # Inverted phase for ANC
                y.append(target)

        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)

        return X, y

    def reset_duplicates(self):
        """Clear the duplicate tracking set."""
        self.used_wave_hashes.clear()

