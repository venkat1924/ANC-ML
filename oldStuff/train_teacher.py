#!/usr/bin/env python3
"""
Train the teacher TCN model for active noise cancellation.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TEACHER_MODEL_PATH
from data.loader import UrbanSoundLoader
from models.teacher import WaveformPredictorTCN


def main():
    print("=" * 60)
    print("Training Teacher Model")
    print("=" * 60)

    # Initialize data loader
    print("\nInitializing data loader...")
    data_loader = UrbanSoundLoader()

    # Initialize model
    print("\nBuilding teacher model...")
    predictor = WaveformPredictorTCN(data_loader=data_loader)
    predictor.summary()

    # Train
    print("\nStarting training...")
    history = predictor.train()

    # Generate test waveform and evaluate
    print("\nEvaluating on random test waveform...")
    test_waveform = data_loader.generate_waveform()
    predicted_waveform = predictor.predict(test_waveform)

    # Calculate combined waveform
    combined_waveform = test_waveform[predictor.sequence_length:] + predicted_waveform

    # Plot results
    from utils.visualization import plot_results
    plot_results(test_waveform, predicted_waveform, combined_waveform)

    # Play audio (only works in Jupyter/IPython)
    try:
        from utils.audio import play_audio
        play_audio(test_waveform, predictor.sample_rate, "Test Input Waveform:")
        play_audio(predicted_waveform, predictor.sample_rate, "Predicted Inverted Waveform:")
        play_audio(combined_waveform, predictor.sample_rate, "Combined Waveform:")
    except Exception as e:
        print(f"Audio playback not available: {e}")

    # Save model
    print("\nSaving model...")
    predictor.save_model()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {TEACHER_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

