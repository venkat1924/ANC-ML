#!/usr/bin/env python3
"""
Train the student TCN model via knowledge distillation.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf

from config import TEACHER_MODEL_PATH, STUDENT_MODEL_PATH
from data.loader import UrbanSoundLoader
from models.student import DistilledStudentTCN


def main():
    print("=" * 60)
    print("Training Student Model via Knowledge Distillation")
    print("=" * 60)

    # Load teacher model
    print(f"\nLoading teacher model from: {TEACHER_MODEL_PATH}")
    if not os.path.exists(TEACHER_MODEL_PATH):
        print("ERROR: Teacher model not found. Please train the teacher first:")
        print("  python train_teacher.py")
        sys.exit(1)

    teacher = tf.keras.models.load_model(TEACHER_MODEL_PATH)
    print("Teacher model loaded successfully.")

    # Initialize data loader
    print("\nInitializing data loader...")
    data_loader = UrbanSoundLoader()

    # Initialize student model
    print("\nBuilding student model...")
    student = DistilledStudentTCN(teacher_model=teacher, data_loader=data_loader)
    student.summary()

    # Train via distillation
    print("\nStarting distillation training...")
    history = student.train()

    # Evaluate on random test waveform
    print("\nEvaluating on random test waveform...")
    test_waveform = data_loader.generate_waveform()
    predicted_waveform = student.predict(test_waveform)

    # Calculate combined waveform
    combined_waveform = test_waveform[student.sequence_length:] + predicted_waveform

    # Plot results
    from utils.visualization import plot_results
    plot_results(test_waveform, predicted_waveform, combined_waveform)

    # Play audio (only works in Jupyter/IPython)
    try:
        from utils.audio import play_audio
        play_audio(test_waveform, student.sample_rate, "Test Input Waveform:")
        play_audio(predicted_waveform, student.sample_rate, "Predicted Inverted Waveform:")
        play_audio(combined_waveform, student.sample_rate, "Combined Waveform:")
    except Exception as e:
        print(f"Audio playback not available: {e}")

    # Save model
    print("\nSaving model...")
    student.save_model()

    print("\n" + "=" * 60)
    print("Distillation training complete!")
    print(f"Model saved to: {STUDENT_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

