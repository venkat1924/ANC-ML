#!/usr/bin/env python3
"""
Evaluate trained ANC models on audio files.
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from config import (
    TEACHER_MODEL_PATH,
    STUDENT_MODEL_PATH,
    SAMPLE_RATE,
    SEQUENCE_LENGTH,
    TEACHER_FILTERS,
    STUDENT_FILTERS,
)
from models.teacher import TeacherTCN
from models.student import StudentTCN
from utils.audio import load_audio, play_audio
from utils.visualization import plot_results


def evaluate_model(model, wav_path, model_name="Model", device=None):
    """
    Evaluate a model on a WAV file.

    Args:
        model: PyTorch model to evaluate.
        wav_path: Path to WAV file.
        model_name: Name for display purposes.
        device: Torch device to use.
    """
    if device is None:
        device = torch.device('cpu')

    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 60}")

    # Load audio
    print(f"Loading: {wav_path}")
    test_waveform = load_audio(wav_path, sr=SAMPLE_RATE)
    print(f"Loaded {len(test_waveform)} samples ({len(test_waveform)/SAMPLE_RATE:.2f}s)")

    # Create sequences
    X_test = []
    for i in range(len(test_waveform) - SEQUENCE_LENGTH):
        X_test.append(test_waveform[i:i + SEQUENCE_LENGTH])
    X_test = np.array(X_test).reshape(-1, SEQUENCE_LENGTH, 1)
    X_tensor = torch.FloatTensor(X_test).to(device)

    # Predict
    print("Predicting anti-phase waveform...")
    model.eval()
    with torch.no_grad():
        predicted_waveform = model(X_tensor).cpu().numpy().flatten()

    # Calculate combined waveform
    combined_waveform = test_waveform[SEQUENCE_LENGTH:] + predicted_waveform

    # Calculate metrics
    silence_target = np.zeros_like(combined_waveform)
    mse = mean_squared_error(silence_target, combined_waveform)

    # Calculate noise reduction in dB
    original_power = np.mean(test_waveform[SEQUENCE_LENGTH:] ** 2)
    residual_power = np.mean(combined_waveform ** 2)
    if residual_power > 0:
        reduction_db = 10 * np.log10(original_power / residual_power)
    else:
        reduction_db = float('inf')

    print(f"\nResults:")
    print(f"  MSE to silence: {mse:.6f}")
    print(f"  Noise reduction: {reduction_db:.2f} dB")

    # Plot results
    plot_results(test_waveform, predicted_waveform, combined_waveform)

    # Play audio
    try:
        play_audio(test_waveform, SAMPLE_RATE, "Original Audio:")
        play_audio(predicted_waveform, SAMPLE_RATE, "Predicted Anti-phase:")
        play_audio(combined_waveform, SAMPLE_RATE, "Combined (Cancelled):")
    except Exception as e:
        print(f"Audio playback not available: {e}")

    return mse, reduction_db


def load_model(model_path, model_type, device):
    """
    Load a PyTorch model from checkpoint.

    Args:
        model_path: Path to model checkpoint.
        model_type: 'teacher' or 'student'.
        device: Torch device to load to.

    Returns:
        Loaded model.
    """
    checkpoint = torch.load(model_path, map_location=device)

    if model_type == 'teacher':
        model = TeacherTCN(
            sequence_length=checkpoint.get('sequence_length', SEQUENCE_LENGTH),
            filters=checkpoint.get('filters', TEACHER_FILTERS)
        )
    else:
        model = StudentTCN(
            sequence_length=checkpoint.get('sequence_length', SEQUENCE_LENGTH),
            filters=checkpoint.get('filters', STUDENT_FILTERS)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ANC models on audio files."
    )
    parser.add_argument(
        "wav_path",
        help="Path to WAV file to evaluate"
    )
    parser.add_argument(
        "--model",
        choices=["teacher", "student", "both"],
        default="teacher",
        help="Which model to use (default: teacher)"
    )
    parser.add_argument(
        "--teacher-path",
        default=TEACHER_MODEL_PATH,
        help=f"Path to teacher model (default: {TEACHER_MODEL_PATH})"
    )
    parser.add_argument(
        "--student-path",
        default=STUDENT_MODEL_PATH,
        help=f"Path to student model (default: {STUDENT_MODEL_PATH})"
    )

    args = parser.parse_args()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Check WAV file exists
    if not os.path.exists(args.wav_path):
        print(f"ERROR: WAV file not found: {args.wav_path}")
        sys.exit(1)

    results = {}

    # Evaluate teacher
    if args.model in ["teacher", "both"]:
        if not os.path.exists(args.teacher_path):
            print(f"ERROR: Teacher model not found: {args.teacher_path}")
            print("Train it first with: python train_teacher.py")
            if args.model == "teacher":
                sys.exit(1)
        else:
            teacher = load_model(args.teacher_path, 'teacher', device)
            mse, db = evaluate_model(teacher, args.wav_path, "Teacher Model", device)
            results["teacher"] = {"mse": mse, "reduction_db": db}

    # Evaluate student
    if args.model in ["student", "both"]:
        if not os.path.exists(args.student_path):
            print(f"ERROR: Student model not found: {args.student_path}")
            print("Train it first with: python train_student.py")
            if args.model == "student":
                sys.exit(1)
        else:
            student = load_model(args.student_path, 'student', device)
            mse, db = evaluate_model(student, args.wav_path, "Student Model", device)
            results["student"] = {"mse": mse, "reduction_db": db}

    # Compare if both models evaluated
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("Comparison")
        print("=" * 60)
        print(f"{'Model':<15} {'MSE':<12} {'Reduction (dB)':<15}")
        print("-" * 42)
        for name, r in results.items():
            print(f"{name:<15} {r['mse']:<12.6f} {r['reduction_db']:<15.2f}")


if __name__ == "__main__":
    main()
