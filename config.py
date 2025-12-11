"""
Configuration file for ANC ML Project.
All hyperparameters, paths, and constants are defined here.
"""

import os

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "urbansound8k")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
TEACHER_MODEL_PATH = os.path.join(MODEL_DIR, "tcn_noise_cancellation.keras")
STUDENT_MODEL_PATH = os.path.join(MODEL_DIR, "distilled_student.keras")

# ==============================================================================
# Audio Parameters
# ==============================================================================
SAMPLE_RATE = 16000          # Samples per second
DURATION = 1.0               # Duration in seconds
SEQUENCE_LENGTH = 50         # Number of previous samples for prediction

# ==============================================================================
# Dataset Parameters
# ==============================================================================
# UrbanSound8K class IDs to use for training
# 0: air_conditioner, 1: car_horn, 4: engine_idling,
# 5: gun_shot, 7: siren, 8: street_music
ALLOWED_CLASSES = [0, 1, 4, 5, 7, 8]

# ==============================================================================
# Teacher Model Parameters
# ==============================================================================
TEACHER_NUM_WAVES = 5        # Number of waveforms for teacher training
TEACHER_EPOCHS = 1           # Training epochs for teacher
TEACHER_BATCH_SIZE = 32      # Batch size for teacher
TEACHER_LEARNING_RATE = 0.001
TEACHER_FILTERS = 64         # Number of filters in Conv1D layers
TEACHER_PATIENCE = 5         # Early stopping patience

# ==============================================================================
# Student Model Parameters
# ==============================================================================
STUDENT_NUM_WAVES = 50       # Number of waveforms for student training
STUDENT_EPOCHS = 20          # Training epochs for student
STUDENT_BATCH_SIZE = 16      # Batch size for student
STUDENT_LEARNING_RATE = 0.0005
STUDENT_FILTERS = 32         # Number of filters in Conv1D layers (smaller)
STUDENT_PATIENCE = 3         # Early stopping patience

# ==============================================================================
# Knowledge Distillation Parameters
# ==============================================================================
DISTILLATION_ALPHA = 0.3     # Weight for teacher loss (0.3 = 30% teacher, 70% ground truth)
DISTILLATION_STEPS_PER_EPOCH = 100

# ==============================================================================
# Visualization Parameters
# ==============================================================================
ZOOM_DURATION = 0.05         # 50ms zoom window for plots

