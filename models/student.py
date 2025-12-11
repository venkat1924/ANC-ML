"""
Student model for ANC using Knowledge Distillation.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEQUENCE_LENGTH,
    SAMPLE_RATE,
    DURATION,
    STUDENT_FILTERS,
    STUDENT_LEARNING_RATE,
    STUDENT_EPOCHS,
    STUDENT_BATCH_SIZE,
    STUDENT_PATIENCE,
    STUDENT_NUM_WAVES,
    STUDENT_MODEL_PATH,
    DISTILLATION_ALPHA,
    DISTILLATION_STEPS_PER_EPOCH,
)
from data.loader import UrbanSoundLoader
from utils.audio import load_audio, play_audio
from utils.visualization import plot_results


def build_student_model(sequence_length=None, filters=None):
    """
    Build the compact student TCN model.

    Args:
        sequence_length: Input sequence length. If None, uses config.
        filters: Number of Conv1D filters. If None, uses config.

    Returns:
        Uncompiled Keras model (compiled during training with custom loss).
    """
    if sequence_length is None:
        sequence_length = SEQUENCE_LENGTH
    if filters is None:
        filters = STUDENT_FILTERS

    inp = layers.Input(shape=(sequence_length, 1))

    # Smaller architecture: 2 layers instead of 3, fewer filters
    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding='causal',
        activation='relu',
        dilation_rate=1
    )(inp)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding='causal',
        activation='relu',
        dilation_rate=2
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    return model


class DistilledStudentTCN:
    """
    Student model trained via knowledge distillation from teacher.
    """

    def __init__(self, teacher_model, data_loader=None):
        """
        Initialize the student model.

        Args:
            teacher_model: Trained teacher Keras model.
            data_loader: Optional UrbanSoundLoader instance. If None, creates one.
        """
        self.sequence_length = SEQUENCE_LENGTH
        self.sample_rate = SAMPLE_RATE
        self.duration = DURATION
        self.num_waves = STUDENT_NUM_WAVES
        self.epochs = STUDENT_EPOCHS
        self.batch_size = STUDENT_BATCH_SIZE
        self.learning_rate = STUDENT_LEARNING_RATE
        self.patience = STUDENT_PATIENCE
        self.alpha = DISTILLATION_ALPHA
        self.steps_per_epoch = DISTILLATION_STEPS_PER_EPOCH

        # Set device
        self._set_device()

        # Teacher model (frozen)
        self.teacher = teacher_model
        self.teacher.trainable = False

        # Build student model
        self.model = build_student_model()

        # Initialize data loader
        if data_loader is not None:
            self.data_loader = data_loader
        else:
            self.data_loader = UrbanSoundLoader()

    def _set_device(self):
        """Check and print available compute device."""
        if tf.config.list_physical_devices('GPU'):
            print("Using GPU for training.")
            self.device = 'GPU'
        else:
            print("Using CPU for training.")
            self.device = 'CPU'

    def _prepare_batch(self):
        """
        Generator that yields batches with combined targets.

        Yields:
            X_batch: Input sequences
            y_combined: Combined targets [ground_truth, teacher_prediction]
        """
        while True:
            X_batch = []
            y_true_batch = []
            y_teacher_batch = []

            for _ in range(self.batch_size):
                # Generate random waveform
                wave = self.data_loader.generate_waveform(avoid_duplicates=False)

                # Random position
                i = random.randint(0, len(wave) - self.sequence_length - 1)
                seq = wave[i:i + self.sequence_length]
                next_sample = wave[i + self.sequence_length]

                X_batch.append(seq)
                y_true_batch.append(-next_sample)  # Inverted for ANC

                # Get teacher prediction
                teacher_pred = self.teacher.predict(
                    np.array([seq]).reshape(1, self.sequence_length, 1),
                    verbose=0
                )[0][0]
                y_teacher_batch.append(teacher_pred)

            X_batch = np.array(X_batch).reshape(-1, self.sequence_length, 1)
            y_combined = np.array([y_true_batch, y_teacher_batch]).T

            yield X_batch, y_combined

    def _distillation_loss(self, y_combined, y_pred):
        """
        Custom distillation loss combining ground truth and teacher guidance.

        Args:
            y_combined: Combined targets [y_true, y_teacher]
            y_pred: Student predictions

        Returns:
            Weighted loss combining student and teacher losses.
        """
        y_true = y_combined[:, 0]
        y_teacher = y_combined[:, 1]

        student_loss = tf.reduce_mean(tf.square(y_true - tf.squeeze(y_pred)))
        teacher_loss = tf.reduce_mean(tf.square(y_teacher - tf.squeeze(y_pred)))

        return (1 - self.alpha) * student_loss + self.alpha * teacher_loss

    def train(self):
        """
        Train the student model via distillation.

        Returns:
            Training history.
        """
        train_gen = self._prepare_batch()

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss=self._distillation_loss,
            metrics=['mse']
        )

        # Memory monitoring callback
        def print_memory(epoch, logs):
            try:
                import psutil
                print(f"Memory: {psutil.virtual_memory().percent}% used")
            except ImportError:
                pass

        history = self.model.fit(
            train_gen,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            callbacks=[
                EarlyStopping(patience=self.patience),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=print_memory)
            ]
        )

        return history

    def predict(self, test_waveform):
        """
        Predict anti-phase waveform for input.

        Args:
            test_waveform: Input waveform array.

        Returns:
            Predicted anti-phase waveform.
        """
        seqs = [
            test_waveform[i:i + self.sequence_length]
            for i in range(len(test_waveform) - self.sequence_length)
        ]
        X = np.array(seqs).reshape(-1, self.sequence_length, 1)

        return self.model.predict(X).flatten()

    def test_with_wav(self, wav_path):
        """
        Test the model with a WAV file.

        Args:
            wav_path: Path to WAV file.
        """
        # Load and normalize test waveform
        test_waveform = load_audio(wav_path, sr=self.sample_rate)

        # Predict
        predicted_waveform = self.predict(test_waveform)

        # Calculate combined waveform
        combined_waveform = test_waveform[self.sequence_length:] + predicted_waveform

        # Calculate MSE to silence
        mse = mean_squared_error(np.zeros_like(combined_waveform), combined_waveform)
        print(f'MSE to silence: {mse}')

        # Plot results
        plot_results(test_waveform, predicted_waveform, combined_waveform)

        # Play audio
        play_audio(test_waveform, self.sample_rate, "Test Input Waveform:")
        play_audio(predicted_waveform, self.sample_rate, "Predicted Inverted Waveform:")
        play_audio(combined_waveform, self.sample_rate, "Combined Waveform:")

    def save_model(self, path=None):
        """
        Save the model to disk.

        Args:
            path: Save path. If None, uses config STUDENT_MODEL_PATH.
        """
        if path is None:
            path = STUDENT_MODEL_PATH

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """
        Load a model from disk.

        Args:
            path: Load path. If None, uses config STUDENT_MODEL_PATH.
        """
        if path is None:
            path = STUDENT_MODEL_PATH

        self.model = models.load_model(path, custom_objects={
            '_distillation_loss': self._distillation_loss
        })
        print(f"Model loaded from {path}")

    def summary(self):
        """Print model summary."""
        self.model.summary()

