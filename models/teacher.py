"""
Teacher model for ANC using Temporal Convolutional Networks (TCN).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEQUENCE_LENGTH,
    SAMPLE_RATE,
    TEACHER_FILTERS,
    TEACHER_LEARNING_RATE,
    TEACHER_EPOCHS,
    TEACHER_BATCH_SIZE,
    TEACHER_PATIENCE,
    TEACHER_NUM_WAVES,
    TEACHER_MODEL_PATH,
)
from data.loader import UrbanSoundLoader
from utils.audio import load_audio, play_audio
from utils.visualization import plot_results


def build_teacher_model(sequence_length=None, filters=None, learning_rate=None):
    """
    Build the teacher TCN model.

    Args:
        sequence_length: Input sequence length. If None, uses config.
        filters: Number of Conv1D filters. If None, uses config.
        learning_rate: Learning rate. If None, uses config.

    Returns:
        Compiled Keras model.
    """
    if sequence_length is None:
        sequence_length = SEQUENCE_LENGTH
    if filters is None:
        filters = TEACHER_FILTERS
    if learning_rate is None:
        learning_rate = TEACHER_LEARNING_RATE

    input_layer = layers.Input(shape=(sequence_length, 1))

    # First TCN block
    tcn = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding='causal',
        activation='relu',
        dilation_rate=1
    )(input_layer)
    tcn = layers.BatchNormalization()(tcn)

    tcn = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding='causal',
        activation='relu',
        dilation_rate=2
    )(tcn)
    tcn = layers.BatchNormalization()(tcn)

    # Second TCN block with increased dilation
    tcn = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding='causal',
        activation='relu',
        dilation_rate=4
    )(tcn)
    tcn = layers.BatchNormalization()(tcn)

    # Output layers
    flatten = layers.Flatten()(tcn)
    output_layer = layers.Dense(1)(flatten)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    return model


class WaveformPredictorTCN:
    """
    Teacher model class for waveform prediction / active noise cancellation.
    """

    def __init__(self, data_loader=None):
        """
        Initialize the teacher model.

        Args:
            data_loader: Optional UrbanSoundLoader instance. If None, creates one.
        """
        self.sequence_length = SEQUENCE_LENGTH
        self.sample_rate = SAMPLE_RATE
        self.num_waves = TEACHER_NUM_WAVES
        self.epochs = TEACHER_EPOCHS
        self.batch_size = TEACHER_BATCH_SIZE
        self.patience = TEACHER_PATIENCE

        # Set device
        self._set_device()

        # Build model
        self.model = build_teacher_model()

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

    def train(self):
        """
        Train the model on prepared data.

        Returns:
            Training history.
        """
        X, y = self.data_loader.prepare_sequences(self.num_waves, invert_target=True)

        early_stopping = EarlyStopping(
            monitor='loss',
            patience=self.patience,
            restore_best_weights=True
        )

        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=[early_stopping]
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
        X_test = []
        for i in range(len(test_waveform) - self.sequence_length):
            X_test.append(test_waveform[i:i + self.sequence_length])

        X_test = np.array(X_test).reshape(len(X_test), self.sequence_length, 1)
        predicted_waveform = self.model.predict(X_test)

        return predicted_waveform.flatten()

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
        silence_target = np.zeros_like(combined_waveform)
        validation_loss = mean_squared_error(silence_target, combined_waveform)
        print(f'Combined to Zero Validation Loss (MSE): {validation_loss}')

        # Plot results
        plot_results(test_waveform, predicted_waveform, combined_waveform)

        # Play audio
        play_audio(test_waveform, self.sample_rate, "Test Input Waveform:")
        play_audio(predicted_waveform, self.sample_rate, "Predicted Inverted Waveform:")
        play_audio(combined_waveform, self.sample_rate, "Combined Waveform:")

    def save_model(self, path=None, include_optimizer=False):
        """
        Save the model to disk.

        Args:
            path: Save path. If None, uses config TEACHER_MODEL_PATH.
            include_optimizer: Whether to include optimizer state.
        """
        if path is None:
            path = TEACHER_MODEL_PATH

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save(path, include_optimizer=include_optimizer)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """
        Load a model from disk.

        Args:
            path: Load path. If None, uses config TEACHER_MODEL_PATH.
        """
        if path is None:
            path = TEACHER_MODEL_PATH

        self.model = models.load_model(path)
        print(f"Model loaded from {path}")

    def summary(self):
        """Print model summary."""
        self.model.summary()

