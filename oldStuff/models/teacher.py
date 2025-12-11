"""
Teacher model for ANC using Temporal Convolutional Networks (TCN).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

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


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution layer.
    Pads only on the left side to ensure no information leakage from the future.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0
        )

    def forward(self, x):
        # Pad left side only for causal convolution
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TeacherTCN(nn.Module):
    """
    Teacher TCN model for waveform prediction.
    Architecture: 3 causal conv blocks with increasing dilation rates.
    """

    def __init__(self, sequence_length=None, filters=None):
        super().__init__()

        if sequence_length is None:
            sequence_length = SEQUENCE_LENGTH
        if filters is None:
            filters = TEACHER_FILTERS

        self.sequence_length = sequence_length
        self.filters = filters

        # First TCN block (dilation=1)
        self.conv1 = CausalConv1d(1, filters, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm1d(filters)

        # Second TCN block (dilation=2)
        self.conv2 = CausalConv1d(filters, filters, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(filters)

        # Third TCN block (dilation=4)
        self.conv3 = CausalConv1d(filters, filters, kernel_size=3, dilation=4)
        self.bn3 = nn.BatchNorm1d(filters)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(filters * sequence_length, 1)

    def forward(self, x):
        # x shape: (batch, sequence_length, 1) -> need (batch, 1, sequence_length)
        x = x.permute(0, 2, 1)

        # TCN blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Output
        x = self.flatten(x)
        x = self.fc(x)

        return x


def build_teacher_model(sequence_length=None, filters=None):
    """
    Build the teacher TCN model.

    Args:
        sequence_length: Input sequence length. If None, uses config.
        filters: Number of Conv1D filters. If None, uses config.

    Returns:
        TeacherTCN model instance.
    """
    return TeacherTCN(sequence_length=sequence_length, filters=filters)


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
        self.learning_rate = TEACHER_LEARNING_RATE

        # Set device
        self._set_device()

        # Build model
        self.model = build_teacher_model()
        self.model.to(self.device)

        # Initialize data loader
        if data_loader is not None:
            self.data_loader = data_loader
        else:
            self.data_loader = UrbanSoundLoader()

    def _set_device(self):
        """Check and set available compute device."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using GPU (CUDA) for training.")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using GPU (MPS) for training.")
        else:
            self.device = torch.device('cpu')
            print("Using CPU for training.")

    def train(self):
        """
        Train the model on prepared data.

        Returns:
            Dictionary containing training history.
        """
        X, y = self.data_loader.prepare_sequences(self.num_waves, invert_target=True)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        history = {'loss': []}
        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.epochs} - loss: {avg_loss:.6f}")

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model state
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    # Restore best weights
                    self.model.load_state_dict(self._best_state)
                    break

        return history

    def predict(self, test_waveform):
        """
        Predict anti-phase waveform for input.

        Args:
            test_waveform: Input waveform array.

        Returns:
            Predicted anti-phase waveform.
        """
        self.model.eval()
        X_test = []
        for i in range(len(test_waveform) - self.sequence_length):
            X_test.append(test_waveform[i:i + self.sequence_length])

        X_test = np.array(X_test).reshape(len(X_test), self.sequence_length, 1)
        X_tensor = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            predicted_waveform = self.model(X_tensor).cpu().numpy()

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

    def save_model(self, path=None):
        """
        Save the model to disk.

        Args:
            path: Save path. If None, uses config TEACHER_MODEL_PATH.
        """
        if path is None:
            path = TEACHER_MODEL_PATH

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'filters': self.model.filters,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """
        Load a model from disk.

        Args:
            path: Load path. If None, uses config TEACHER_MODEL_PATH.
        """
        if path is None:
            path = TEACHER_MODEL_PATH

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def summary(self):
        """Print model summary."""
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
