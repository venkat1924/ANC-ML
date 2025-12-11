"""
Student model for ANC using Knowledge Distillation.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

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
from models.teacher import CausalConv1d


class StudentTCN(nn.Module):
    """
    Student TCN model for waveform prediction.
    Smaller architecture: 2 causal conv blocks with fewer filters.
    """

    def __init__(self, sequence_length=None, filters=None):
        super().__init__()

        if sequence_length is None:
            sequence_length = SEQUENCE_LENGTH
        if filters is None:
            filters = STUDENT_FILTERS

        self.sequence_length = sequence_length
        self.filters = filters

        # First TCN block (dilation=1)
        self.conv1 = CausalConv1d(1, filters, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm1d(filters)

        # Second TCN block (dilation=2)
        self.conv2 = CausalConv1d(filters, filters, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(filters)

        # Output layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(filters * sequence_length, 1)

    def forward(self, x):
        # x shape: (batch, sequence_length, 1) -> need (batch, 1, sequence_length)
        x = x.permute(0, 2, 1)

        # TCN blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Output
        x = self.flatten(x)
        x = self.fc(x)

        return x


def build_student_model(sequence_length=None, filters=None):
    """
    Build the compact student TCN model.

    Args:
        sequence_length: Input sequence length. If None, uses config.
        filters: Number of Conv1D filters. If None, uses config.

    Returns:
        StudentTCN model instance.
    """
    return StudentTCN(sequence_length=sequence_length, filters=filters)


class DistilledStudentTCN:
    """
    Student model trained via knowledge distillation from teacher.
    """

    def __init__(self, teacher_model, data_loader=None):
        """
        Initialize the student model.

        Args:
            teacher_model: Trained teacher PyTorch model.
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
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Build student model
        self.model = build_student_model()
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

    def _generate_batch(self):
        """
        Generate a batch of training data with teacher predictions.

        Returns:
            X_batch: Input sequences tensor
            y_true_batch: Ground truth targets tensor
            y_teacher_batch: Teacher predictions tensor
        """
        X_batch = []
        y_true_batch = []

        for _ in range(self.batch_size):
            # Generate random waveform
            wave = self.data_loader.generate_waveform(avoid_duplicates=False)

            # Random position
            i = random.randint(0, len(wave) - self.sequence_length - 1)
            seq = wave[i:i + self.sequence_length]
            next_sample = wave[i + self.sequence_length]

            X_batch.append(seq)
            y_true_batch.append(-next_sample)  # Inverted for ANC

        X_batch = np.array(X_batch).reshape(-1, self.sequence_length, 1)
        X_tensor = torch.FloatTensor(X_batch).to(self.device)
        y_true_tensor = torch.FloatTensor(y_true_batch).to(self.device)

        # Get teacher predictions
        with torch.no_grad():
            y_teacher_tensor = self.teacher(X_tensor).squeeze()

        return X_tensor, y_true_tensor, y_teacher_tensor

    def _distillation_loss(self, y_pred, y_true, y_teacher):
        """
        Custom distillation loss combining ground truth and teacher guidance.

        Args:
            y_pred: Student predictions
            y_true: Ground truth targets
            y_teacher: Teacher predictions

        Returns:
            Weighted loss combining student and teacher losses.
        """
        student_loss = F.mse_loss(y_pred.squeeze(), y_true)
        teacher_loss = F.mse_loss(y_pred.squeeze(), y_teacher)

        return (1 - self.alpha) * student_loss + self.alpha * teacher_loss

    def train(self):
        """
        Train the student model via distillation.

        Returns:
            Dictionary containing training history.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        history = {'loss': [], 'mse': []}
        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_mse = 0.0

            for step in range(self.steps_per_epoch):
                X_batch, y_true, y_teacher = self._generate_batch()

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self._distillation_loss(y_pred, y_true, y_teacher)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # Calculate MSE to ground truth
                with torch.no_grad():
                    mse = F.mse_loss(y_pred.squeeze(), y_true).item()
                epoch_mse += mse

            avg_loss = epoch_loss / self.steps_per_epoch
            avg_mse = epoch_mse / self.steps_per_epoch
            history['loss'].append(avg_loss)
            history['mse'].append(avg_mse)

            # Memory monitoring
            try:
                import psutil
                mem_percent = psutil.virtual_memory().percent
                print(f"Epoch {epoch + 1}/{self.epochs} - loss: {avg_loss:.6f} - mse: {avg_mse:.6f} - Memory: {mem_percent}%")
            except ImportError:
                print(f"Epoch {epoch + 1}/{self.epochs} - loss: {avg_loss:.6f} - mse: {avg_mse:.6f}")

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
        seqs = [
            test_waveform[i:i + self.sequence_length]
            for i in range(len(test_waveform) - self.sequence_length)
        ]
        X = np.array(seqs).reshape(-1, self.sequence_length, 1)
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions.flatten()

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
            path: Load path. If None, uses config STUDENT_MODEL_PATH.
        """
        if path is None:
            path = STUDENT_MODEL_PATH

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
