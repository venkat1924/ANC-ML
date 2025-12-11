# Active Noise Cancellation using Machine Learning

A machine learning approach to Active Noise Cancellation (ANC) using Temporal Convolutional Networks (TCN) with knowledge distillation for model compression.

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Usage](#usage)
7. [Configuration Reference](#configuration-reference)
8. [Project Structure](#project-structure)
9. [Model Details](#model-details)
10. [References](#references)

---

## Overview

This project implements a sample-by-sample audio prediction system for active noise cancellation. The core idea is to predict the **anti-phase (inverted)** of the next audio sample based on a sequence of previous samples. When this predicted anti-phase signal is added to the original signal, destructive interference cancels the noise.

The project includes:
- **Teacher Model**: A larger TCN model trained on ground truth data
- **Student Model**: A compact model trained via knowledge distillation from the teacher, suitable for edge deployment

---

## How It Works

### The ANC Principle

Traditional ANC systems generate a sound wave that is the exact opposite (180 degrees out of phase) of the incoming noise. When combined, the two waves cancel each other out through destructive interference.

### ML-Based Approach

This implementation trains a neural network to:

1. Take a sequence of `N` previous audio samples as input
2. Predict the **negated** next sample: `y = -x[t+1]`
3. Add the prediction to the original signal: `x[t+1] + predicted ≈ 0`

The model learns temporal patterns in the audio to anticipate what comes next, then outputs the inverse.

### Training Target

```
Input:  [x[0], x[1], x[2], ..., x[49]]  (50 samples)
Target: -x[50]                          (inverted next sample)
```

When the model predicts correctly:
```
x[50] + predicted ≈ x[50] + (-x[50]) ≈ 0 (silence)
```

---

## Architecture

### Temporal Convolutional Network (TCN)

TCNs are chosen for this task because:

1. **Causal Convolutions**: Output at time `t` only depends on inputs at times `≤ t`. This is essential for real-time prediction where future samples are unavailable.

2. **Dilated Convolutions**: Exponentially increasing dilation rates (1, 2, 4) expand the receptive field without increasing parameters:
   - Dilation 1: sees samples at offsets [0, 1, 2]
   - Dilation 2: sees samples at offsets [0, 2, 4]
   - Dilation 4: sees samples at offsets [0, 4, 8]

3. **Parallelizable**: Unlike RNNs, convolutions can be computed in parallel during training.

### Causal Padding in PyTorch

PyTorch does not have built-in causal padding like Keras. The implementation pads only the left side:

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))  # Pad left only
        return self.conv(x)
```

### Teacher Model Architecture

```
Input (batch, 50, 1)
    │
    ▼
Permute to (batch, 1, 50)
    │
    ▼
CausalConv1D(64 filters, kernel=3, dilation=1) + BatchNorm1D + ReLU
    │
    ▼
CausalConv1D(64 filters, kernel=3, dilation=2) + BatchNorm1D + ReLU
    │
    ▼
CausalConv1D(64 filters, kernel=3, dilation=4) + BatchNorm1D + ReLU
    │
    ▼
Flatten
    │
    ▼
Linear(1) → Output (predicted anti-phase sample)
```

### Student Model Architecture (Distilled)

```
Input (batch, 50, 1)
    │
    ▼
Permute to (batch, 1, 50)
    │
    ▼
CausalConv1D(32 filters, kernel=3, dilation=1) + BatchNorm1D + ReLU
    │
    ▼
CausalConv1D(32 filters, kernel=3, dilation=2) + BatchNorm1D + ReLU
    │
    ▼
Flatten
    │
    ▼
Linear(1) → Output (predicted anti-phase sample)
```

The student has ~50% fewer parameters: 2 layers instead of 3, 32 filters instead of 64.

### Knowledge Distillation

The student model is trained with a combined loss:

```
Loss = (1 - α) × MSE(student, ground_truth) + α × MSE(student, teacher_prediction)
```

With `α = 0.3`:
- 70% weight on ground truth (actual inverted samples)
- 30% weight on teacher predictions (learned behavior)

This allows the smaller student to learn from both the true signal and the teacher's "soft knowledge" about patterns.

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM recommended
- GPU optional but recommended for training (CUDA or MPS)

### Setup

```bash
# Clone or navigate to project directory
cd ANC

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| torch | Neural network framework |
| numpy | Numerical operations |
| librosa | Audio loading and processing |
| soundata | UrbanSound8K dataset management |
| soundfile | Audio file I/O |
| matplotlib | Visualization |
| scikit-learn | Metrics (MSE) |
| psutil | Memory monitoring |
| IPython | Audio playback in notebooks |

---

## Dataset Setup

This project uses the **UrbanSound8K** dataset, which contains 8,732 labeled sound excerpts (≤4 seconds) of urban sounds.

### Option 1: Automatic Download (via soundata)

The first time you run training, soundata will prompt to download the dataset:

```bash
python train_teacher.py
# Follow prompts to download (~6GB)
```

The dataset will be downloaded to soundata's default location.

### Option 2: Manual Download

1. Download from: https://urbansounddataset.weebly.com/urbansound8k.html
2. Extract to a directory
3. Update `DATA_DIR` in `config.py` to point to your extracted location

### Dataset Structure

```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

### Filtered Classes

The project uses 6 of the 10 classes (urban noise types suitable for ANC):

| Class ID | Sound Type |
|----------|------------|
| 0 | Air conditioner |
| 1 | Car horn |
| 4 | Engine idling |
| 5 | Gun shot |
| 7 | Siren |
| 8 | Street music |

Excluded classes (2: children playing, 3: dog bark, 6: jackhammer, 9: drilling) are speech-like or impulsive sounds less suited for this ANC approach.

---

## Usage

### 1. Train the Teacher Model

```bash
python train_teacher.py
```

This will:
- Load the UrbanSound8K dataset
- Build the teacher TCN model
- Train on randomly selected audio clips
- Evaluate on a test waveform
- Save the model to `saved_models/tcn_noise_cancellation.pt`

### 2. Train the Student Model (Knowledge Distillation)

```bash
python train_student.py
```

**Prerequisite**: Teacher model must be trained first.

This will:
- Load the trained teacher model
- Build the compact student model
- Train via distillation (learning from both ground truth and teacher)
- Save to `saved_models/distilled_student.pt`

### 3. Evaluate on Audio Files

```bash
# Evaluate with teacher model
python evaluate.py path/to/audio.wav --model teacher

# Evaluate with student model
python evaluate.py path/to/audio.wav --model student

# Compare both models
python evaluate.py path/to/audio.wav --model both
```

#### Evaluation Output

The evaluation script produces:
- **MSE to silence**: How close the combined signal is to zero (lower is better)
- **Noise reduction (dB)**: Power reduction in decibels (higher is better)
- **5-panel visualization**:
  1. Input waveform
  2. Predicted anti-phase waveform
  3. Combined waveform (should be near-zero)
  4. Combined vs Input overlay
  5. Residual noise in dB
- **Audio playback**: Original, predicted, and combined audio (in Jupyter/IPython)

### Programmatic Usage

```python
from models.teacher import WaveformPredictorTCN
from data.loader import UrbanSoundLoader

# Initialize
loader = UrbanSoundLoader()
model = WaveformPredictorTCN(data_loader=loader)

# Train
history = model.train()

# Test on a WAV file
model.test_with_wav("path/to/test.wav")

# Save
model.save_model()
```

```python
import torch
from models.teacher import TeacherTCN
from models.student import DistilledStudentTCN
from data.loader import UrbanSoundLoader
from config import TEACHER_MODEL_PATH, SEQUENCE_LENGTH, TEACHER_FILTERS

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load teacher
checkpoint = torch.load(TEACHER_MODEL_PATH, map_location=device)
teacher = TeacherTCN(
    sequence_length=checkpoint.get('sequence_length', SEQUENCE_LENGTH),
    filters=checkpoint.get('filters', TEACHER_FILTERS)
)
teacher.load_state_dict(checkpoint['model_state_dict'])
teacher.to(device)
teacher.eval()

# Create and train student
loader = UrbanSoundLoader()
student = DistilledStudentTCN(teacher_model=teacher, data_loader=loader)
student.train()
student.save_model()
```

---

## Configuration Reference

All configurable parameters are in `config.py`. Modify these to experiment with different settings.

### Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_DIR` | Project root | Base directory for relative paths |
| `DATA_DIR` | `data/urbansound8k` | Dataset location |
| `MODEL_DIR` | `saved_models/` | Directory for saved models |
| `TEACHER_MODEL_PATH` | `saved_models/tcn_noise_cancellation.pt` | Teacher model save path |
| `STUDENT_MODEL_PATH` | `saved_models/distilled_student.pt` | Student model save path |

### Audio Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate in Hz |
| `DURATION` | 1.0 | Audio clip duration in seconds |
| `SEQUENCE_LENGTH` | 50 | Number of input samples (3.125ms at 16kHz) |

### Dataset Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALLOWED_CLASSES` | [0, 1, 4, 5, 7, 8] | UrbanSound8K class IDs to use |

### Teacher Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEACHER_NUM_WAVES` | 5 | Number of audio clips for training |
| `TEACHER_EPOCHS` | 1 | Training epochs |
| `TEACHER_BATCH_SIZE` | 32 | Batch size |
| `TEACHER_LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `TEACHER_FILTERS` | 64 | Conv1D filter count |
| `TEACHER_PATIENCE` | 5 | Early stopping patience |

### Student Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STUDENT_NUM_WAVES` | 50 | Number of audio clips for training |
| `STUDENT_EPOCHS` | 20 | Training epochs |
| `STUDENT_BATCH_SIZE` | 16 | Batch size |
| `STUDENT_LEARNING_RATE` | 0.0005 | Adam optimizer learning rate |
| `STUDENT_FILTERS` | 32 | Conv1D filter count |
| `STUDENT_PATIENCE` | 3 | Early stopping patience |

### Knowledge Distillation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DISTILLATION_ALPHA` | 0.3 | Teacher loss weight (0.3 = 30% teacher, 70% ground truth) |
| `DISTILLATION_STEPS_PER_EPOCH` | 100 | Training steps per epoch |

### Visualization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ZOOM_DURATION` | 0.05 | Plot zoom window in seconds (50ms) |

---

## Project Structure

```
ANC/
├── config.py                 # All hyperparameters and paths
├── data/
│   ├── __init__.py
│   └── loader.py             # UrbanSoundLoader class
├── models/
│   ├── __init__.py
│   ├── teacher.py            # TeacherTCN, WaveformPredictorTCN, CausalConv1d
│   └── student.py            # StudentTCN, DistilledStudentTCN
├── utils/
│   ├── __init__.py
│   ├── audio.py              # load_audio(), normalize_audio(), play_audio()
│   └── visualization.py      # plot_results()
├── train_teacher.py          # Teacher training script
├── train_student.py          # Student distillation script
├── evaluate.py               # Model evaluation CLI
├── requirements.txt          # Python dependencies
├── saved_models/             # Created on first model save
│   ├── tcn_noise_cancellation.pt
│   └── distilled_student.pt
└── README.md                 # This file
```

### Module Descriptions

#### `data/loader.py`

- `UrbanSoundLoader`: Manages dataset loading and preprocessing
  - `download_dataset()`: Download UrbanSound8K via soundata
  - `generate_waveform()`: Load random clip, normalize to [-1, 1]
  - `prepare_sequences()`: Create sliding window training data with inverted targets
  - `reset_duplicates()`: Clear duplicate tracking for fresh training runs

#### `models/teacher.py`

- `CausalConv1d`: Custom causal convolution layer (left-padding only)
- `TeacherTCN`: PyTorch nn.Module with 3-layer TCN architecture
- `build_teacher_model()`: Construct the TeacherTCN model
- `WaveformPredictorTCN`: Full training/inference wrapper class
  - `train()`: Train on prepared sequences with early stopping
  - `predict()`: Generate anti-phase predictions
  - `test_with_wav()`: Full evaluation pipeline for a WAV file
  - `save_model()`, `load_model()`: Model persistence via torch.save/load

#### `models/student.py`

- `StudentTCN`: PyTorch nn.Module with compact 2-layer TCN
- `build_student_model()`: Construct the StudentTCN model
- `DistilledStudentTCN`: Distillation training wrapper class
  - `_generate_batch()`: Generate batches with teacher predictions
  - `_distillation_loss()`: Combined loss function
  - `train()`: Distillation training loop with early stopping
  - `predict()`, `test_with_wav()`: Same interface as teacher

#### `utils/audio.py`

- `load_audio()`: Load and normalize audio via librosa
- `normalize_audio()`: Scale to [-1, 1] range
- `play_audio()`: IPython audio widget display

#### `utils/visualization.py`

- `plot_results()`: 5-panel matplotlib visualization

---

## Model Details

### Parameter Counts

| Model | Layers | Filters | Approximate Parameters |
|-------|--------|---------|------------------------|
| Teacher | 3 CausalConv1D + Linear | 64 | ~220K |
| Student | 2 CausalConv1D + Linear | 32 | ~55K |

### Receptive Field

With sequence length 50 and dilation rates [1, 2, 4]:
- Each Conv1D with kernel size 3 adds `(kernel_size - 1) × dilation_rate` to receptive field
- Teacher effective receptive field: covers the full 50-sample input
- Student (dilations [1, 2]): slightly smaller receptive field

### Inference Time

At 16kHz sample rate:
- Each prediction produces 1 sample
- For real-time ANC, the model must predict faster than 62.5μs per sample
- The student model's smaller size makes it more suitable for real-time edge deployment

### Device Support

The implementation automatically detects and uses:
- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2/M3)
- **CPU**: Fallback for all systems

### Loss Function

- **Teacher**: Standard MSE between predicted and inverted ground truth
- **Student**: Weighted combination:
  ```
  L = 0.7 × MSE(pred, -ground_truth) + 0.3 × MSE(pred, teacher_pred)
  ```

### Model Checkpoint Format

Models are saved as PyTorch state dictionaries with metadata:

```python
{
    'model_state_dict': model.state_dict(),
    'sequence_length': 50,
    'filters': 64  # or 32 for student
}
```

---

## References

- UrbanSound8K Dataset: J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", ACM Multimedia 2014
- Temporal Convolutional Networks: Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- Knowledge Distillation: Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network"
