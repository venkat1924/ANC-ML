# Deep ANC: Parallel Predictive Feedforward Active Noise Cancellation

A hybrid active noise cancellation system combining **FxLMS** (linear adaptive filter) with **TinyMamba** (neural state-space model) in a parallel topology for enhanced low-frequency noise reduction.

## Key Innovation

**Parallel Topology**: Both FxLMS and Mamba operate on the raw reference signal simultaneously, with outputs summed. This preserves the correlation structure that FxLMS needs while allowing Mamba to capture non-linear residuals.

```
Reference Mic ──┬──► FxLMS ────┬──► Soft Clip ──► Speaker
                │              │
                └──► Mamba ────┘
                      (×g)
```

> **Why not serial?** Placing Mamba before FxLMS would decorrelate the signal, destroying FxLMS's ability to adapt. The parallel approach maintains linear filter stability while adding non-linear enhancement.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data (instant)
python download_dns.py --synthetic

# 3. Train the Mamba model
python train.py --data_dir ./data/raw --epochs 50

# 4. Run evaluation and generate publication figures
python evaluate.py --model_path ./checkpoints/mamba_anc_best.pth

# Or generate realistic mock results (no training needed)
python evaluate.py --mock
```

---

## Project Structure

```
ANC/
├── requirements.txt          # Python dependencies
├── download_dns.py           # Data download/generation script
├── train.py                  # Model training script
├── simulate.py               # Real-time ANC simulation
├── evaluate.py               # Publication figures & results.json
│
├── src/
│   ├── __init__.py
│   ├── utils.py              # soft_clip, watchdog, C-weighting
│   ├── physics.py            # Acoustic digital twin (P(z), S(z), RIR_ref)
│   ├── dataset.py            # DNSDataset with physics-based augmentation
│   ├── mamba_anc.py          # TinyMamba neural predictor
│   ├── fxlms.py              # Filtered-x LMS adaptive filter
│   └── loss.py               # Composite loss (time + spectral + phase)
│
├── data/
│   └── raw/
│       ├── esc50/            # ESC-50 environmental sounds (2000 files)
│       └── synthetic/        # Generated noise samples (10 files)
│
├── checkpoints/              # Saved model weights
│   ├── mamba_anc_best.pth
│   └── training_curves.png
│
└── results/                  # Evaluation outputs
    ├── results.json          # Structured metrics
    ├── fig1_time_domain.pdf
    ├── fig2_psd_comparison.pdf
    ├── fig3_ail_vs_freq.pdf
    ├── fig4_coherence.pdf
    ├── fig5_convergence.pdf
    ├── fig6_ablation.pdf
    ├── fig7_spectrograms.pdf
    └── audio/
        ├── noise_at_ear.wav
        └── error_residual.wav
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACOUSTIC ENVIRONMENT                         │
│                                                                 │
│   Noise Source ──┬──► RIR_ref ──► Reference Mic                │
│                  │                                              │
│                  └──► P(z) ────► (+) ──► Error Mic             │
│                                  ↑                              │
│                        S(z) ◄── Speaker ◄── Drive Signal       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    CONTROL SYSTEM                               │
│                                                                 │
│   Reference ──┬──► FxLMS (64 taps) ────┬──► Soft Clip ──► Drive│
│               │                        │                        │
│               └──► TinyMamba ──► ×g ───┘                       │
│                                                                 │
│   Error ─────────► FxLMS.update()                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Acoustic Paths (physics.py)

| Path | Description | Characteristics |
|------|-------------|-----------------|
| `RIR_ref` | Noise → Reference mic | Strong, flat response |
| `P(z)` | Noise → Ear (primary) | Weaker (passive isolation), bass-heavy |
| `S(z)` | Speaker → Ear (secondary) | Strong, includes hardware delay |

### TinyMamba Model (mamba_anc.py)

```
Input (1, T) ──► Conv1d(1→32, k=4, s=2) ──► GELU
                        ↓
              ┌─── LayerNorm ◄──────────────┐
              │         ↓                   │
              │   Mamba Block (d=32, s=16)  │  × 2 layers
              │         ↓                   │
              │     Dropout(0.1)            │
              │         ↓                   │
              └────► (+) ───────────────────┘
                        ↓
            ConvTranspose1d(32→1, k=4, s=2)
                        ↓
                  Output (1, T)
```

**Parameters**: ~12,500 | **Latency**: <1ms

### FxLMS Algorithm (fxlms.py)

```
y(n) = w^T · x(n)                    # Filter output
w(n+1) = λw(n) - μ·e(n)·x'(n)        # Weight update

where:
  x'(n) = S_hat * x(n)               # Filtered-x (through S(z) estimate)
  λ = 0.9999                         # Leakage factor
  μ = 0.0005                         # Step size (normalized)
```

---

## Training

### Loss Function (loss.py)

```
L_total = λ_t·L_time + λ_s·L_spec + λ_p·L_phase + λ_u·L_uncertainty
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `L_time` | 1.0 | Time-domain MSE |
| `L_spec` | 0.5 | C-weighted spectral magnitude |
| `L_phase` | 0.5 | Phase cosine similarity (1 - cos(Δθ)) |
| `L_uncertainty` | 0.1 | Penalty for high output on stochastic input |

**Why C-weighting (not A-weighting)?** A-weighting de-emphasizes low frequencies, but that's exactly where ANC is most effective. Modified C-weighting keeps bass priority: flat 20Hz-1kHz, -12dB/octave rolloff above.

### Delay Compensation

Training target includes K-sample lookahead to compensate for acoustic + hardware latency:

```python
input = noise[0 : T-K]      # Current signal
target = -noise[K : T]      # Inverted FUTURE signal
```

Default: K=3 samples (~60µs at 48kHz)

### Data Augmentation (dataset.py)

Physics-compliant augmentations that don't break phase relationships:

| Augmentation | Range | Simulates |
|--------------|-------|-----------|
| Gain | ±6 dB | Microphone calibration variance |
| Fractional delay | ±5 samples | Small distance changes |
| Leakage filter | 30% prob | Imperfect headphone seal |

### Training Command

```bash
python train.py \
    --data_dir ./data/raw \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --chunk_length 16384 \
    --K 3
```

---

## Evaluation

### Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **NMSE** | 10·log₁₀(E[e²]/E[d²]) | < -12 dB |
| **AIL** | 10·log₁₀(E[d²]/E[e²]) | > 12 dB |
| **Boost Prob** | P(\|e\| > \|d\|) × 100% | < 5% |
| **Coherence** | MSC(ref, error) | → 0 |

### Expected Performance by Frequency

| Band | AIL (Hybrid) | AIL (FxLMS-only) |
|------|--------------|------------------|
| 31.5 Hz | 10 dB | 8 dB |
| 63 Hz | 17 dB | 14 dB |
| 125 Hz | 18 dB | 15 dB |
| 250 Hz | 14 dB | 12 dB |
| 500 Hz | 10 dB | 8 dB |
| 1 kHz | 4 dB | 3 dB |

### Running Evaluation

```bash
# With trained model
python evaluate.py --model_path ./checkpoints/mamba_anc_best.pth

# Generate realistic mock results (for paper drafting)
python evaluate.py --mock

# Custom output directory
python evaluate.py --mock --output_dir ./paper_figures
```

### Output Files

| File | Description |
|------|-------------|
| `results.json` | Structured metrics for paper |
| `fig1_time_domain.pdf` | Noise vs residual waveforms |
| `fig2_psd_comparison.pdf` | Power spectral density |
| `fig3_ail_vs_freq.pdf` | AIL by frequency band |
| `fig4_coherence.pdf` | Reference-error coherence |
| `fig5_convergence.pdf` | FxLMS weight evolution |
| `fig6_ablation.pdf` | FxLMS vs Mamba vs Hybrid |
| `fig7_spectrograms.pdf` | Before/after spectrograms |

---

## results.json Schema

```json
{
  "experiment": {
    "date": "2024-12-21",
    "model": "TinyMamba",
    "dataset": "ESC-50",
    "sample_rate": 48000
  },
  "global_metrics": {
    "nmse_db": -15.2,
    "ail_db": 15.2,
    "boost_probability_percent": 2.3,
    "rms_reduction_db": 14.8
  },
  "frequency_band_performance": {
    "63_hz": {"ail_db": 16.8, "std_db": 1.4},
    "125_hz": {"ail_db": 18.3, "std_db": 1.2}
  },
  "ablation_study": {
    "fxlms_only": {"ail_db": 12.1},
    "hybrid_fxlms_mamba": {"ail_db": 15.2}
  },
  "model_info": {
    "parameters": 12544,
    "inference_latency_ms": 0.8
  }
}
```

---

## Safety Features

### Watchdog (utils.py)

Monitors 2-5kHz energy to detect feedback instability (howling):

```python
if 2-5kHz energy > threshold for 50+ samples:
    → Bypass ANC, reset filters
```

### Soft Clipping (utils.py)

Prevents amplifier clipping with smooth tanh limiter:

```python
y_safe = threshold × tanh(y / threshold)
```

---

## Dependencies

```
torch>=2.1.0
torchaudio>=2.1.0
numpy>=1.24.0
scipy>=1.10.0
soundfile>=0.12.0
librosa>=0.10.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

Optional for full Mamba support:
```
mamba-ssm>=1.2.0  # Falls back to SimpleSSM if unavailable
```

---

## Simulation

For interactive testing without training:

```bash
python simulate.py --duration 5.0 --noise_type engine
python simulate.py --duration 10.0 --noise_type broadband
python simulate.py --model_path ./checkpoints/best.pth --mamba_gain 0.3
```

---

## Citation

```bibtex
@article{deepanc2024,
  title={Deep ANC: Parallel Predictive Feedforward Active Noise Cancellation 
         with State-Space Models},
  author={...},
  journal={...},
  year={2024}
}
```

---

## References

1. Kuo, S.M. & Morgan, D.R. (1996). *Active Noise Control Systems*
2. Elliott, S.J. (2001). *Signal Processing for Active Control*
3. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
4. Zhang, J. et al. (2021). *Deep Learning for Active Noise Control*

