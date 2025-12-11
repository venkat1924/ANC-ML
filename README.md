# Deep ANC: Parallel Predictive Feedforward Active Noise Cancellation

A **software-only** simulation of a hybrid Active Noise Cancellation system combining classical adaptive filtering (FxLMS) with deep learning (Mamba SSM). Designed to model real-time headphone ANC with physics-accurate acoustic paths.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Architecture](#architecture)
4. [Module Reference](#module-reference)
5. [Signal Flow](#signal-flow)
6. [Training](#training)
7. [Simulation](#simulation)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Configuration Reference](#configuration-reference)
10. [Dependencies](#dependencies)
11. [Quick Start](#quick-start)

---

## System Overview

### What This System Does

Active Noise Cancellation works by generating anti-noise that destructively interferes with unwanted sound. This implementation simulates a **feedforward ANC** headphone system where:

1. A **reference microphone** captures environmental noise
2. A **controller** predicts what anti-noise to generate
3. A **speaker** plays the anti-noise into the ear cup
4. An **error microphone** measures residual noise for adaptation

### Why Hybrid (FxLMS + Neural Network)?

| Controller | Strengths | Weaknesses |
|------------|-----------|------------|
| **FxLMS** | Converges reliably on linear/periodic noise, mathematically guaranteed stable | Cannot model nonlinearities, limited by filter length |
| **Neural (Mamba)** | Can learn complex nonlinear patterns, predictive capability | Requires training data, can fail on unseen distributions |

**Parallel combination** preserves FxLMS's stability guarantees while allowing the neural network to handle residual nonlinear components that FxLMS misses.

### Critical Design Principle: Parallel, Not Serial

```
                     ┌──────────────┐
                     │   TinyMamba  │──→ y_deep
     ┌───→ x(n) ────┤              │
     │               └──────────────┘
Ref ─┤                              ┌───→ (+) ─→ y_total ─→ Speaker
Mic  │               ┌──────────────┐    ↑
     └───→ x(n) ────┤    FxLMS     │────┘
                     │   ← e(n) ←───────── Error Mic
                     └──────────────┘
```

**Why parallel?** Serial topology (Mamba → FxLMS) would decorrelate the reference signal, destroying the statistical relationship FxLMS needs to converge. In parallel, both controllers receive the raw reference, and their outputs are summed.

---

## Theoretical Foundations

### The Causality Constraint

The fundamental physics limitation in ANC:

```
                    P(z) delay
    Noise ─────────────────────────────→ Ear
       │                                  ↑
       └─→ Ref Mic → Controller → Speaker ─┘
           │←────── S(z) delay ──────→│
```

**For cancellation to work:**
```
Processing_Time + S(z)_delay ≤ P(z)_delay
```

If the anti-noise arrives **after** the noise, cancellation fails. This is why:
1. The primary path P(z) must have sufficient delay (acoustic distance)
2. The secondary path S(z) must be as short as possible
3. Processing latency must be minimized

### Phase Accuracy Requirements

| Phase Error | Effect on Cancellation |
|-------------|------------------------|
| 0° | Perfect cancellation (100%) |
| 30° | 6 dB reduction (50%) |
| **60°** | **0 dB (no effect)** |
| 90° | 3 dB amplification |
| 180° | Doubling (worst case) |

**Critical threshold: 60°.** Beyond this, ANC makes noise *worse*. This motivates:
- Phase-linear filters only (no IIR in signal path)
- Explicit phase loss term in neural network training
- Conservative output when phase is uncertain

### Why C-Weighting, Not A-Weighting

A-weighting (used for perceived loudness) de-emphasizes bass:
- 100 Hz: -19 dB
- 50 Hz: -30 dB

But passive headphone isolation already handles highs well. **ANC is most needed at bass frequencies where foam fails.** C-weighting maintains flat response from 20-1000 Hz, matching ANC's target band.

### Delay Compensation in Training

Hardware introduces latency (ADC + DAC ≈ 60µs = 3 samples @ 48kHz). The neural network must **predict K samples into the future**:

```
Input:  noise[0 : T-K]      # Present
Target: -noise[K : T]       # Inverted future (K=3 samples)
```

This trains the model to output what the anti-noise *will need to be* when it actually arrives at the ear.

---

## Architecture

### Directory Structure

```
/home/jovyan/ANC/
├── requirements.txt          # Dependencies with versions
├── src/
│   ├── __init__.py          # Package exports
│   ├── utils.py             # Safety, DSP utilities, weighting curves
│   ├── physics.py           # Acoustic digital twin (P(z), S(z), RIR_ref)
│   ├── dataset.py           # Data pipeline with physics-based augmentation
│   ├── mamba_anc.py         # TinyMamba neural network
│   ├── fxlms.py             # FxLMS adaptive filter
│   └── loss.py              # Composite training loss
├── train.py                 # Offline training script
├── simulate.py              # Real-time simulation loop
├── data/raw/                # Training audio files (UrbanSound8K, FSD50K)
└── checkpoints/             # Saved model weights
```

### Acoustic Path Model

Three distinct impulse responses model the headphone environment:

```
                    ┌─────────────────────────────────────┐
                    │           HEADPHONE PHYSICS         │
                    │                                     │
   Noise ──────────►│─── RIR_ref ───► Reference Mic      │
   Source           │                                     │
                    │─── P(z) ──────► Ear                │──► Error Mic
                    │                  ↑                  │
   Drive ──────────►│─── Volterra ─► S(z) ─┘             │
                    │                                     │
                    └─────────────────────────────────────┘
```

| Path | Description | Typical Characteristics |
|------|-------------|------------------------|
| `RIR_ref` | Noise source → Reference mic | Short, direct, high gain |
| `P(z)` | Noise source → Ear (primary) | Longer delay, low-passed by cup |
| `S(z)` | Speaker → Ear (secondary) | Short (6cm acoustic path), includes hardware delay |

**Key insight:** `RIR_ref ≠ P(z)`. The reference mic sees the noise before it reaches the ear, providing the causal margin needed for processing.

---

## Module Reference

### `src/utils.py` — Core Utilities

#### `soft_clip(x, threshold=0.95)`

Tanh-based soft limiter preventing amplifier clipping.

```python
def soft_clip(x, threshold=0.95):
    return threshold * tanh(x / threshold)
```

| Input | Output |
|-------|--------|
| 0.5 | 0.462 |
| 0.95 | 0.716 |
| 1.5 | 0.864 |
| 10.0 | 0.950 |

**Purpose:** Prevents hard clipping artifacts while allowing full dynamic range below threshold. In simulation, threshold is set to 2.0 to accommodate signal scaling.

#### `watchdog_check(error_buffer, fs=48000, threshold_db=-3.0, sustained_samples=50)`

Anti-howling detector monitoring 2-5kHz energy.

**Returns:** `(triggered: bool, energy_db: float)`

**Trigger condition:** Band-limited RMS exceeds threshold for sustained duration.

**Action when triggered:** Reset FxLMS weights, set Mamba gain to 0, enter bypass mode.

**Why 2-5kHz?** Howling (acoustic feedback) manifests as resonance peaks in this mid-frequency band. Bass energy during normal operation doesn't trigger false positives.

#### `fractional_delay(signal, delay_samples, filter_length=31)`

Sub-sample delay using windowed sinc interpolation.

**Algorithm:**
1. Generate sinc kernel centered at fractional delay
2. Apply Hann window to suppress ringing
3. Convolve signal with windowed sinc
4. Apply integer delay via array shift

**Use case:** Data augmentation (simulating acoustic path changes) without phase distortion.

#### `modified_c_weight(freqs)`

Custom frequency weighting for ANC loss function.

| Frequency | Weight |
|-----------|--------|
| < 20 Hz | 0 (DC protection) |
| 20-1000 Hz | 1.0 (flat) |
| > 1000 Hz | (1000/f)² (-12 dB/octave) |

**Linear scale output** (not dB). Multiply with spectral magnitudes directly.

#### `generate_leakage_tf(fs, leak_type)`

Generates transfer functions simulating imperfect headphone seal.

| Type | Cutoff | Gain | Blend |
|------|--------|------|-------|
| light | 80 Hz | -3 dB | 70% direct |
| medium | 150 Hz | -6 dB | 50% direct |
| heavy | 300 Hz | -12 dB | 30% direct |

**Physics:** Seal leakage creates a high-pass effect (bass escapes), modeled as HP filter blended with direct path.

---

### `src/physics.py` — Acoustic Digital Twin

#### `class AcousticPhysics`

Complete acoustic environment simulation.

**Constructor:**
```python
AcousticPhysics(
    fs=48000,                    # Sample rate
    hardware_delay_samples=3,    # ADC+DAC latency (60µs @ 48kHz)
    ir_length=256,               # Impulse response length
    use_pyroomacoustics=True     # Use PRA for RIR generation
)
```

**Path Generation (with pyroomacoustics):**

| Path | Room | Source Position | Mic Position | Post-processing |
|------|------|-----------------|--------------|-----------------|
| `RIR_ref` | 3×3×2.5m, absorption=0.4 | (1.5, 1.5, 1.5) | (1.6, 1.5, 1.5) | Normalize |
| `P(z)` | 3×3×2.5m, absorption=0.3 | (1.5, 1.5, 1.5) | (1.65, 1.5, 1.5) | LP filter @ 800Hz |
| `S(z)` | 0.1×0.1×0.05m (ear cup) | (0.02, 0.05, 0.025) | (0.08, 0.05, 0.025) | Prepend hardware delay |

**Path Generation (synthetic fallback):**
```python
RIR_ref = [0, 0.9, 0, 0, 0.1, ...]           # Direct + reflection
P_z = [0, ..., 0.4, ..., 0.1, ..., 0.05, ...] # Delayed, attenuated
S_z = [0, 0, 0, 0.8, 0, 0, 0.1, ...]         # Starts at hardware_delay
```

#### `speaker_nonlinearity(x, drive_level=0.8)`

3rd-order Volterra approximation of speaker distortion.

```python
def speaker_nonlinearity(x, drive_level=0.8):
    a = 0.1 * drive_level
    return x - a * (x ** 3)
```

**Physics:** At high excursions, speaker cone suspension exhibits soft saturation. This introduces odd harmonics (3rd, 5th...) that FxLMS cannot cancel.

#### `simulate_paths(noise_source, drive_signal)`

Full simulation of parallel acoustic paths.

**Returns:** `(ref_mic, error_mic, noise_at_ear, anti_at_ear)`

**Computation:**
```
ref_mic = convolve(noise_source, RIR_ref)
noise_at_ear = convolve(noise_source, P(z))
anti_at_ear = convolve(speaker_nonlinearity(drive_signal), S(z))
error_mic = noise_at_ear + anti_at_ear   # Parallel summation
```

---

### `src/fxlms.py` — Filtered-x LMS Adaptive Filter

#### `class FxLMS`

Standard FxLMS with NLMS normalization and safety features.

**Constructor:**
```python
FxLMS(
    n_taps=64,                      # Filter length
    learning_rate=0.005,            # Step size µ (tune carefully)
    secondary_path_estimate=None,   # S_hat(z) for filtered-x
    leakage=0.9999                  # Leaky LMS coefficient
)
```

**Internal State:**
- `w`: Adaptive weights, shape `(n_taps,)`
- `x_buffer`: Reference signal history for FIR filtering
- `fx_buffer`: Filtered-x history for weight update
- `ref_history`: Reference history for S_hat filtering
- `power_estimate`: Running power for NLMS normalization

#### `update(x_n, e_n)`

Weight adaptation using FxLMS algorithm.

**Algorithm:**
```
1. ref_history ← roll(ref_history); ref_history[0] = x_n
2. x_filtered = dot(ref_history, S_hat)
3. fx_buffer ← roll(fx_buffer); fx_buffer[0] = x_filtered
4. power_estimate = α * power_estimate + (1-α) * x_filtered²
5. µ_norm = µ / (power_estimate + ε)
6. gradient = e_n * fx_buffer
7. gradient = clip(gradient, max_norm=1.0)
8. w = leakage * w - µ_norm * gradient   ← NEGATIVE SIGN CRITICAL
```

**Why negative sign?** ANC minimizes error by making `y * S(z) ≈ -d(n)`. Standard LMS uses positive sign for system identification (estimating unknown filter). ANC is the inverse: we know the target (inverted noise) and adjust weights to produce it.

#### `predict(x_n)`

Generate anti-noise sample.

```python
def predict(x_n):
    x_buffer = roll(x_buffer); x_buffer[0] = x_n
    return dot(w, x_buffer)
```

#### `step(x_n, e_n)`

Combined predict + update for simulation loop.

**Causal order:**
1. Update weights with previous error (already measured)
2. Predict current output (to be measured next cycle)

---

### `src/mamba_anc.py` — Neural Predictor

#### `class TinyMambaANC`

Lightweight State Space Model for non-linear noise prediction.

**Architecture:**
```
Input (B, 1, T)
    │
    ▼
┌───────────────────────────────────┐
│ Conv1d(1→32, k=4, stride=2) + SiLU│  Encoder
└───────────────────────────────────┘
    │ (B, 32, T/2)
    ▼ transpose → (B, T/2, 32)
┌───────────────────────────────────┐
│ LayerNorm → Mamba(d=32, state=16) │  Block 1
│              + Skip Connection    │
└───────────────────────────────────┘
    │
┌───────────────────────────────────┐
│ LayerNorm → Mamba(d=32, state=16) │  Block 2
│              + Skip Connection    │
└───────────────────────────────────┘
    │ transpose → (B, 32, T/2)
    ▼
┌───────────────────────────────────┐
│ ConvTranspose1d(32→1, k=4, s=2)   │  Decoder
└───────────────────────────────────┘
    │
Output (B, 1, T)
```

**Constructor:**
```python
TinyMambaANC(
    d_model=32,     # Hidden dimension
    d_state=16,     # SSM state dimension
    n_layers=2,     # Number of Mamba blocks
    d_conv=4,       # Conv kernel in Mamba
    expand=1,       # Expansion factor (1 = minimal)
    dropout=0.0     # Dropout probability
)
```

**Parameter Count:** ~50K (suitable for edge deployment)

**Phase Linearity:** Strided Conv1d encoder preserves linear phase, unlike IIR filters which introduce group delay distortion.

**Fallback Mode:** If `mamba_ssm` is unavailable, uses `SimpleSSM` (basic conv + gating, no selective scan). For production, use actual Mamba.

#### `apply_leaky_state(leak_factor=0.95)`

Prevents numerical drift during extended inference.

**Implementation note:** For actual Mamba, this requires accessing internal `conv_state` and `ssm_state`. In current implementation, this is a placeholder. For 24/7 deployment, implement proper state decay.

---

### `src/loss.py` — Composite Training Loss

#### `composite_loss(pred, target, fs=48000, ...)`

Combined objective function for ANC training.

**Formula:**
```
L_total = λ₁·L_time + λ₂·L_spec + λ₃·L_phase + λ₄·L_uncertainty
```

**Default weights:** `λ = [1.0, 0.5, 0.5, 0.1]`

**Returns:** `(total_loss, {"time": ..., "spectral": ..., "phase": ..., "uncertainty": ...})`

#### Individual Loss Components

**`time_loss(pred, target)`**
```python
L_time = MSE(pred, target)
```
Basic waveform matching. Necessary but not sufficient—identical MSE can have vastly different phase alignments.

**`spectral_loss(pred, target, fs=48000, use_log=True)`**

C-weighted spectral magnitude distance.

```python
pred_fft = rfft(pred)
target_fft = rfft(target)
weights = modified_c_weight(frequencies)
if use_log:
    pred_mag = log10(|pred_fft| + 1e-8)
    target_mag = log10(|target_fft| + 1e-8)
L_spec = mean(weights * (pred_mag - target_mag)²)
```

**Why log?** Log-magnitude distance is more perceptual—a 6 dB error sounds the same whether at -30 dB or -60 dB absolute level.

**`phase_cosine_loss(pred, target, fs=48000, magnitude_gate=0.01)`**

Explicit phase alignment penalty.

```python
cos_sim = Re(pred_fft * conj(target_fft)) / (|pred_fft| * |target_fft| + ε)
L_phase = mean(weights * (1 - cos_sim) * magnitude_mask)
```

**Magnitude gate:** Only penalize phase where there's energy. Phase of near-zero components is undefined.

**Loss value interpretation:**
| Value | Meaning |
|-------|---------|
| 0 | Perfect phase alignment |
| 1 | 90° average error |
| 2 | 180° (complete inversion) |

**`uncertainty_penalty(pred, input_energy=None, threshold=0.1)`**

Penalizes high output when prediction confidence should be low.

**Purpose:** Stochastic noise (wind gusts, impacts) is unpredictable. Better to output nothing than guess wrong. This loss encourages conservative output when input energy is low or irregular.

---

### `src/dataset.py` — Data Pipeline

#### `class ANCDataset`

PyTorch Dataset with physics-based augmentation.

**Constructor:**
```python
ANCDataset(
    root_dir="./data/raw",           # Directory with .wav files
    sample_rate=48000,               # Target sample rate
    chunk_size=16384,                # Samples per chunk (~340ms)
    delay_compensation=3,            # K samples future prediction
    augment=True,                    # Enable augmentation
    gain_range_db=(-6.0, 6.0),       # Random gain range
    delay_range_samples=(-5.0, 5.0), # Random delay range
    leakage_probability=0.3          # Probability of leakage TF
)
```

**File discovery:** Recursively searches for `.wav`, `.flac`, `.mp3`

**Audio loading priority:** `torchaudio` → `soundfile`

**Processing pipeline:**
1. Load audio file
2. Convert to mono (average channels)
3. Resample to target rate (scipy.signal.resample)
4. Pad/crop to chunk_size
5. Normalize to 0.9 peak
6. Apply augmentations (if enabled)
7. Create input/target pair with delay shift

**Augmentations (physics-based):**

| Augmentation | Range | Simulates |
|--------------|-------|-----------|
| Gain | ±6 dB | Mic calibration variance |
| Fractional delay | ±5 samples | Acoustic path changes |
| Leakage TF | 30% probability | Imperfect headphone seal |

**Output format:**
```python
input:  torch.Tensor (1, chunk_size - K)   # Present noise
target: torch.Tensor (1, chunk_size - K)   # INVERTED future noise
```

**Synthetic mode:** If no files found, generates synthetic engine+wind noise for testing.

---

## Signal Flow

### Complete Simulation Loop (`simulate.py`)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION                                 │
├──────────────────────────────────────────────────────────────────────┤
│ 1. Create AcousticPhysics(fs=48000, hardware_delay=3)                │
│ 2. Create FxLMS(n_taps=64, µ=0.0005, S_hat=physics.S_z_hat)         │
│ 3. Create TinyMamba(d_model=32, n_layers=2)                          │
│ 4. Load model weights (if available)                                 │
│ 5. Generate test noise                                               │
│ 6. Pre-compute: ref_full = noise * RIR_ref                          │
│                 noise_at_ear = noise * P(z)                         │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     CHUNK LOOP (64 samples)                          │
├──────────────────────────────────────────────────────────────────────┤
│ For each chunk:                                                       │
│                                                                       │
│ A. MAMBA PREDICTION (chunk-based for efficiency)                     │
│    ┌─────────────────────────────────────────────────────────────┐   │
│    │ x_tensor = ref_chunk.reshape(1, 1, chunk_size)              │   │
│    │ y_deep_chunk = model(x_tensor).squeeze() * g                │   │
│    └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ B. SAMPLE LOOP (within chunk, for FxLMS adaptation)                  │
│    ┌─────────────────────────────────────────────────────────────┐   │
│    │ For i in chunk_size:                                        │   │
│    │   ref_sample = ref_full[n]                                  │   │
│    │   y_lin = fxlms.predict(ref_sample)                         │   │
│    │   y_deep = y_deep_chunk[i]                                  │   │
│    │   y_total = soft_clip(y_lin + y_deep, threshold=2.0)        │   │
│    │   speaker_out = speaker_nonlinearity(y_total)               │   │
│    │   anti_at_ear = convolve_sample(speaker_out, S(z))          │   │
│    │   error = noise_at_ear[n] + anti_at_ear                     │   │
│    │   fxlms.update(ref_sample, error)                           │   │
│    └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ C. WATCHDOG CHECK (end of chunk)                                     │
│    If 2-5kHz energy > threshold for 50+ samples:                     │
│      → Reset FxLMS weights                                           │
│      → Set g = 0                                                     │
│      → Enter bypass mode                                             │
└──────────────────────────────────────────────────────────────────────┘
```

### Why This Hybrid Processing?

- **Mamba:** Processed chunk-by-chunk (64 samples) because Python sample-by-sample overhead is prohibitive (~10µs/sample). This approximates DSP block-based processing.

- **FxLMS:** Must run sample-by-sample for proper adaptation. Each error measurement immediately affects the next weight update.

- **Pre-computed paths:** Full convolution done once before loop for accuracy. Sample-by-sample buffer convolution is an approximation that accumulates error.

---

## Training

### `train.py` Usage

```bash
python train.py \
    --data_dir ./data/raw \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --sample_rate 48000 \
    --chunk_size 16384 \
    --delay_k 3 \
    --d_model 32 \
    --n_layers 2
```

### Training Loop Details

**Input preparation:**
```python
input = noise[:, :, :-K]      # All but last K samples
target = -noise[:, :, K:]     # Inverted, shifted forward K samples
```

**Forward pass:**
```python
output = model(input)                              # (B, 1, T-K)
min_len = min(output.shape[-1], target.shape[-1])  # Handle stride mismatch
output = output[:, :, :min_len]
target = target[:, :, :min_len]
loss, components = criterion(output, target, input_signal=input)
```

**Backward pass:**
```python
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stability
optimizer.step()
```

**Scheduler:** Cosine annealing from `lr` to `lr * 0.01` over all epochs.

**Checkpointing:** Saves `mamba_anc_latest.pth` every epoch, `mamba_anc_best.pth` on validation improvement.

---

## Simulation

### `simulate.py` Usage

```bash
# FxLMS only (Mamba disabled)
python simulate.py --duration 10 --noise_type engine --mamba_gain 0.0

# With trained Mamba
python simulate.py --model_path ./checkpoints/mamba_anc_best.pth --mamba_gain 0.3

# Quick test with sine wave
python simulate.py --noise_type sine --duration 5
```

### Noise Types

| Type | Description | FxLMS Performance |
|------|-------------|-------------------|
| `sine` | 100 Hz pure tone | Excellent (perfect predictor) |
| `engine` | Harmonics of 50 Hz with modulation | Good (periodic) |
| `wind` | Low-passed Gaussian | Poor (stochastic) |
| `mixed` | 70% engine + 30% wind | Moderate |
| `sweep` | 20-2000 Hz log chirp | Tracking test |

### Output Plot

Saved to `--output_plot` (default: `./anc_results.png`):

1. **Top panel:** Noise at ear (blue) vs Error with ANC (red) — first 500ms
2. **Second panel:** FxLMS output (green) vs Mamba output (magenta)
3. **Third panel:** Full duration comparison
4. **Bottom panel:** Error spectrogram

---

## Evaluation Metrics

### Primary Metrics

| Metric | Formula | Target | Meaning |
|--------|---------|--------|---------|
| **NMSE** | `10*log10(E[e²]/E[d²])` | < -15 dB | Normalized error power |
| **Active Insertion Loss** | `-NMSE` | > 15 dB | Effective attenuation |
| **Boost Probability** | `mean(|e| > |d|) * 100` | < 0.1% | % time ANC amplifies |

### Secondary Metrics (for development)

| Metric | Purpose |
|--------|---------|
| **Coherence (MSC)** | Should approach 0 after convergence (error uncorrelated with reference) |
| **Impulse Artifact Rate** | Peak detection in `diff(error)` — measures clicks/pops |
| **Hiss Level** | A-weighted RMS during silence — measures added noise floor |

### Metric Calculation

```python
def compute_metrics(noise_at_ear, error, fs):
    skip = fs // 4  # Skip first 250ms (convergence transient)
    noise = noise_at_ear[skip:]
    err = error[skip:]
    
    nmse = 10 * log10(mean(err²) / mean(noise²))
    boost_prob = mean(|err| > |noise|) * 100
    
    return {"nmse_db": nmse, "boost_probability": boost_prob}
```

---

## Configuration Reference

### FxLMS Tuning

| Parameter | Default | Tuning Guidance |
|-----------|---------|-----------------|
| `n_taps` | 64 | Increase for longer path delays, decrease for faster adaptation |
| `learning_rate` | 0.0005 | **Critical.** Increase until unstable, back off 50%. |
| `leakage` | 0.9999 | Closer to 1 = slower drift, closer to 0.99 = faster forgetting |

**Learning rate sensitivity:**
- Too high: Weights diverge (error explodes)
- Too low: Slow convergence, poor tracking
- Optimal: Fastest convergence without instability

### Mamba Tuning

| Parameter | Default | Tuning Guidance |
|-----------|---------|-----------------|
| `mamba_gain` | 0.0 | Start at 0, increase by 0.1 after verifying FxLMS stability |
| `d_model` | 32 | Increase for more capacity, decrease for lower latency |
| `n_layers` | 2 | More layers = more capacity but more compute |

### Physics Tuning

| Parameter | Default | Tuning Guidance |
|-----------|---------|-----------------|
| `hardware_delay` | 3 samples | Match actual ADC+DAC latency |
| `P(z) attenuation` | -7 dB | Passive isolation of headphone cup |
| `S(z) delay` | 3 samples | Acoustic path from speaker to ear (~6cm) |

### Safety Thresholds

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `soft_clip threshold` | 2.0 | Maximum drive signal amplitude |
| `watchdog threshold` | 6.0 dB | 2-5kHz energy trigger level |
| `watchdog sustained` | 50 samples | Sustained high energy before trigger |

---

## Dependencies

### `requirements.txt`

```
torch>=2.1.0
torchaudio>=2.1.0
numpy>=1.24.0
scipy>=1.10.0
pyroomacoustics>=0.7.0
mamba-ssm>=1.2.0
tqdm>=4.65.0
matplotlib>=3.7.0
soundfile>=0.12.0
```

### Installation Notes

**mamba-ssm:** Requires CUDA for GPU acceleration. CPU fallback uses `SimpleSSM`.

```bash
pip install mamba-ssm  # May require: pip install causal-conv1d first
```

**pyroomacoustics:** Optional. Falls back to synthetic paths if unavailable.

```bash
pip install pyroomacoustics
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd /home/jovyan/ANC
pip install -r requirements.txt
```

### 2. Test FxLMS (No Training Required)

```bash
python simulate.py --noise_type sine --duration 5 --mamba_gain 0.0
```

Expected output:
```
NMSE: -30 to -40 dB (PASS)
Boost Probability: <0.1% (PASS)
```

### 3. Train Mamba (Optional)

```bash
# Add audio files to data/raw/ first
python train.py --data_dir ./data/raw --epochs 50
```

### 4. Run Full Hybrid Simulation

```bash
python simulate.py \
    --model_path ./checkpoints/mamba_anc_best.pth \
    --noise_type engine \
    --mamba_gain 0.3 \
    --duration 10
```

---

## Technical Notes

### Known Limitations

1. **Mamba sample-by-sample:** Current implementation processes chunks, not true sample-by-sample. Real hardware would use C/CUDA kernels for sample-rate processing.

2. **Pre-computed paths:** Simulation pre-computes full convolutions for accuracy. Real-time would use circular buffers with sample-by-sample accumulation.

3. **State management:** `apply_leaky_state()` is a stub. Production requires proper SSM state access and decay.

### Common Issues

**FxLMS diverges:**
- Reduce `learning_rate` by 50%
- Check `S_hat` estimate accuracy
- Verify path physics (P(z) delay > S(z) delay + processing)

**Poor Mamba performance:**
- Ensure training data matches test noise distribution
- Increase `delay_compensation` if using slower hardware
- Check phase loss is decreasing during training

**Watchdog false triggers:**
- Increase `threshold_db`
- Increase `sustained_samples`
- Check for DC offset in signals

---

## License

MIT License — See LICENSE file.

## Citation

If using this code for research:

```bibtex
@software{deep_anc_2024,
  title={Deep ANC: Parallel Predictive Feedforward Active Noise Cancellation},
  author={...},
  year={2024},
  url={https://github.com/...}
}
```

