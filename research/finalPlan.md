### Phase 1: Architecture & Signal Flow
**Selected Topology:** **Hybrid Multi-Band Mamba + FxLMS**
This separates non-linear prediction (DL) from phase-matching (Linear Adaptive).

1.  **The "Brain" (Deep Learning Core):**
    * **Model:** Multi-Band Mamba (State Space Model).
    * **Input Handling:** **Learnable Analysis Filterbanks** (IIR-based). Do **not** use STFT (block-buffering) for inference; use filterbanks to maintain near-zero latency while providing spectral separation.
    * **Role:** Predicts the noise waveform $K$ samples into the future (Delay Compensation) and denoises non-linear components.
    * **State:** The model maintains a global hidden state; it does *not* reset per frame.

2.  **The "Reflex" (Linear Control Loop):**
    * **Component:** Standard FxLMS (Filtered-x Least Mean Squares) Adaptive Filter.
    * **Role:** Takes the "cleaned/predicted" reference from Mamba and applies final linear gain/phase shift to match the instantaneous secondary path (ear cup dynamics).

### Phase 2: Data Engineering & Physics
1.  **Physics-Compliant Pipeline:**
    * **Order:** `Raw Noise` $\to$ `Phase-Aware Mixing` $\to$ `RIR Convolution` $\to$ `Secondary Path Simulation` $\to$ `Model Input`.
    * **Constraint:** Do not apply time-stretching *after* RIR/Path simulation. This breaks the phase relationship required for cancellation.

2.  **Dataset Composition:**
    * **Source:** 20 hours non-speech environmental noise (10k types).
    * **Augmentation:** Randomize Secondary Path Gain ($\pm 20\%$) and Phase ($\pm 30^\circ$) to prevent "Head-in-Vise" fragility.

### Phase 3: Training & Loss Landscape

**Loss Function Strategy:**
We utilize a **Composite Loss** because pure A-weighted MSE ignores phase (physics failure), while pure Complex loss ignores human sensitivity (perceptual failure).

**Formula:**
$$L_{total} = \lambda_{1}L_{Complex} + \lambda_{2}L_{A-Freq}$$

1.  **$L_{Complex}$ (Physics/Cancellation Priority):**
    * *Definition:* $\frac{1}{N} \sum |Y_{target} - Y_{pred}|^2$ in the Time Domain (Cartesian distance).
    * *Why:* This is mathematically non-negotiable for ANC. Minimizing this directly minimizes the residual acoustic pressure. It implicitly handles phase without the wrapping issues of Polar coordinates.
    * *Delay Compensation:* Target $Y_{target}$ is shifted $K$ samples ahead to account for $T_{elec}$.

2.  **$L_{A-Freq}$ (Perception Priority):**
    * *Definition:* Frequency-domain MSE weighted by the IEC 61672 A-weighting curve.
    * *Why:* Integrated per your request. It forces the model to spend its limited parameter capacity on the frequencies humans actually hear (500Hz - 4kHz), rather than wasting capacity on sub-bass (<50Hz) or ultra-high freq (>10kHz) where cancellation is less perceptible.

### Phase 4: Evaluation Protocol

**Primary Metric (Physics):**
* **NMSE (Normalized Mean Square Error):** Target $< -10$ dB. Measures raw energy reduction.

**Secondary Metrics (Perception & Robustness):**
* **DNSMOS P.835 (BAK):** Target $> 3.5$ MOS. Crucial for detecting "musical noise" artifacts that NMSE misses.
* **Quiet Zone Diameter:** Target $> 10$ cm. Measures spatial robustness (head-in-vise effect). If attenuation drops $< 10$ dB within a 5cm movement, the model is overfitted to a specific point in space.
* **Kurtosis Reduction Ratio (KRR):** Target $> 1.0$. Ensures the model isn't trading continuous noise reduction for impulsive popping artifacts.

### Phase 5: Model Compression & Deployment Optimization

1. **Three-Step Compression Path** (per *deploymentConstraints.md*):
   * **Structured Pruning:** Remove ~40 % of channels with the lowest phase-sensitivity → **↓ params, < 0.5 dB NMSE loss**.
   * **Knowledge Distillation:** Train a tiny student (≤ 0.5 MB) from the full Mamba teacher using the composite loss above → **↓30 % additional size, < 1 dB NMSE loss**.
   * **Phase-Aware QAT:** Quantize to INT8 using **APoT** (or LSQ) with a phase penalty term → **75 % memory reduction**, final model ≈ 0.3 MB, ≤ 15 µs / sample on HiFi-4.

2. **Latency Budget Tracking:** Ensure post-compression inference time < 10 µs to leave ≥ 5 µs guard for watchdog logic.

3. **Validation after each stage:** Re-compute NMSE, DNSMOS-BAK, and Quiet-Zone to confirm < 2 dB cumulative regression.

### Phase 6: Deployment & Integration Checklist

**Hardware constraints:** $20 \mu s$ total latency budget (Analog-to-Analog).

**Microphone Geometry & Setup:**
* [ ] **Geometry:** **End-fire Fixed-Beamforming** array for reference mics. This maximizes acoustic look-ahead.
* [ ] **Gain Normalization:** Mismatched gains $>1$ dB between array mics must be normalized in pre-processing to prevent spatial aliasing in the model input.

**Runtime Safety & Failure Modes:**
* [ ] **Wind Noise Saturation:** Detect high-energy turbulence ($<50$ Hz). **Action:** Disengage ANC or switch to High-Pass Feedforward mode immediately to prevent amp clipping.
* [ ] **Causality Violation (Latency Spike):** Implement a **Safety Bypass**. If processing time $> 15 \mu s$ for 3 consecutive samples, bypass the NPU/Mamba core to prevent constructive interference (howling).
* [ ] **Secondary Path Drift:** The Deep Network uses a *fixed* nominal path. The FxLMS loop must run in parallel to handle the $\pm 20\%$ deviation caused by headset fit/leakage.