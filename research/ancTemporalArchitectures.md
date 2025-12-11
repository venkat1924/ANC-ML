### Implementation Selection Matrix

| Architecture | Params | FLOPs | Latency (ms) | Max Receptive Field (ms) | Generalization (NMSE) | Recommendation |
|---|---|---|---|---|---|---|
| **TCN** (Conv-TasNet) | ~2.3M - 7M | ~5 - 10 G | 2 - 10 ms (Tunable) | High (>500ms) | -10.5 dB (Avg) | **Legacy / Baseline** |
| **CRN** (DeepANC/DCCRN) | ~14 - 15M [1] | 7.2 G [2] | 10 - 20 ms (Frame-locked) | Medium (LSTM limited) | -10.69 dB [2] | **Robust Baseline** |
| **Mamba** (DeepASC) | **8.0M** (Small) [1] | **2.86 G** [1] | **< 2.6 ms** [3] | **Infinite** (Global State) | **-13.46 dB** (Best) [2] | **High-Perf / Edge** |
| **Transformer** (ARN) | 15.9M [1] | 5.28 G [2] | 0 ms (Predictive) [4] | Global (Windowed) | -11.61 dB [2] | **Latency-Critical** |

### Recommendations for ANC Scenario

*   **Best for SOTA performance:** **Multi-Band Mamba (DeepASC)** because it achieves the lowest Normalized Mean Square Error (-13.46 dB), improving over baselines by ~2.8 dB while handling high-frequency speech components that CRNs miss.[1, 2]
*   **Best for real-time deployment:** **Multi-Band Mamba (Small)** because it delivers superior cancellation with only **2.86 GFLOPs** (vs. 7.2G for CRN) and linear $O(N)$ inference complexity, fitting comfortably within tight DSP power budgets.[1, 5]
*   **Recommended for your first iteration:** **Attentive Recurrent Network (ARN)** because its **Delay-Compensated Training** mechanism explicitly solves the hardware latency problem (providing 0ms or negative effective latency), simplifying the initial deployment on slower prototype hardware.[4]

### Critical Architectural Feature (Must-Have)

*   **Delay-Compensated Training (Predictive Modeling):**
    This feature is non-negotiable because the "Triangle of Impossibility" ($T_{elec} + T_{proc} < T_{acous}$) physically prevents standard causal reaction; the network *must* predict the noise waveform $K$ samples into the future to effectively generate anti-noise that arrives exactly when the physical noise wave hits the ear.[4]