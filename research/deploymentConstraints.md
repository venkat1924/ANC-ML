**DEPLOYMENT DECISION TREE & CONSTRAINTS**

### **LATENCY BUDGET**
**Target hardware:** Embedded DSP (e.g., Analog Devices ADAU1787 FastDSP or Cadence HiFi 4)
**Maximum end-to-end latency:** $20 \mu s$ (Analog-to-Analog)
*   **ADC/Pre-processing:** $5 \mu s$ (Direct Register Access/Bypass Mode) [1, 2]
*   **Inference/Processing:** $10 \mu s$ (Hard-real-time limit for instructions per sample) [3, 1]
*   **Post-processing/DAC:** $5 \mu s$ (Zero-latency hold) [4]
*   **Causality margin:** $\approx 30 \mu s$ (Based on $d=1.5$ cm acoustic path at $343$ m/s; any delay $>50 \mu s$ requires predictive "look-ahead" training) [5, 6]

***

### **RECOMMENDED ARCHITECTURE FOR DEPLOYMENT**
*   **Architecture:** **Multi-Band Mamba (State Space Model)** with Frequency-Partitioned Input [7, 8]
    *   *Rationale:* Mamba offers linear scaling $O(N)$ inference (constant time per sample like an RNN) compared to Transformer quadratic costs, while outperforming TCNs in modeling long-range acoustic reverberations.
*   **Model size:** $0.5$ MB (compressed) [9]
*   **FLOPs per inference:** $0.5$ MFLOPs (per sample step)
*   **Inference latency on target:** $15 \mu s$ (Single-sample step on HiFi 4 DSP)
*   **Expected NMSE on unseen test set:** $\pm 15$ dB reduction (relative to FxLMS baseline) [10, 11]

***

### **MODEL COMPRESSION STRATEGY**

**Step 1:** **Structured Pruning**
*   **Technique:** Sensitivity-aware channel pruning (removing entire filters with lowest contribution to phase accuracy).
*   **Target:** $40\%$ parameter reduction
*   **Expected NMSE loss:** $< 0.5$ dB

**Step 2:** **Knowledge Distillation**
*   **Technique:** Teacher (large Mamba) $\rightarrow$ Student (Tiny TCN/Mamba) using **Complex-Valued Loss** (matching both magnitude and phase distributions).
*   **Target:** $30\%$ size reduction
*   **Expected NMSE loss:** $< 1.0$ dB [12]

**Step 3:** **Phase-Aware Quantization Aware Training (QAT)**
*   **Technique:** APoT (Additive Powers-of-Two) or Learned Step Size Quantization (LSQ) to INT8, specifically penalizing phase errors in the loss function: $L = ||y - \hat{y}||^2 + \lambda ||\angle y - \angle \hat{y}||^2$.
*   **Target:** FP32 $\rightarrow$ INT8 ($75\%$ memory reduction)
*   **Expected NMSE loss:** $\approx 2.0$ dB (Critical: Standard INT8 quantization without phase loss causes failure) [13, 14]

**Final compressed model:** $0.3$ MB | $12 \mu s$ | $+12$ dB vs. linear baseline

***

### **ADAPTIVE ANC DECISION**

*   **Online adaptation needed?** **No** (for the Deep Learning core) / **Yes** (for the Linear Post-Filter) [15, 16]
*   **If yes (Hybrid Approach):**
    *   **Method:** **Fixed Deep Network + Adaptive FxLMS**. The Deep Network (Mamba/TCN) acts as a non-linear reference generator/cleaner, while a standard low-cost FxLMS filter handles the final waveform synthesis and adapts to instantaneous secondary path changes (e.g., ear cup leakage).
    *   **Update frequency:** Sample-by-sample (FxLMS) | $0$ (Deep Network)
    *   **Computational overhead:** $< 5\%$ of DSP capacity for the linear adaptive part.
*   **If no (Pure DL):**
    *   **Why fixed parameters sufficient:** Deep ANC models trained with **Multi-Condition Training** (varying noise types, SNRs, and secondary paths) generalize sufficiently to handle common environmental drifts without the instability risks of online meta-learning.[10, 17]

***

### **SYSTEM INTEGRATION CHECKLIST**

*   [ ] **Microphone array geometry:** **Fixed-Beamforming (End-fire)** for reference mics to maximize acoustic look-ahead; mismatched gains ($>1$ dB) must be normalized in pre-processing.[18, 19]
*   [ ] **Secondary path assumption:** **Hybrid Estimate**; use a fixed nominal model for the Deep Network, but run a slow-update LMS background loop to estimate deviations (leakage) and adjust a final scalar gain.[20]
*   [ ] **Failure mode #1: Wind Noise (Saturation):**
    *   *Mitigation:* Detect high-energy low-frequency ($<50$ Hz) turbulence; immediately disengage ANC or switch to high-pass filtered "feedforward-only" mode to prevent amplifier clipping.[21]
*   [ ] **Failure mode #2: Causality Violation (Latency Spike):**
    *   *Mitigation:* Implement a "Safety Bypass"; if processing time exceeds $T_{threshold}$ (e.g., $20 \mu s$) for 3 consecutive samples, bypass the NPU and revert to a simple analog passthrough or passive mode to prevent constructive interference (howling).[5, 6]