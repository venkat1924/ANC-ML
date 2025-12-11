Here is the Practical Implementation Specification based on the research findings.

### **1. Minimum Training Dataset**
- Number of distinct noise types: 10,000 (Non-speech environmental sounds)
- Samples per type: 1 (approx. 6-10 seconds each)
- Total audio duration: ~20 hours
- Acoustic conditions to cover:
  * Room T60 range: 0.1s – 1.0s (Randomized)
  * SNR range: -5 dB to +20 dB
  * Loudspeaker saturation levels: 0% (Linear) to 100% (Hard Clipping)
  * Secondary path mismatch tolerance: ±20% Gain / ±30° Phase (if training Implicit models)

### **2. Secondary Path Handling Decision**
- **Approach:** Hybrid Architecture (HAD-ANC or SFANC)
- **Method:** Offline Deep Learning for feature extraction/nonlinearity coupled with an Online Adaptive Filter (LMS) for path tracking.
- **Why:** Deep networks learn global noise patterns, while the adaptive filter handles real-time acoustic path changes (temperature/movement) that fixed neural weights cannot track.
- **Expected NMSE improvement over implicit:** 3–6 dB (in dynamic/mismatched acoustic environments).

### **3. Data Augmentation Checklist (ranked by effectiveness)**
1.  **RIR Convolution (Domain Randomization)** | **High** (Enables spatial generalization) | **Required**
2.  **Phase-Aware Noise Mixing** | **High** (Prevents amplification of sensor floor) | **Required**
3.  **Source-Level Pitch Shifting** | **Medium** (Generalizes harmonic engine/fan noise) | **Optional**

### **4. Critical Dependency**
- **[Must-have #1]: Physics-Compliant Augmentation Pipeline**
    *   *What fails without it:* If augmentation (time-stretch/pitch-shift) is applied *after* RIR convolution, the phase relationship between reference and error signals breaks, causing the model to learn incorrect anti-noise phases (leading to noise boosting).
- **[Must-have #2]: Online Secondary Path Estimator (or robust linear buffer)**
    *   *What fails without it:* The system diverges (instability) when the physical distance between loudspeaker and error microphone changes by as little as 5cm (approx. 30° phase shift at 2kHz).