### Decision Summary: Spectral Representations for Deep ANC

#### 1. Recommended Primary Representation: **Multi-Band State Space Model (Multi-Band Mamba)**
*   **Why chosen:** It partitions audio into frequency bands (spectral) but processes them with linear-complexity state-space models (time-domain), achieving state-of-the-art attenuation (-13.87 dB) and handling high-frequency speech components that standard STFT models miss.
*   **Computational cost:** **2.4 GFLOPs** (approx. 33% of standard DeepANC/CRN cost of 7.2 GFLOPs).
*   **Phase accuracy requirement:** **< 15–20°** error margin is required for >10dB cancellation; errors >60° cause constructive interference (noise boosting).

#### 2. Alternative Representations (ranked by practical utility)

| Representation Name | Cost | Phase Fidelity | Use Case |
| :--- | :--- | :--- | :--- |
| **Complex Spectral Mapping (CRN)** | High (7.2 GFLOPs) | Good (Cartesian optimization avoids wrapping) | Stationary broadband noise where minor latency is acceptable. |
| **WaveNet-Volterra** | Medium | Excellent (Direct waveform modeling) | Environments with significant loudspeaker non-linearity (saturation). |
| **Hybrid DecNet-LMS** | Low | High (LMS handles phase adaptation) | Decentralized arrays (e.g., automotive cabins) requiring stability. |

#### 3. Specific Implementation Details
*   **Frame size in samples:** **256 samples** (16 ms at 16 kHz) for low-latency configs; up to 320 samples (20 ms) for standard.
*   **Hop size in samples:** **128 samples** (50% overlap).
*   **STFT window type:** **Hann** (Standard) or **Square-root Hann** (if reconstruction requires perfect properties).
*   **Frequency resolution needed:** **~30–60 Hz** minimum bin width.
*   **Filterbank Decision:** **Learnable Filterbanks** (or fixed analysis filter banks processed by Mamba).
    *   *Rationale:* Fixed STFT imposes rigid time-frequency resolution trade-offs (Heisenberg uncertainty); learnable/multi-band approaches adapt receptive fields to cancel both transient and tonal noise components efficiently.

#### 4. Critical Gotchas
*   **Algorithmic Latency Violation:** Standard STFT requires accumulating a full frame ($T_{frame}$), exceeding the acoustic causality budget ($\approx 100\mu s$).
    *   *How to avoid:* Implement **Delay-Compensated Training** (train network to predict $K$ samples ahead) or use **Revised Overlap-Add** (set overlap samples to zero to reduce latency to $T_{hop}$).
*   **Phase Wrapping Instability:** Training networks to predict "Magnitude + Phase" (Polar) creates non-convex loss surfaces due to the $\pm \pi$ discontinuity.
    *   *How to avoid:* Always use **Complex Spectral Mapping** (predict Real and Imaginary components) or **Complex-Valued Neural Networks (CVNNs)** which optimize phase implicitly via Cartesian distance.