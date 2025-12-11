Here is the **Loss & Evaluation Protocol Card** for Deep ANC systems.

### **LOSS FUNCTION SELECTION**

**Primary Loss (for training):**

```
Name: A-Weighted Frequency-Domain MSE (Delay-Compensated)
Formula: L_primary = (1/N) * Σ |W_A(f) *|^2
Why better than MSE: Aligns error minimization with human auditory sensitivity (IEC 61672), reducing wasted energy on sub-bass frequencies.
Hyperparameters:
```

**Secondary Loss (if multi-task):**

```
Name: Speech Preservation Loss (Negative SNR / Inverted Signal Loss)
Weight in combined loss: [α ≈ 0.2 to 0.5 depending on speech priority]
When to use: Only for "Transparency Mode" or communication headsets where speech bands (300Hz-3.4kHz) must be preserved.[1]
```

-----

### **EVALUATION METRICS**

**Primary Metric (for validation & comparisons):**

```
Name: Normalized Mean Square Error (NMSE)
Formula: NMSE_dB = 10 * log10( Σ e(t)^2 / Σ d(t)^2 )
Why chosen: Industry standard for quantifying physical energy reduction; allows direct benchmarking against FxLMS baselines.[1]
What it measures: The ratio of residual error power to original noise power.
Acceptable threshold:[1]
```

**Secondary Metrics (ranked by importance):**

1.  **DNSMOS P.835 (BAK)** | `Model_Inference(e(t)) -> [1.0 - 5.0]` | Measures perceptual background noise quality, penalizing "musical noise" artifacts that NMSE misses.[2] | **\> 3.5 MOS**
2.  **Kurtosis Reduction Ratio (KRR)** | `K(noise) / K(error)` where `K(x) = E[(x-μ)^4]/σ^4` | Quantifies the suppression of impulsive "popping" artifacts; ensures error distribution is Gaussian. | **\> 1.0 (Targeting K\_error ≈ 3)**
3.  **Quiet Zone Diameter** | `Distance(r) where Attenuation(r) ≥ 10 dB` | Determines spatial robustness; prevents "head-in-vise" effect. | **\> 10 cm (binaural width)**

-----

### **GENERALIZATION TEST PROTOCOL**

Run these specific tests:

  - **Test \#1:** → Expected NMSE degradation: **12-15%** (approx. 2-3 dB drop) [1]
  - **Test \#2:** → Expected NMSE degradation: **10-12%** [1]
  - **Test \#3:** → Expected NMSE degradation: **15-20%** (Instability risk high) [1]