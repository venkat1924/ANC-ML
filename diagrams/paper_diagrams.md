# Deep ANC: IEEE Publication Diagrams

Mermaid source code for all publication figures. Render using:
- [Mermaid Live Editor](https://mermaid.live/)
- VS Code Mermaid extension
- `mmdc` CLI: `mmdc -i paper_diagrams.md -o output.svg`

---

## Fig. 1: System Architecture

Complete hybrid ANC system showing acoustic environment and parallel control topology.

```mermaid
flowchart TB
    subgraph Acoustic [Acoustic Environment]
        NS[Noise Source]
        NS --> RIR["RIR_ref"]
        NS --> Pz["P(z)"]
        RIR --> RefMic[Reference Mic]
        Pz --> SumNode(("+"))
        Spk[Speaker] --> Sz["S(z)"]
        Sz --> SumNode
        SumNode --> ErrMic[Error Mic]
    end
    
    subgraph Control [Parallel Control System]
        RefMic --> FxLMS["FxLMS<br/>64 taps"]
        RefMic --> Mamba["TinyMamba<br/>SSM"]
        FxLMS --> Sum2(("+"))
        Mamba --> Gain["x g"]
        Gain --> Sum2
        Sum2 --> Clip[SoftClip]
        Clip --> Spk
        ErrMic -.->|"e(n)"| FxLMS
    end
```

**Caption**: Block diagram of the proposed hybrid ANC system. The acoustic environment (top) shows parallel paths: reference mic path (RIR_ref), primary path P(z) from noise to ear, and secondary path S(z) from speaker to ear. The control system (bottom) combines FxLMS adaptive filter with TinyMamba neural predictor in parallel topology, preserving correlation for FxLMS adaptation.

---

## Fig. 2: TinyMamba Neural Network Architecture

Detailed architecture of the state-space model predictor.

```mermaid
flowchart TB
    subgraph Enc [Encoder]
        In["x(n) : B x 1 x T"]
        Conv["Conv1d 1-32, k=4, s=2"]
        G1[GELU]
        In --> Conv --> G1
    end
    
    subgraph Core [Mamba Core x2]
        LN[LayerNorm]
        MB["Mamba SSM<br/>d=32, state=16"]
        Dr[Dropout]
        Res(("+"))
        G1 --> LN --> MB --> Dr --> Res
        G1 --> Res
    end
    
    subgraph Dec [Decoder]
        CT["ConvT1d 32-1, k=4, s=2"]
        Out["y(n) : B x 1 x T"]
        Res --> CT --> Out
    end
```

**Caption**: TinyMamba architecture. Strided convolution encoder downsamples by 2x for efficient processing. Two Mamba SSM blocks with skip connections capture temporal dependencies. Transposed convolution decoder restores original resolution. Total parameters: 15,872.

---

## Fig. 3: Training Pipeline with Delay Compensation

Data flow showing K-sample lookahead for latency compensation.

```mermaid
flowchart LR
    subgraph Data [Dataset]
        A[Audio 48kHz]
        C[Chunk 16384]
        Aug[Augment]
        A --> C --> Aug
    end
    
    subgraph Delay [K-Sample Shift]
        Aug --> X["x: noise(0:T-K)"]
        Aug --> Y["y: -noise(K:T)"]
    end
    
    subgraph Loss [Composite Loss]
        X --> M[TinyMamba]
        M --> L1[L_time]
        M --> L2[L_spec]
        M --> L3[L_phase]
        Y --> L1
        Y --> L2
        Y --> L3
        L1 --> LT[L_total]
        L2 --> LT
        L3 --> LT
    end
```

**Caption**: Training pipeline with delay compensation. Audio is chunked and augmented with physics-based transforms. The K-sample shift (K=3 at 48kHz) creates input-target pairs where the model learns to predict inverted future noise, compensating for acoustic and hardware latency.

---

## Fig. 4: Real-Time Processing Loop

Sequence diagram of chunk-based inference.

```mermaid
sequenceDiagram
    participant Ref as Reference Mic
    participant FxLMS as FxLMS Filter
    participant Mamba as TinyMamba SSM
    participant Sum as Parallel Sum
    participant Spk as Speaker
    participant Path as S(z) Path
    participant Err as Error Mic
    
    loop Every 64 samples at 48kHz
        Ref->>FxLMS: x(n) reference
        Ref->>Mamba: x(n:n+63) chunk
        FxLMS->>Sum: y_lin linear output
        Mamba->>Sum: g * y_deep neural output
        Sum->>Spk: soft_clip(y_total)
        Spk->>Path: drive signal
        Path->>Err: anti-noise arrives
        Err-->>FxLMS: e(n) for weight update
    end
```

**Caption**: Real-time processing sequence. Every 64 samples (~1.3ms at 48kHz), reference signal is processed by FxLMS (sample-wise) and TinyMamba (chunk-wise) in parallel. Outputs are summed and soft-clipped before driving the speaker. Error signal feeds back to FxLMS for online adaptation.

---

## Fig. 5: Mamba SSM Internal Structure

Detailed view of selective state space mechanism.

```mermaid
flowchart LR
    subgraph Input [Input Processing]
        X[x]
        ProjIn["Linear D->2E"]
        X --> ProjIn
    end
    
    subgraph SSM [Selective SSM]
        ProjIn --> Conv1D["Conv1D k=4"]
        Conv1D --> SiLU1[SiLU]
        SiLU1 --> SSMCore["S6 Scan<br/>h = Ah + Bx"]
        ProjIn --> Gate[Gate Branch]
        Gate --> SiLU2[SiLU]
        SSMCore --> Mult(("x"))
        SiLU2 --> Mult
    end
    
    subgraph Output [Output]
        Mult --> ProjOut["Linear E->D"]
        ProjOut --> Y[y]
    end
```

**Caption**: Internal structure of Mamba SSM block. Input is projected and split into SSM and gate branches. The SSM branch applies causal convolution followed by selective state space scan (S6). Gate branch provides input-dependent modulation. This design enables efficient O(n) sequence processing with O(1) memory per step.

---

## Fig. 6: Composite Loss Function

Visualization of multi-objective training loss.

```mermaid
flowchart TB
    subgraph Inputs [Model Outputs]
        Pred["Prediction y"]
        Tgt["Target -x(K:T)"]
    end
    
    subgraph TimeDomain [Time Domain]
        Pred --> MSE["L_time = MSE"]
        Tgt --> MSE
    end
    
    subgraph FreqDomain [Frequency Domain]
        Pred --> FFT1[FFT]
        Tgt --> FFT2[FFT]
        FFT1 --> CW["C-Weight"]
        FFT2 --> CW
        CW --> Lspec["L_spec"]
        FFT1 --> Phase["Phase Diff"]
        FFT2 --> Phase
        Phase --> Lphase["L_phase = 1-cos"]
    end
    
    subgraph Regularization [Regularization]
        Pred --> Var["Variance"]
        Var --> Luncert["L_uncertainty"]
    end
    
    subgraph Total [Weighted Sum]
        MSE -->|"1.0"| LTotal["L_total"]
        Lspec -->|"0.5"| LTotal
        Lphase -->|"0.5"| LTotal
        Luncert -->|"0.1"| LTotal
    end
```

**Caption**: Composite loss function combining time-domain MSE, C-weighted spectral magnitude loss (prioritizing 20-800Hz), phase cosine similarity, and uncertainty penalty. Weights empirically tuned for ANC performance.

---

## Fig. 7: Ablation Study Results

Comparative performance visualization (conceptual).

```mermaid
flowchart LR
    subgraph Methods [Methods Compared]
        M1["FxLMS Only"]
        M2["Mamba Only"]
        M3["Hybrid"]
    end
    
    subgraph Performance [Active Insertion Loss]
        M1 --> P1["12.1 dB"]
        M2 --> P2["10.2 dB"]
        M3 --> P3["16.8 dB"]
    end
    
    subgraph Improvement [Relative Gain]
        P3 --> Gain["+4.7 dB vs FxLMS<br/>+6.6 dB vs Mamba"]
    end
```

**Caption**: Ablation study comparing FxLMS-only, Mamba-only, and hybrid approaches. The parallel hybrid topology achieves 4.7 dB improvement over FxLMS alone by capturing non-linear residuals without disrupting linear filter convergence.

---

## Rendering Instructions

### Option 1: Mermaid Live Editor
1. Go to https://mermaid.live/
2. Paste diagram code
3. Export as SVG or PNG

### Option 2: Mermaid CLI (mmdc)
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i paper_diagrams.md -o fig1.svg -c mermaid.config.json
```

### Option 3: VS Code
1. Install "Markdown Preview Mermaid Support" extension
2. Open this file and preview

### IEEE Style Configuration (mermaid.config.json)
```json
{
  "theme": "neutral",
  "themeVariables": {
    "fontFamily": "Times New Roman, serif",
    "fontSize": "14px"
  },
  "flowchart": {
    "curve": "basis",
    "padding": 20
  }
}
```

---

## LaTeX Integration

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig1_system_architecture.pdf}
    \caption{Block diagram of the proposed hybrid ANC system...}
    \label{fig:system}
\end{figure}
```

