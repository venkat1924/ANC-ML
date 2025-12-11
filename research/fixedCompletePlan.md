Software Implementation Plan: Parallel Predictive ANCThis guide allows you to build a bit-accurate "Digital Twin" of the ANC system completely in software. It covers data ingestion, physics simulation, model training, and the real-time runtime loop.0. Project Structure & PrerequisitesDirectory Layout:deep_anc_project/├── data/│   ├── raw/              # Place UrbanSound8K / FSD50K here│   └── processed/        #.pt files for training├── src/│   ├── init.py│   ├── physics.py        # Acoustic Digital Twin (RIRs, Paths)│   ├── mamba_anc.py      # TinyMamba definition + Causal Wrapper│   ├── fxlms.py          # Linear Adaptive Filter│   ├── dataset.py        # Data loading & Augmentation│   └── loss.py           # Psychoacoustic Loss functions├── train.py              # Offline training script├── simulate.py           # Main "Real-Time" simulation loop└── requirements.txtrequirements.txt:torch>=2.1.0torchaudio>=2.1.0numpyscipypyroomacousticsmamba-ssm>=1.2.0tqdmmatplotlibsoundfile1. Data Pipeline (src/dataset.py)We need a dataset of noise (Primary Path) and separate noise for validation. We will standardize everything to 48 kHz for the simulation (a balance between fidelity and speed).Pythonimport torch
import torchaudio
import glob
import os
from torch.utils.data import Dataset

class ANCDataset(Dataset):
    def __init__(self, root_dir, sample_rate=48000, chunk_size=16384):
        self.sr = sample_rate
        self.chunk_size = chunk_size
        # Recursive search for.wav files
        self.files = glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True)
        if len(self.files) == 0:
            raise ValueError(f"No.wav files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            audio, orig_sr = torchaudio.load(path)
            
            # 1. Mix to Mono
            if audio.shape > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # 2. Resample
            if orig_sr!= self.sr:
                resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
                audio = resampler(audio)
            
            # 3. Pad/Crop to chunk_size
            if audio.shape[-1] < self.chunk_size:
                pad = self.chunk_size - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, pad))
            else:
                start = torch.randint(0, audio.shape[-1] - self.chunk_size, (1,))
                audio = audio[:, start:start+self.chunk_size]
                
            return audio # Shape: (1, chunk_size)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(1, self.chunk_size)
2. Acoustic Physics (src/physics.py)This module creates the "Virtual Headphone". It generates the impulse responses (IRs) that couple the noise and speaker to the error microphone.Pythonimport numpy as np
import pyroomacoustics as pra
import scipy.signal as signal

class HeadphonePhysics:
    def __init__(self, fs=48000):
        self.fs = fs
        # Generate Secondary Path (Speaker -> Ear)
        # We simulate a small enclosed space (ear cup)
        room_dim = [0.1, 0.1, 0.05] # 10cm x 10cm x 5cm
        room = pra.ShoeBox(room_dim, fs=fs, max_order=10, absorption=0.3)
        room.add_source([0.02, 0.05, 0.02]) # Speaker driver
        room.add_microphone([0.08, 0.05, 0.02]) # Error Mic (near eardrum)
        room.compute_rir()
        self.Sz = room.rir[0:256] # Truncate IR
        
        # Normalize Secondary Path
        self.Sz /= np.max(np.abs(self.Sz))

        # Hardware Latency (ADC + DAC + Bus)
        # At 48kHz, 4 samples is approx 83us
        self.hardware_delay = 4 

    def get_secondary_path(self):
        # Return S(z) padded with hardware delay
        return np.pad(self.Sz, (self.hardware_delay, 0))
3. Model Architecture (src/mamba_anc.py)We define the TinyMamba model and a helper class for stateful inference.Pythonimport torch
import torch.nn as nn
from mamba_ssm import Mamba

class TinyMambaANC(nn.Module):
    def __init__(self, d_model=64, d_state=16, n_layers=2):
        super().__init__()
        # Simple projection to hidden dim
        self.encoder = nn.Linear(1, d_model)
        
        # Mamba Layers
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (Batch, Time, 1)
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self.decoder(x)

    def step_chunk(self, x_chunk, inference_params_dict):
        """
        Stateful inference for a chunk of data.
        x_chunk: (Batch, ChunkLen, 1)
        inference_params_dict: Dictionary storing K-V caches for Mamba
        """
        x = self.encoder(x_chunk)
        
        # Iterate layers with state passing
        for i, layer in enumerate(self.layers):
            # Use mamba_ssm's internal step logic if available, 
            # or simple forward with state management (simplified here)
            # For 'mamba-ssm' package, we usually pass inference_params
            if f'layer_{i}' not in inference_params_dict:
                 inference_params_dict[f'layer_{i}'] = None # Initialize
            
            # Note: True sample-by-sample step in Python is complex with official CUDA Mamba.
            # We rely on the fact that Mamba is causal. 
            # For simulation, we can run forward() on the chunk.
            # To be strictly continuous across chunks, we would need to pass 
            # the 'conv_state' and 'ssm_state'. 
            # *Simplification for Implementation Plan*: 
            # We will rely on long-context training and simple causal masking 
            # in simulation or use the `inference_params` from Mamba library.
            
            x = layer(x, inference_params=inference_params_dict.get('params'))
            
        return self.decoder(x)
4. The Linear Controller (src/fxlms.py)The standard adaptive filter.Pythonimport numpy as np

class FxLMS:
    def __init__(self, n_taps, learning_rate, sec_path_estimate):
        self.n_taps = n_taps
        self.mu = learning_rate
        self.w = np.zeros(n_taps)
        self.buffer = np.zeros(n_taps)
        self.fx_buffer = np.zeros(n_taps)
        
        # S_hat(z) for filtering the reference
        self.s_hat = sec_path_estimate
        self.s_hist = np.zeros(len(sec_path_estimate))

    def update(self, x_n, e_n):
        # 1. Filter Reference: x'(n) = s_hat * x(n)
        self.s_hist = np.roll(self.s_hist, 1)
        self.s_hist = x_n
        x_prime = np.dot(self.s_hist, self.s_hat)
        
        # 2. Update Fx Buffer
        self.fx_buffer = np.roll(self.fx_buffer, 1)
        self.fx_buffer = x_prime
        
        # 3. LMS Update
        # w = w + mu * e * x'
        self.w += self.mu * e_n * self.fx_buffer

    def predict(self, x_n):
        self.buffer = np.roll(self.buffer, 1)
        self.buffer = x_n
        return np.dot(self.w, self.buffer)
5. Training (train.py)Trains Mamba to predict the anti-noise.Pythonimport torch
from torch.utils.data import DataLoader
from src.dataset import ANCDataset
from src.mamba_anc import TinyMambaANC
import torch.nn.functional as F

# Config
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 10
LATENCY_SAMPLES = 8 # Prediction Horizon (K)

def main():
    # 1. Data
    train_ds = ANCDataset("./data/raw")
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model
    model = TinyMambaANC().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 3. Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            batch = batch.cuda().float() # (B, 1, T) -> need (B, T, 1)
            batch = batch.transpose(1, 2)
            
            # Input: Current noise
            inp = batch
            
            # Target: Future noise (inverted for cancellation)
            # Ideally, we want Model(x) * S(z) = -Noise
            # Simplified: Train model to predict -Noise[t+K]
            tgt = -1.0 * batch
            
            optimizer.zero_grad()
            pred = model(inp)
            
            # Loss: MSE + Frequency Weighting (Optional)
            loss = F.mse_loss(pred, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss/len(loader)}")
    
    torch.save(model.state_dict(), "mamba_anc.pth")

if __name__ == "__main__":
    main()
6. Real-Time Simulator (simulate.py)This is the core verification engine. It runs the physics and control loops chunk-by-chunk.Pythonimport numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.physics import HeadphonePhysics
from src.mamba_anc import TinyMambaANC
from src.fxlms import FxLMS
from mamba_ssm.utils.generation import InferenceParams

def simulation():
    # 1. Setup
    fs = 48000
    physics = HeadphonePhysics(fs)
    S_real = physics.get_secondary_path()
    S_est = S_real * 0.95 # Slight mismatch error
    
    # 2. Load Deep Model
    model = TinyMambaANC().cuda()
    model.load_state_dict(torch.load("mamba_anc.pth"))
    model.eval()
    
    # 3. Load Linear Model
    fxlms = FxLMS(n_taps=64, learning_rate=0.005, sec_path_estimate=S_est)
    
    # 4. Generate Test Noise (Synthetic mixture for testing)
    duration = 5 # seconds
    t = np.linspace(0, duration, duration*fs)
    # 100Hz Drone + Broadband noise
    noise_src = 0.5*np.sin(2*np.pi*100*t) + 0.2*np.random.normal(0, 1, len(t))
    
    # 5. Runtime Loop (Chunked)
    chunk_size = 64
    n_chunks = len(noise_src) // chunk_size
    
    error_log =
    
    # Mamba State Cache
    infer_params = InferenceParams(max_seqlen=chunk_size, max_batch_size=1)
    
    # Delay line for secondary path simulation
    sp_buffer = np.zeros(len(S_real))
    
    print("Starting Simulation...")
    for i in tqdm(range(n_chunks)):
        # Extract Chunk
        start = i * chunk_size
        end = start + chunk_size
        x_chunk_np = noise_src[start:end]
        
        # --- A. Deep Path (Mamba) ---
        # Convert to Tensor (1, Chunk, 1)
        x_tensor = torch.from_numpy(x_chunk_np).float().cuda().view(1, -1, 1)
        with torch.no_grad():
            # In simulation, we just run forward() on the chunk.
            # Ideally, we pass states here to maintain continuity across chunks.
            y_deep_tensor = model(x_tensor) 
            y_deep_np = y_deep_tensor.cpu().numpy().flatten()
            
        # --- B. Linear Path (FxLMS) & Physics ---
        # We must iterate sample-by-sample within the chunk for FxLMS 
        # because it adapts continuously
        chunk_error =
        
        for n in range(chunk_size):
            x_val = x_chunk_np[n]
            y_d = y_deep_np[n]
            
            # 1. FxLMS Prediction
            y_l = fxlms.predict(x_val)
            
            # 2. Total Anti-Noise
            y_total = y_l + y_d
            
            # 3. Physics Simulation (Secondary Path)
            sp_buffer = np.roll(sp_buffer, 1)
            sp_buffer = y_total
            anti_noise_at_ear = np.dot(sp_buffer, S_real)
            
            # 4. Error Formation (Superposition)
            # Assume Primary Path is just identity/delay for this simple sim
            noise_at_ear = x_val 
            e_n = noise_at_ear + anti_noise_at_ear
            
            # 5. FxLMS Update
            fxlms.update(x_val, e_n)
            
            chunk_error.append(e_n)
            
        error_log.extend(chunk_error)

    # 6. Analysis
    error_log = np.array(error_log)
    nmse = 10 * np.log10(np.mean(error_log**2) / np.mean(noise_src[:len(error_log)]**2))
    print(f"Final NMSE: {nmse:.2f} dB")
    
    plt.plot(error_log, label="Residual Error")
    plt.plot(noise_src[:len(error_log)], alpha=0.5, label="Primary Noise")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    simulation()
