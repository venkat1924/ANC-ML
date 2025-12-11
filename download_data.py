#!/usr/bin/env python3
"""
Download environmental noise datasets for ANC training.

Datasets included:
- ESC-50: Environmental Sound Classification (50 classes, 2000 clips)
- DEMAND: Diverse Environments Multichannel Acoustic Noise Database
- MS-SNSD: Microsoft Scalable Noisy Speech Dataset (noise portion)

Usage:
    python download_data.py --all
    python download_data.py --esc50
    python download_data.py --demand
    python download_data.py --ms-snsd
"""

import argparse
import os
import subprocess
import zipfile
import tarfile
import shutil
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"


def run_cmd(cmd, desc=None):
    """Run a shell command with progress indication."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Warning: Command exited with code {result.returncode}")
    return result.returncode == 0


def download_esc50():
    """
    Download ESC-50: Environmental Sound Classification dataset.
    
    Source: https://github.com/karolpiczak/ESC-50
    Size: ~600 MB
    Contents: 2000 5-second clips, 50 classes including:
        - Engine idling, car horn, siren (vehicles)
        - Rain, wind, thunderstorm (weather)
        - Helicopter, airplane, train (transport)
    """
    print("\n" + "="*60)
    print("  Downloading ESC-50 Dataset")
    print("  (Environmental Sound Classification - 2000 clips)")
    print("="*60)
    
    esc_dir = DATA_DIR / "esc50"
    esc_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = esc_dir / "ESC-50-master.zip"
    
    # Download from GitHub
    if not zip_path.exists():
        url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
        run_cmd(f'curl -L -o "{zip_path}" "{url}"', "Downloading ESC-50...")
    
    # Extract
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(esc_dir)
        
        # Move audio files to raw directory
        audio_src = esc_dir / "ESC-50-master" / "audio"
        if audio_src.exists():
            # Filter for ANC-relevant categories
            # ESC-50 categories that are good for ANC training:
            anc_categories = [
                "1-",   # Dog (impulse sounds)
                "2-",   # Rain
                "3-",   # Crying baby
                "4-",   # Door knock
                "5-",   # Helicopter
                "6-",   # Rooster
                "7-",   # Sea waves
                "8-",   # Sneezing
                "9-",   # Clock tick
                "10-",  # Helicopter
                "11-",  # Chainsaw
                "12-",  # Siren
                "13-",  # Car horn
                "14-",  # Engine
                "15-",  # Train
                "16-",  # Church bells
                "17-",  # Airplane
                "18-",  # Fireworks
                "19-",  # Crow
                "20-",  # Wind
                "21-",  # Water drops
                "22-",  # Wind
                "23-",  # Pouring water
                "24-",  # Toilet flush
                "25-",  # Thunderstorm
                "38-",  # Vacuum cleaner
                "41-",  # Engine idling
                "42-",  # Drilling
                "44-",  # Washing machine
            ]
            
            dest_dir = RAW_DIR / "esc50"
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            count = 0
            for f in audio_src.glob("*.wav"):
                # Copy all files (let the dataset handle filtering)
                shutil.copy(f, dest_dir / f.name)
                count += 1
            
            print(f"Copied {count} audio files to {dest_dir}")
    
    print("ESC-50 download complete!")


def download_demand():
    """
    Download DEMAND: Diverse Environments Multichannel Acoustic Noise Database.
    
    Source: https://zenodo.org/record/1227121
    Size: ~2.5 GB (we'll download a subset)
    Contents: Real-world noise recordings from 18 environments:
        - DKITCHEN, DLIVING, DWASHING (domestic)
        - NFIELD, NPARK, NRIVER (nature)
        - OOFFICE (office)
        - PCAFETER, PRESTO (public)
        - STRAFFIC, SPSQUARE (street)
        - TBUS, TCAR, TMETRO (transport)
    
    Note: Full dataset is large. We download channel 1 only.
    """
    print("\n" + "="*60)
    print("  Downloading DEMAND Dataset (subset)")
    print("  (Diverse Environments Multichannel Acoustic Noise)")
    print("="*60)
    
    demand_dir = DATA_DIR / "demand"
    demand_dir.mkdir(parents=True, exist_ok=True)
    
    # DEMAND is hosted on Zenodo - we'll get specific channels
    # These are the 16kHz versions (smaller, we'll resample)
    base_url = "https://zenodo.org/record/1227121/files"
    
    # Key environments for ANC (transport and domestic noise)
    environments = [
        "TCAR",       # Inside car
        "TBUS",       # Inside bus  
        "TMETRO",     # Metro/subway
        "STRAFFIC",   # Street traffic
        "DWASHING",   # Washing machine
        "DKITCHEN",   # Kitchen noise
        "DLIVING",    # Living room
        "OOFFICE",    # Office
        "PCAFETER",   # Cafeteria
        "NPARK",      # Park (nature)
    ]
    
    dest_dir = RAW_DIR / "demand"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for env in environments:
        zip_name = f"{env}.zip"
        zip_path = demand_dir / zip_name
        
        if not zip_path.exists():
            url = f"{base_url}/{zip_name}?download=1"
            success = run_cmd(
                f'curl -L -o "{zip_path}" "{url}"',
                f"Downloading {env}..."
            )
            if not success:
                print(f"  Failed to download {env}, skipping...")
                continue
        
        # Extract channel 1 only (to save space)
        if zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for member in zf.namelist():
                        if "ch01.wav" in member:
                            # Extract and rename
                            zf.extract(member, demand_dir)
                            src = demand_dir / member
                            dst = dest_dir / f"{env}.wav"
                            if src.exists():
                                shutil.move(str(src), str(dst))
                                print(f"  Extracted: {dst.name}")
            except zipfile.BadZipFile:
                print(f"  Bad zip file: {zip_path}, skipping...")
                zip_path.unlink()
    
    # Cleanup extracted directories
    for d in demand_dir.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
    
    print("DEMAND download complete!")


def download_ms_snsd():
    """
    Download MS-SNSD: Microsoft Scalable Noisy Speech Dataset (noise portion).
    
    Source: https://github.com/microsoft/MS-SNSD
    Size: ~200 MB (noise only)
    Contents: Real-world noise recordings including:
        - AirConditioner, AirportAnnouncements
        - Babble, Bus, Cafe
        - Car, CopyMachine
        - Field, Kitchen, LivingRoom
        - Metro, Munching, Office
        - Restaurant, ShuttingDoor
        - Traffic, Typing, VacuumCleaner, WasherDryer
    """
    print("\n" + "="*60)
    print("  Downloading MS-SNSD Dataset (noise portion)")
    print("  (Microsoft Scalable Noisy Speech Dataset)")
    print("="*60)
    
    snsd_dir = DATA_DIR / "ms_snsd"
    snsd_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone the repository (shallow, noise_train folder only)
    repo_dir = snsd_dir / "MS-SNSD"
    
    if not repo_dir.exists():
        # Use git sparse checkout to get only noise files
        run_cmd(
            f'git clone --depth 1 --filter=blob:none --sparse '
            f'https://github.com/microsoft/MS-SNSD.git "{repo_dir}"',
            "Cloning MS-SNSD repository..."
        )
        
        # Sparse checkout noise directories
        os.chdir(repo_dir)
        run_cmd('git sparse-checkout set noise_train noise_test')
        os.chdir(DATA_DIR.parent)
    
    # Copy noise files to raw directory
    dest_dir = RAW_DIR / "ms_snsd"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for subdir in ["noise_train", "noise_test"]:
        src_dir = repo_dir / subdir
        if src_dir.exists():
            for f in src_dir.glob("*.wav"):
                shutil.copy(f, dest_dir / f.name)
                count += 1
    
    print(f"Copied {count} noise files to {dest_dir}")
    print("MS-SNSD download complete!")


def download_fma_small():
    """
    Download FMA Small for testing (NOT for ANC training - music, not noise).
    Included only as an example of curl-based download.
    """
    print("FMA is music - not suitable for ANC training. Skipping.")


def generate_synthetic():
    """
    Generate synthetic noise samples for immediate testing.
    Useful when downloads are slow or unavailable.
    """
    print("\n" + "="*60)
    print("  Generating Synthetic Noise Samples")
    print("  (For immediate testing)")
    print("="*60)
    
    import numpy as np
    from scipy.io import wavfile
    from scipy import signal
    
    synth_dir = RAW_DIR / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    
    fs = 48000
    duration = 30  # 30 seconds each
    
    def save_wav(name, audio):
        # Normalize and convert to int16
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(synth_dir / f"{name}.wav", fs, audio_int)
        print(f"  Generated: {name}.wav")
    
    t = np.arange(int(fs * duration)) / fs
    
    # 1. Engine-like noise (harmonics of 50Hz with modulation)
    engine = np.zeros(len(t))
    for harm in [1, 2, 3, 4, 5, 6]:
        freq = 50 * harm
        amp = 0.3 / harm
        phase = np.random.uniform(0, 2 * np.pi)
        engine += amp * np.sin(2 * np.pi * freq * t + phase)
    # RPM variation
    engine *= (1 + 0.2 * np.sin(2 * np.pi * 0.3 * t))
    save_wav("engine_50hz", engine)
    
    # 2. HVAC / Air conditioner (broadband low-frequency)
    hvac = np.random.randn(len(t))
    b, a = signal.butter(4, 300 / (fs/2), btype='low')
    hvac = signal.filtfilt(b, a, hvac)
    # Add 60Hz hum
    hvac += 0.2 * np.sin(2 * np.pi * 60 * t)
    save_wav("hvac_drone", hvac)
    
    # 3. Traffic noise (low rumble + occasional events)
    traffic = np.random.randn(len(t))
    b, a = signal.butter(3, 200 / (fs/2), btype='low')
    traffic = signal.filtfilt(b, a, traffic)
    # Add occasional "car pass" events
    for _ in range(10):
        pos = np.random.randint(fs, len(t) - fs)
        event = np.exp(-np.linspace(-3, 3, fs)**2) * 0.5
        traffic[pos:pos+fs] += event * np.random.randn(fs)
    save_wav("traffic_rumble", traffic)
    
    # 4. Fan noise (tonal + broadband)
    fan = np.random.randn(len(t)) * 0.3
    b, a = signal.butter(4, [100/(fs/2), 2000/(fs/2)], btype='band')
    fan = signal.filtfilt(b, a, fan)
    # Add blade pass frequency (~120Hz)
    fan += 0.4 * np.sin(2 * np.pi * 120 * t)
    fan += 0.2 * np.sin(2 * np.pi * 240 * t)
    save_wav("fan_blade", fan)
    
    # 5. Train/subway (rhythmic + rumble)
    train = np.random.randn(len(t))
    b, a = signal.butter(3, 150 / (fs/2), btype='low')
    train = signal.filtfilt(b, a, train)
    # Rail joints every ~0.5 seconds
    joint_interval = int(0.5 * fs)
    for i in range(0, len(t), joint_interval):
        if i + 1000 < len(t):
            train[i:i+1000] += 0.5 * np.exp(-np.linspace(0, 5, 1000)) * np.random.randn(1000)
    save_wav("train_rumble", train)
    
    # 6. Airplane cabin (high broadband)
    airplane = np.random.randn(len(t))
    b, a = signal.butter(4, [50/(fs/2), 1000/(fs/2)], btype='band')
    airplane = signal.filtfilt(b, a, airplane)
    save_wav("airplane_cabin", airplane)
    
    # 7. Generator / motor (strong harmonics)
    motor = np.zeros(len(t))
    base_freq = 100  # 6000 RPM = 100Hz
    for harm in [1, 2, 3, 4, 5, 6, 7, 8]:
        amp = 0.25 / np.sqrt(harm)
        motor += amp * np.sin(2 * np.pi * base_freq * harm * t)
    save_wav("motor_100hz", motor)
    
    # 8. Wind (very low frequency gusts)
    wind = np.random.randn(len(t))
    b, a = signal.butter(2, 100 / (fs/2), btype='low')
    wind = signal.filtfilt(b, a, wind)
    # Amplitude modulation (gusts)
    wind *= (1 + 0.5 * np.sin(2 * np.pi * 0.1 * t)) * (1 + 0.3 * np.sin(2 * np.pi * 0.4 * t))
    save_wav("wind_gusts", wind)
    
    print(f"\nGenerated 8 synthetic noise samples in {synth_dir}")


def print_summary():
    """Print summary of downloaded data."""
    print("\n" + "="*60)
    print("  DOWNLOAD SUMMARY")
    print("="*60)
    
    if not RAW_DIR.exists():
        print("No data downloaded yet.")
        return
    
    total_files = 0
    total_size = 0
    
    for subdir in RAW_DIR.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob("*.wav"))
            size = sum(f.stat().st_size for f in files)
            total_files += len(files)
            total_size += size
            print(f"  {subdir.name}: {len(files)} files ({size / 1e6:.1f} MB)")
    
    print(f"\n  TOTAL: {total_files} audio files ({total_size / 1e6:.1f} MB)")
    print(f"  Location: {RAW_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for ANC training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py --all          # Download everything
    python download_data.py --synthetic    # Quick: generate synthetic noise
    python download_data.py --esc50        # Download ESC-50 only
    python download_data.py --demand       # Download DEMAND only
        """
    )
    
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets")
    parser.add_argument("--esc50", action="store_true",
                        help="Download ESC-50 (Environmental Sound Classification)")
    parser.add_argument("--demand", action="store_true",
                        help="Download DEMAND (Multichannel Noise Database)")
    parser.add_argument("--ms-snsd", action="store_true",
                        help="Download MS-SNSD (Microsoft noise dataset)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic noise (fast, no download)")
    
    args = parser.parse_args()
    
    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Default to synthetic if nothing specified
    if not any([args.all, args.esc50, args.demand, args.ms_snsd, args.synthetic]):
        print("No dataset specified. Use --help for options.")
        print("Generating synthetic data for quick testing...")
        args.synthetic = True
    
    if args.all or args.synthetic:
        generate_synthetic()
    
    if args.all or args.esc50:
        download_esc50()
    
    if args.all or args.demand:
        download_demand()
    
    if args.all or args.ms_snsd:
        download_ms_snsd()
    
    print_summary()
    
    print("\n" + "="*60)
    print("  NEXT STEPS")
    print("="*60)
    print("""
  1. Train the model:
     python train.py --data_dir ./data/raw --epochs 50

  2. Run simulation with trained model:
     python simulate.py --model_path ./checkpoints/mamba_anc_best.pth --mamba_gain 0.3
""")


if __name__ == "__main__":
    main()

