#!/usr/bin/env python3
"""
Training script for TinyMamba ANC model.

Trains the neural predictor using DNS Challenge noise data
with delay compensation and composite loss function.

Usage:
    python train.py --data_dir ./data/raw --epochs 50
    python train.py --synthetic --epochs 20  # Use synthetic data
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.mamba_anc import TinyMambaANC, create_model
from src.dataset import DNSDataset, SyntheticNoiseDataset
from src.loss import CompositeLoss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    component_sums = {"time": 0, "spectral": 0, "phase": 0, "uncertainty": 0}
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Match target length if needed
        min_len = min(outputs.shape[-1], targets.shape[-1])
        outputs = outputs[..., :min_len]
        targets = targets[..., :min_len]
        
        # Compute loss
        loss, components = criterion(outputs, targets, input_signal=inputs)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        for k, v in components.items():
            if k in component_sums:
                component_sums[k] += v
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "time": f"{components['time']:.4f}",
            "phase": f"{components['phase']:.4f}"
        })
    
    # Average metrics
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in component_sums.items()}
    
    return {"loss": avg_loss, **avg_components}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    component_sums = {"time": 0, "spectral": 0, "phase": 0, "uncertainty": 0}
    n_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            min_len = min(outputs.shape[-1], targets.shape[-1])
            outputs = outputs[..., :min_len]
            targets = targets[..., :min_len]
            
            loss, components = criterion(outputs, targets)
            
            total_loss += loss.item()
            for k, v in components.items():
                if k in component_sums:
                    component_sums[k] += v
            n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in component_sums.items()}
    
    return {"loss": avg_loss, **avg_components}


def plot_training_curves(history: dict, save_path: Path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Time loss
    axes[0, 1].plot(epochs, history["train_time"], label="Train")
    axes[0, 1].plot(epochs, history["val_time"], label="Val")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Time-Domain MSE")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Phase loss
    axes[1, 0].plot(epochs, history["train_phase"], label="Train")
    axes[1, 0].plot(epochs, history["val_phase"], label="Val")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Phase Cosine Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Spectral loss
    axes[1, 1].plot(epochs, history["train_spectral"], label="Train")
    axes[1, 1].plot(epochs, history["val_spectral"], label="Val")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("C-Weighted Spectral Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TinyMamba ANC model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data/raw",
                        help="Path to training data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of real audio")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum files to load (for testing)")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="tiny",
                        choices=["tiny", "small", "medium"],
                        help="Model size preset")
    parser.add_argument("--d_model", type=int, default=None,
                        help="Override model dimension")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Override number of layers")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--chunk_length", type=int, default=16384,
                        help="Audio chunk length in samples")
    parser.add_argument("--K", type=int, default=3,
                        help="Delay compensation in samples")
    parser.add_argument("--fs", type=int, default=48000,
                        help="Sample rate")
    
    # Loss weights
    parser.add_argument("--lambda_time", type=float, default=1.0)
    parser.add_argument("--lambda_spec", type=float, default=0.5)
    parser.add_argument("--lambda_phase", type=float, default=0.5)
    parser.add_argument("--lambda_uncertainty", type=float, default=0.1)
    
    # Output arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\nLoading datasets...")
    if args.synthetic:
        print("Using synthetic noise data")
        train_dataset = SyntheticNoiseDataset(
            n_samples=2000,
            chunk_length=args.chunk_length,
            K=args.K,
            fs=args.fs
        )
        val_dataset = SyntheticNoiseDataset(
            n_samples=500,
            chunk_length=args.chunk_length,
            K=args.K,
            fs=args.fs,
            seed=42
        )
    else:
        data_dir = Path(args.data_dir)
        
        # Find subdirectories with audio
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if subdirs:
            # Use all subdirectories
            all_files = []
            for subdir in subdirs:
                all_files.extend(list(subdir.rglob("*.wav")))
            print(f"Found {len(all_files)} audio files in {len(subdirs)} subdirectories")
        
        train_dataset = DNSDataset(
            data_dir=args.data_dir,
            chunk_length=args.chunk_length,
            K=args.K,
            fs=args.fs,
            augment=True,
            max_files=args.max_files
        )
        
        # Use a portion for validation (no augmentation)
        val_dataset = DNSDataset(
            data_dir=args.data_dir,
            chunk_length=args.chunk_length,
            K=args.K,
            fs=args.fs,
            augment=False,
            max_files=args.max_files // 5 if args.max_files else None,
            seed=42
        )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Create model
    print("\nCreating model...")
    model_kwargs = {}
    if args.d_model is not None:
        model_kwargs["d_model"] = args.d_model
    if args.n_layers is not None:
        model_kwargs["n_layers"] = args.n_layers
    
    model = create_model(args.model_size, **model_kwargs)
    model = model.to(device)
    
    n_params = model.get_num_params()
    print(f"Model: TinyMamba ({args.model_size})")
    print(f"Parameters: {n_params:,}")
    
    # Create loss function
    criterion = CompositeLoss(
        fs=args.fs,
        lambda_time=args.lambda_time,
        lambda_spec=args.lambda_spec,
        lambda_phase=args.lambda_phase,
        lambda_uncertainty=args.lambda_uncertainty
    )
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_time": [], "val_time": [],
        "train_phase": [], "val_phase": [],
        "train_spectral": [], "val_spectral": []
    }
    
    best_val_loss = float("inf")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_time"].append(train_metrics["time"])
        history["val_time"].append(val_metrics["time"])
        history["train_phase"].append(train_metrics["phase"])
        history["val_phase"].append(val_metrics["phase"])
        history["train_spectral"].append(train_metrics["spectral"])
        history["val_spectral"].append(val_metrics["spectral"])
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Train Phase: {train_metrics['phase']:.4f} | Val Phase: {val_metrics['phase']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = checkpoint_dir / "mamba_anc_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args)
            }, best_path)
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"mamba_anc_epoch{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "args": vars(args)
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
    
    # Save final model
    final_path = checkpoint_dir / "mamba_anc_final.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_metrics["loss"],
        "args": vars(args)
    }, final_path)
    print(f"\nSaved final model: {final_path}")
    
    # Plot training curves
    plot_training_curves(history, checkpoint_dir / "training_curves.png")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

