"""
Training Script for Deep ANC TinyMamba Model.

Trains the neural network component of the hybrid ANC system.
Uses delay-compensated targets and composite loss function.

Usage:
    python train.py --data_dir ./data/raw --epochs 50 --batch_size 16
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ANCDataset
from src.mamba_anc import TinyMambaANC
from src.loss import CompositeLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyMamba ANC model")
    parser.add_argument("--data_dir", type=str, default="./data/raw",
                        help="Directory containing training audio files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--sample_rate", type=int, default=48000,
                        help="Audio sample rate")
    parser.add_argument("--chunk_size", type=int, default=16384,
                        help="Audio chunk size in samples")
    parser.add_argument("--delay_k", type=int, default=3,
                        help="Delay compensation K samples (~60µs at 48kHz)")
    parser.add_argument("--d_model", type=int, default=32,
                        help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of Mamba layers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda, cpu, or auto)")
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_time = 0
    total_spec = 0
    total_phase = 0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Handle shape mismatch from strided conv
        min_len = min(outputs.shape[-1], targets.shape[-1])
        outputs = outputs[:, :, :min_len]
        targets = targets[:, :, :min_len]
        
        # Compute loss
        loss, components = criterion(outputs, targets, input_signal=inputs[:, :, :min_len])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += components["total"]
        total_time += components["time"]
        total_spec += components["spectral"]
        total_phase += components["phase"]
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{components['total']:.4f}",
            "time": f"{components['time']:.4f}",
            "phase": f"{components['phase']:.4f}"
        })
    
    return {
        "loss": total_loss / n_batches,
        "time": total_time / n_batches,
        "spectral": total_spec / n_batches,
        "phase": total_phase / n_batches
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Validate model."""
    model.eval()
    
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            min_len = min(outputs.shape[-1], targets.shape[-1])
            outputs = outputs[:, :, :min_len]
            targets = targets[:, :, :min_len]
            
            loss, components = criterion(outputs, targets)
            total_loss += components["total"]
            n_batches += 1
    
    return {"loss": total_loss / n_batches}


def main():
    args = parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    print(f"Loading dataset from {args.data_dir}")
    dataset = ANCDataset(
        root_dir=args.data_dir,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        delay_compensation=args.delay_k,
        augment=not args.no_augment
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for production
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = TinyMambaANC(
        d_model=args.d_model,
        d_state=16,
        n_layers=args.n_layers,
        d_conv=4,
        expand=1
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Loss function (composite: time + C-weighted spectral + phase)
    criterion = CompositeLoss(
        fs=args.sample_rate,
        lambda_time=1.0,
        lambda_spec=0.5,
        lambda_phase=0.5,
        lambda_uncertainty=0.1
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float("inf")
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Delay compensation K={args.delay_k} samples (~{args.delay_k * 1e6 / args.sample_rate:.1f}µs)")
    print("-" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Time: {train_metrics['time']:.4f}, "
              f"Spec: {train_metrics['spectral']:.4f}, "
              f"Phase: {train_metrics['phase']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "best_loss": best_loss,
            "args": vars(args)
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.output_dir, "mamba_anc_latest.pth"))
        
        # Save best
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "mamba_anc_best.pth"))
            print(f"  -> New best model saved (loss: {best_loss:.4f})")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model saved to {args.output_dir}/mamba_anc_best.pth")


if __name__ == "__main__":
    main()

