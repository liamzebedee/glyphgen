"""
Training script for GlyphNetwork.

Implements:
- Loss functions (MSE reconstruction + edge loss)
- Training loop with optimizer and scheduler
- Validation and checkpointing
- CLI interface

Usage:
    python src/train.py --epochs 100
    python src/train.py --resume outputs/checkpoint.pt
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.config import CONFIG
from src.dataset import create_dataloaders
from src.glyphnet import GlyphNetwork, count_parameters
from src.utils import get_device, save_checkpoint, load_checkpoint, ensure_dir


def compute_edge_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute edge-aware loss using Sobel gradient magnitude.

    Encourages sharp glyph boundaries by penalizing differences in
    gradient magnitude between output and target images.

    Args:
        output: Predicted glyph images (B, 1, H, W)
        target: Ground truth images (B, 1, H, W)

    Returns:
        Scalar edge loss
    """
    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=output.dtype,
        device=output.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=output.dtype,
        device=output.device,
    ).view(1, 1, 3, 3)

    # Compute gradients for output
    out_grad_x = F.conv2d(output, sobel_x, padding=1)
    out_grad_y = F.conv2d(output, sobel_y, padding=1)
    out_grad_mag = torch.sqrt(out_grad_x**2 + out_grad_y**2 + 1e-8)

    # Compute gradients for target
    tgt_grad_x = F.conv2d(target, sobel_x, padding=1)
    tgt_grad_y = F.conv2d(target, sobel_y, padding=1)
    tgt_grad_mag = torch.sqrt(tgt_grad_x**2 + tgt_grad_y**2 + 1e-8)

    # MSE of gradient magnitudes
    return F.mse_loss(out_grad_mag, tgt_grad_mag)


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    edge_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined reconstruction + edge loss.

    Args:
        output: Predicted glyph images (B, 1, H, W)
        target: Ground truth images (B, 1, H, W)
        edge_weight: Weight for edge loss term

    Returns:
        Tuple of (total_loss, reconstruction_loss, edge_loss)
    """
    recon_loss = F.mse_loss(output, target)
    edge_loss = compute_edge_loss(output, target)
    total_loss = recon_loss + edge_weight * edge_loss
    return total_loss, recon_loss, edge_loss


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    style_z: torch.Tensor,
    edge_weight: float,
) -> Tuple[float, float, float]:
    """
    Train for one epoch.

    Args:
        model: GlyphNetwork model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        style_z: Shared style vector (64-dim)
        edge_weight: Weight for edge loss

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_edge_loss)
    """
    model.train()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    edge_loss_sum = 0.0
    n_batches = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        prev_chars = batch["prev_char"].to(device)
        curr_chars = batch["curr_char"].to(device)

        # Expand style_z to batch size
        batch_size = images.size(0)
        style_batch = style_z.unsqueeze(0).expand(batch_size, -1)

        optimizer.zero_grad()

        # Forward pass
        output = model(prev_chars, curr_chars, style_batch)

        # Compute loss
        total_loss, recon_loss, edge_loss = compute_loss(
            output, images, edge_weight
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        edge_loss_sum += edge_loss.item()
        n_batches += 1

    return (
        total_loss_sum / n_batches,
        recon_loss_sum / n_batches,
        edge_loss_sum / n_batches,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    style_z: torch.Tensor,
    edge_weight: float,
) -> Tuple[float, float, float]:
    """
    Validate the model.

    Args:
        model: GlyphNetwork model
        val_loader: Validation data loader
        device: Device to use
        style_z: Shared style vector
        edge_weight: Weight for edge loss

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_edge_loss)
    """
    model.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    edge_loss_sum = 0.0
    n_batches = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        prev_chars = batch["prev_char"].to(device)
        curr_chars = batch["curr_char"].to(device)

        batch_size = images.size(0)
        style_batch = style_z.unsqueeze(0).expand(batch_size, -1)

        output = model(prev_chars, curr_chars, style_batch)

        total_loss, recon_loss, edge_loss = compute_loss(
            output, images, edge_weight
        )

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        edge_loss_sum += edge_loss.item()
        n_batches += 1

    return (
        total_loss_sum / n_batches,
        recon_loss_sum / n_batches,
        edge_loss_sum / n_batches,
    )


def train(
    epochs: int,
    resume_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Train GlyphNetwork.

    Args:
        epochs: Number of epochs to train
        resume_path: Path to checkpoint to resume from
        checkpoint_dir: Directory for checkpoints
        device: Device to use

    Returns:
        Tuple of (trained_model, style_z)
    """
    if device is None:
        device = get_device()

    if checkpoint_dir is None:
        checkpoint_dir = CONFIG.outputs_dir
    ensure_dir(checkpoint_dir)

    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {CONFIG.batch_size}")
    print(f"Learning rate: {CONFIG.learning_rate}")
    print(f"Edge loss weight: {CONFIG.edge_loss_weight}")

    # Create model
    model = GlyphNetwork().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Initialize style_z (learnable during online learning, fixed here)
    style_z = torch.zeros(CONFIG.style_dim, device=device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )

    # Learning rate scheduler with warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG.warmup_epochs,
        T_mult=2,
        eta_min=CONFIG.min_lr,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume from checkpoint if specified
    if resume_path is not None and resume_path.exists():
        print(f"Resuming from {resume_path}")
        checkpoint = load_checkpoint(resume_path, model, optimizer)
        start_epoch = checkpoint["epoch"] + 1
        if "style_z" in checkpoint:
            style_z = checkpoint["style_z"].to(device)
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        # Train
        train_loss, train_recon, train_edge = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            style_z,
            CONFIG.edge_loss_weight,
        )

        # Validate
        val_loss, val_recon, val_edge = validate(
            model,
            val_loader,
            device,
            style_z,
            CONFIG.edge_loss_weight,
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log progress
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train: {train_loss:.4f} (R:{train_recon:.4f} E:{train_edge:.4f}) | "
            f"Val: {val_loss:.4f} (R:{val_recon:.4f} E:{val_edge:.4f}) | "
            f"LR: {current_lr:.2e}"
        )

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            # Also save style_z
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "style_z": style_z,
                    "loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"  -> Saved best checkpoint (val_loss: {val_loss:.4f})")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "style_z": style_z,
                    "loss": val_loss,
                },
                periodic_path,
            )

    # Save final checkpoint
    final_path = checkpoint_dir / "final_checkpoint.pt"
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "style_z": style_z,
            "loss": val_loss,
        },
        final_path,
    )
    print(f"Training complete. Final checkpoint: {final_path}")

    return model, style_z


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train GlyphNetwork")
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG.num_epochs,
        help=f"Number of epochs (default: {CONFIG.num_epochs})",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints (default: outputs/)",
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        resume_path=args.resume,
        checkpoint_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
