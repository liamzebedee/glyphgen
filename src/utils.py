"""
Shared utilities for Generative Font Renderer.
"""

import torch
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    return text.lower().replace(" ", "-").replace("_", "-")


def font_id(index: int, personality: str) -> str:
    """Generate font ID like '1-fruity'."""
    return f"{index}-{slugify(personality)}"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not."""
    path.mkdir(parents=True, exist_ok=True)
    return path
