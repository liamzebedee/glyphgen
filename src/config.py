"""
Configuration for Generative Font Renderer.

All hyperparameters and paths in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Central configuration for the font renderer."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def networks_dir(self) -> Path:
        return self.project_root / "networks"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def fonts_dir(self) -> Path:
        return self.project_root / "fonts"

    @property
    def font_path(self) -> Path:
        return self.fonts_dir / "OpenSans-Regular.ttf"

    # Dataset
    image_size: int = 128
    num_augmentations: int = 20
    train_val_split: float = 0.9

    # Character set: a-z (26 chars), plus padding token (0)
    # prev_char: 0 = start token, 1-26 = a-z
    # curr_char: 1-26 = a-z
    vocab_size: int = 27  # 0 (padding) + 26 letters

    # Augmentation ranges
    rotation_range: tuple = (-5, 5)  # degrees
    scale_range: tuple = (0.9, 1.1)
    translate_range: tuple = (-4, 4)  # pixels
    blur_range: tuple = (0, 1)
    noise_std: float = 0.02

    # Model architecture
    char_embed_dim: int = 64
    style_dim: int = 64
    hidden_dim: int = 512
    init_spatial: int = 4
    init_channels: int = 256

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Loss weights
    edge_loss_weight: float = 0.1

    # Inference
    target_inference_ms: float = 5.0

    # Online learning
    online_lr: float = 1e-4
    baseline_momentum: float = 0.9
    perturbation_std: float = 0.1

    # Personalities
    personalities: List[str] = field(
        default_factory=lambda: ["fruity", "dumb", "aggressive-sans"]
    )

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.networks_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.fonts_dir.mkdir(exist_ok=True)


# Global config instance
CONFIG = Config()
