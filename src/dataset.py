"""
Dataset loading utilities for Generative Font Renderer.

Provides PyTorch Dataset class and DataLoader factory for training data.
Supports train/validation splits with configurable parameters.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.config import CONFIG


class GlyphDataset(Dataset):
    """
    PyTorch Dataset wrapping the pre-generated glyph training data.

    Loads from data/train_dataset.pt which contains:
    - images: (N, 1, 128, 128) float32 in [0, 1]
    - prev_chars: (N,) int64 in [0, 26]
    - curr_chars: (N,) int64 in [1, 26]

    Each sample returns a dict with keys: image, prev_char, curr_char
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Load the glyph dataset from disk.

        Args:
            data_path: Path to .pt file (default: data/train_dataset.pt)

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if data_path is None:
            data_path = CONFIG.data_dir / "train_dataset.pt"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                "Run 'python src/generate_dataset.py' first."
            )

        # Load the dataset
        data = torch.load(data_path, weights_only=True)

        self.images = data["images"]
        self.prev_chars = data["prev_chars"]
        self.curr_chars = data["curr_chars"]

        # Validate shapes
        n_samples = len(self.images)
        assert self.prev_chars.shape == (n_samples,), \
            f"prev_chars shape mismatch: {self.prev_chars.shape}"
        assert self.curr_chars.shape == (n_samples,), \
            f"curr_chars shape mismatch: {self.curr_chars.shape}"

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dict with keys: image (1, 128, 128), prev_char (scalar), curr_char (scalar)
        """
        return {
            "image": self.images[idx],
            "prev_char": self.prev_chars[idx],
            "curr_char": self.curr_chars[idx],
        }


def create_dataloaders(
    batch_size: Optional[int] = None,
    train_val_split: Optional[float] = None,
    data_path: Optional[Path] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        batch_size: Batch size (default: CONFIG.batch_size)
        train_val_split: Fraction for training (default: CONFIG.train_val_split)
        data_path: Path to dataset (default: data/train_dataset.pt)
        num_workers: DataLoader workers (default: 0 for single-process)
        pin_memory: Pin memory for GPU transfer (default: True)
        seed: Random seed for split (default: 42)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = CONFIG.batch_size
    if train_val_split is None:
        train_val_split = CONFIG.train_val_split

    # Load full dataset
    dataset = GlyphDataset(data_path)
    n_samples = len(dataset)

    # Create reproducible split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator).tolist()

    n_train = int(n_samples * train_val_split)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def create_train_loader(
    batch_size: Optional[int] = None,
    data_path: Optional[Path] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the full dataset (no validation split).

    Args:
        batch_size: Batch size (default: CONFIG.batch_size)
        data_path: Path to dataset (default: data/train_dataset.pt)
        num_workers: DataLoader workers (default: 0)
        pin_memory: Pin memory for GPU transfer (default: True)

    Returns:
        DataLoader for full dataset
    """
    if batch_size is None:
        batch_size = CONFIG.batch_size

    dataset = GlyphDataset(data_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    # Quick test
    print("Loading dataset...")
    dataset = GlyphDataset()
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"  image shape: {sample['image'].shape}")
    print(f"  prev_char: {sample['prev_char']}")
    print(f"  curr_char: {sample['curr_char']}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    batch = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"  image: {batch['image'].shape}")
    print(f"  prev_char: {batch['prev_char'].shape}")
    print(f"  curr_char: {batch['curr_char'].shape}")
