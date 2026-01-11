"""
Tests for dataset loading utilities.

Tests cover:
- Dataset loading and sample access
- Data shapes and value ranges
- DataLoader creation with train/val split
- Reproducibility with fixed seed
- Edge cases and error handling
"""

import pytest
import torch
from pathlib import Path

from src.config import CONFIG
from src.dataset import GlyphDataset, create_dataloaders, create_train_loader


class TestGlyphDataset:
    """Tests for GlyphDataset class."""

    @pytest.fixture
    def dataset(self):
        """Load dataset fixture."""
        return GlyphDataset()

    def test_dataset_loads(self, dataset):
        """Dataset loads without error."""
        assert dataset is not None
        assert len(dataset) > 0

    def test_dataset_size(self, dataset):
        """Dataset has expected 14,040 samples."""
        # 26 chars x 27 contexts x 20 augmentations = 14,040
        expected = 26 * 27 * 20
        assert len(dataset) == expected, \
            f"Expected {expected} samples, got {len(dataset)}"

    def test_sample_keys(self, dataset):
        """Sample has expected keys."""
        sample = dataset[0]
        assert "image" in sample
        assert "prev_char" in sample
        assert "curr_char" in sample

    def test_image_shape(self, dataset):
        """Image has shape (1, 128, 128)."""
        sample = dataset[0]
        assert sample["image"].shape == (1, 128, 128)

    def test_image_dtype(self, dataset):
        """Image is float32."""
        sample = dataset[0]
        assert sample["image"].dtype == torch.float32

    def test_image_range(self, dataset):
        """Image values in [0, 1]."""
        sample = dataset[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_char_ids_dtype(self, dataset):
        """Character IDs are int64."""
        sample = dataset[0]
        assert sample["prev_char"].dtype == torch.int64
        assert sample["curr_char"].dtype == torch.int64

    def test_prev_char_range(self, dataset):
        """prev_char in [0, 26]."""
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            assert 0 <= sample["prev_char"] <= 26, \
                f"Sample {i}: prev_char={sample['prev_char']}"

    def test_curr_char_range(self, dataset):
        """curr_char in [1, 26]."""
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            assert 1 <= sample["curr_char"] <= 26, \
                f"Sample {i}: curr_char={sample['curr_char']}"

    def test_all_chars_represented(self, dataset):
        """All 26 characters appear in dataset."""
        curr_chars = set()
        for i in range(len(dataset)):
            curr_chars.add(dataset[i]["curr_char"].item())

        expected = set(range(1, 27))
        assert curr_chars == expected, \
            f"Missing chars: {expected - curr_chars}"

    def test_all_contexts_represented(self, dataset):
        """All 27 context values appear in dataset."""
        prev_chars = set()
        for i in range(len(dataset)):
            prev_chars.add(dataset[i]["prev_char"].item())

        expected = set(range(0, 27))
        assert prev_chars == expected, \
            f"Missing contexts: {expected - prev_chars}"

    def test_images_not_blank(self, dataset):
        """Images contain actual character pixels."""
        sample = dataset[0]
        # Should have variation (not all zeros or all ones)
        assert sample["image"].std() > 0.01, "Image appears blank"

    def test_dataset_file_not_found(self, tmp_path):
        """Raises error for missing file."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            GlyphDataset(data_path=tmp_path / "nonexistent.pt")


class TestDataLoaders:
    """Tests for DataLoader creation."""

    def test_create_dataloaders(self):
        """create_dataloaders returns train and val loaders."""
        train_loader, val_loader = create_dataloaders()

        assert train_loader is not None
        assert val_loader is not None

    def test_train_val_split_sizes(self):
        """Train/val split produces correct sizes."""
        train_loader, val_loader = create_dataloaders()

        total_samples = 14040
        expected_train = int(total_samples * CONFIG.train_val_split)

        # Account for drop_last in train loader
        train_samples = len(train_loader) * CONFIG.batch_size
        val_samples = sum(len(batch["image"]) for batch in val_loader)

        # Train samples should be ~90% (minus dropped last batch)
        assert train_samples >= expected_train - CONFIG.batch_size
        assert val_samples > 0

    def test_batch_shapes(self):
        """Batches have correct shapes."""
        train_loader, _ = create_dataloaders()
        batch = next(iter(train_loader))

        assert batch["image"].shape == (CONFIG.batch_size, 1, 128, 128)
        assert batch["prev_char"].shape == (CONFIG.batch_size,)
        assert batch["curr_char"].shape == (CONFIG.batch_size,)

    def test_batch_dtypes(self):
        """Batches have correct dtypes."""
        train_loader, _ = create_dataloaders()
        batch = next(iter(train_loader))

        assert batch["image"].dtype == torch.float32
        assert batch["prev_char"].dtype == torch.int64
        assert batch["curr_char"].dtype == torch.int64

    def test_train_loader_shuffles(self):
        """Train loader produces different order on iteration."""
        train_loader, _ = create_dataloaders()

        batch1 = next(iter(train_loader))
        batch2 = next(iter(train_loader))

        # Different batches (unless extremely unlucky)
        # Note: After first iter, loader state is different
        assert not torch.equal(batch1["curr_char"], batch2["curr_char"]) or \
               not torch.equal(batch1["prev_char"], batch2["prev_char"])

    def test_val_loader_no_shuffle(self):
        """Val loader produces same order across epochs."""
        _, val_loader = create_dataloaders(seed=42)

        first_epoch_chars = []
        for batch in val_loader:
            first_epoch_chars.append(batch["curr_char"].clone())

        _, val_loader2 = create_dataloaders(seed=42)
        second_epoch_chars = []
        for batch in val_loader2:
            second_epoch_chars.append(batch["curr_char"].clone())

        for b1, b2 in zip(first_epoch_chars, second_epoch_chars):
            assert torch.equal(b1, b2), "Val loader should be deterministic"

    def test_custom_batch_size(self):
        """Custom batch size is respected."""
        train_loader, _ = create_dataloaders(batch_size=16)
        batch = next(iter(train_loader))

        assert batch["image"].shape[0] == 16

    def test_custom_split_ratio(self):
        """Custom train/val split is respected."""
        train_loader, val_loader = create_dataloaders(train_val_split=0.8)

        # With 0.8 split, train should have ~80% of 14040 = 11232 samples
        train_samples = len(train_loader) * CONFIG.batch_size
        assert train_samples >= 11000  # Approximate due to drop_last


class TestCreateTrainLoader:
    """Tests for create_train_loader (full dataset, no split)."""

    def test_full_dataset(self):
        """Returns loader for full dataset."""
        loader = create_train_loader()

        total_samples = len(loader) * CONFIG.batch_size
        expected = 14040 - (14040 % CONFIG.batch_size)  # Account for drop_last

        assert total_samples == expected

    def test_custom_batch_size(self):
        """Custom batch size works."""
        loader = create_train_loader(batch_size=32)
        batch = next(iter(loader))

        assert batch["image"].shape[0] == 32


class TestSplitReproducibility:
    """Tests for reproducible train/val splits."""

    def test_same_seed_same_split(self):
        """Same seed produces identical splits (verified via val loader which doesn't shuffle)."""
        _, val1 = create_dataloaders(seed=42)
        _, val2 = create_dataloaders(seed=42)

        # Val loader doesn't shuffle, so should produce identical batches
        batch1 = next(iter(val1))
        batch2 = next(iter(val2))

        # Validation batches should match with same seed
        assert torch.equal(batch1["image"], batch2["image"])

    def test_different_seed_different_split(self):
        """Different seeds produce different splits."""
        train1, _ = create_dataloaders(seed=42)
        train2, _ = create_dataloaders(seed=123)

        batch1 = next(iter(train1))
        batch2 = next(iter(train2))

        # First batches should differ with different seeds
        assert not torch.equal(batch1["curr_char"], batch2["curr_char"])


class TestDataIntegrity:
    """Tests for data integrity and consistency."""

    @pytest.fixture
    def dataset(self):
        return GlyphDataset()

    def test_consistent_indexing(self, dataset):
        """Same index returns same data."""
        sample1 = dataset[100]
        sample2 = dataset[100]

        assert torch.equal(sample1["image"], sample2["image"])
        assert sample1["prev_char"] == sample2["prev_char"]
        assert sample1["curr_char"] == sample2["curr_char"]

    def test_valid_image_distribution(self, dataset):
        """Images have reasonable pixel distributions."""
        # Sample a few images
        for i in [0, 1000, 5000, 10000]:
            sample = dataset[i]
            img = sample["image"]

            # Should have both dark and light regions
            dark_fraction = (img < 0.5).float().mean()
            light_fraction = (img >= 0.5).float().mean()

            # Characters are black on white, so most pixels should be light
            assert light_fraction > 0.5, f"Sample {i}: too few light pixels"
            # But there should be some dark pixels (the character)
            assert dark_fraction > 0.01, f"Sample {i}: no dark pixels"
