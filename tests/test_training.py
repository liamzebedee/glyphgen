"""
Tests for training pipeline.

Tests cover:
- Loss function computation
- Edge loss with Sobel gradients
- Single training step
- Validation step
- Checkpoint save/resume
- Training loop runs without error
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.train import (
    compute_edge_loss,
    compute_loss,
    train_one_epoch,
    validate,
    train,
)
from src.glyphnet import GlyphNetwork
from src.config import CONFIG


class TestLossFunctions:
    """Tests for loss computation."""

    def test_compute_loss_returns_three_tensors(self):
        """Test compute_loss returns total, recon, and edge losses."""
        output = torch.rand(4, 1, 128, 128)
        target = torch.rand(4, 1, 128, 128)

        total, recon, edge = compute_loss(output, target, edge_weight=0.1)

        assert isinstance(total, torch.Tensor)
        assert isinstance(recon, torch.Tensor)
        assert isinstance(edge, torch.Tensor)
        assert total.dim() == 0  # scalar
        assert recon.dim() == 0
        assert edge.dim() == 0

    def test_compute_loss_positive(self):
        """Test all loss components are non-negative."""
        output = torch.rand(4, 1, 128, 128)
        target = torch.rand(4, 1, 128, 128)

        total, recon, edge = compute_loss(output, target)

        assert total >= 0
        assert recon >= 0
        assert edge >= 0

    def test_compute_loss_zero_for_identical_inputs(self):
        """Test reconstruction loss is zero when output equals target."""
        image = torch.rand(4, 1, 128, 128)

        total, recon, edge = compute_loss(image, image)

        assert recon.item() == pytest.approx(0.0, abs=1e-6)

    def test_compute_loss_weight_affects_total(self):
        """Test edge weight affects total loss."""
        output = torch.rand(4, 1, 128, 128)
        target = torch.rand(4, 1, 128, 128)

        total_low, _, edge = compute_loss(output, target, edge_weight=0.1)
        total_high, _, _ = compute_loss(output, target, edge_weight=1.0)

        # Higher weight should give higher total loss (when edge > 0)
        if edge > 0:
            assert total_high > total_low


class TestEdgeLoss:
    """Tests for edge-aware loss."""

    def test_edge_loss_shape(self):
        """Test edge loss is a scalar."""
        output = torch.rand(4, 1, 128, 128)
        target = torch.rand(4, 1, 128, 128)

        edge_loss = compute_edge_loss(output, target)

        assert edge_loss.dim() == 0

    def test_edge_loss_positive(self):
        """Test edge loss is non-negative."""
        output = torch.rand(4, 1, 128, 128)
        target = torch.rand(4, 1, 128, 128)

        edge_loss = compute_edge_loss(output, target)

        assert edge_loss >= 0

    def test_edge_loss_zero_for_identical(self):
        """Test edge loss is zero for identical images."""
        image = torch.rand(4, 1, 128, 128)

        edge_loss = compute_edge_loss(image, image)

        assert edge_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_edge_loss_detects_edge_difference(self):
        """Test edge loss is higher when edges differ."""
        # Create images with different edge content
        smooth = torch.ones(1, 1, 128, 128) * 0.5
        with_edges = torch.ones(1, 1, 128, 128) * 0.5
        with_edges[:, :, 60:68, :] = 1.0  # horizontal line

        edge_loss_same = compute_edge_loss(smooth, smooth)
        edge_loss_diff = compute_edge_loss(smooth, with_edges)

        assert edge_loss_diff > edge_loss_same

    def test_edge_loss_gradient_flows(self):
        """Test gradients flow through edge loss."""
        output = torch.rand(2, 1, 128, 128, requires_grad=True)
        target = torch.rand(2, 1, 128, 128)

        edge_loss = compute_edge_loss(output, target)
        edge_loss.backward()

        assert output.grad is not None
        assert not torch.all(output.grad == 0)


class TestTrainingStep:
    """Tests for single training steps."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return GlyphNetwork()

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader with 2 batches."""
        batch_size = 4
        batches = []
        for _ in range(2):
            batches.append({
                "image": torch.rand(batch_size, 1, 128, 128),
                "prev_char": torch.randint(0, 27, (batch_size,)),
                "curr_char": torch.randint(1, 27, (batch_size,)),
            })
        return batches

    def test_train_one_epoch_returns_losses(self, model, mock_dataloader):
        """Test train_one_epoch returns three loss values."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        style_z = torch.zeros(64)
        device = torch.device("cpu")

        total, recon, edge = train_one_epoch(
            model,
            mock_dataloader,
            optimizer,
            device,
            style_z,
            edge_weight=0.1,
        )

        assert isinstance(total, float)
        assert isinstance(recon, float)
        assert isinstance(edge, float)
        assert total > 0
        assert recon >= 0
        assert edge >= 0

    def test_train_step_updates_weights(self, model, mock_dataloader):
        """Test training step actually updates model weights."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        style_z = torch.zeros(64)
        device = torch.device("cpu")

        # Get initial weights
        initial_weights = model.mlp[0].weight.clone()

        train_one_epoch(
            model,
            mock_dataloader,
            optimizer,
            device,
            style_z,
            edge_weight=0.1,
        )

        # Weights should have changed
        assert not torch.allclose(model.mlp[0].weight, initial_weights)

    def test_validate_returns_losses(self, model, mock_dataloader):
        """Test validate returns three loss values."""
        style_z = torch.zeros(64)
        device = torch.device("cpu")

        total, recon, edge = validate(
            model,
            mock_dataloader,
            device,
            style_z,
            edge_weight=0.1,
        )

        assert isinstance(total, float)
        assert isinstance(recon, float)
        assert isinstance(edge, float)

    def test_validate_does_not_update_weights(self, model, mock_dataloader):
        """Test validation does not change model weights."""
        style_z = torch.zeros(64)
        device = torch.device("cpu")

        initial_weights = model.mlp[0].weight.clone()

        validate(
            model,
            mock_dataloader,
            device,
            style_z,
            edge_weight=0.1,
        )

        assert torch.allclose(model.mlp[0].weight, initial_weights)


class TestGradientStability:
    """Tests for gradient stability during training."""

    def test_no_nan_gradients(self):
        """Test gradients don't become NaN."""
        model = GlyphNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = {
            "image": torch.rand(4, 1, 128, 128),
            "prev_char": torch.randint(0, 27, (4,)),
            "curr_char": torch.randint(1, 27, (4,)),
        }
        style_z = torch.randn(64)

        for _ in range(5):
            optimizer.zero_grad()
            output = model(
                batch["prev_char"],
                batch["curr_char"],
                style_z.unsqueeze(0).expand(4, -1),
            )
            total_loss, _, _ = compute_loss(output, batch["image"])
            total_loss.backward()

            # Check no NaN gradients
            for param in model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
                    assert not torch.isinf(param.grad).any()

            optimizer.step()

    def test_loss_decreases_over_steps(self):
        """Test loss generally decreases with training."""
        model = GlyphNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed target to overfit
        target = torch.rand(4, 1, 128, 128)
        prev_char = torch.randint(0, 27, (4,))
        curr_char = torch.randint(1, 27, (4,))
        style_z = torch.zeros(4, 64)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            output = model(prev_char, curr_char, style_z)
            total_loss, _, _ = compute_loss(output, target)
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # Final loss should be less than initial
        assert losses[-1] < losses[0]


class TestCheckpointing:
    """Tests for checkpoint save/resume."""

    def test_checkpoint_roundtrip(self):
        """Test model can be saved and loaded."""
        model1 = GlyphNetwork()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        style_z = torch.randn(64)

        # Do some training to change weights
        prev_char = torch.randint(0, 27, (4,))
        curr_char = torch.randint(1, 27, (4,))
        target = torch.rand(4, 1, 128, 128)

        optimizer1.zero_grad()
        output = model1(prev_char, curr_char, style_z.unsqueeze(0).expand(4, -1))
        loss, _, _ = compute_loss(output, target)
        loss.backward()
        optimizer1.step()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "epoch": 5,
                    "model_state_dict": model1.state_dict(),
                    "optimizer_state_dict": optimizer1.state_dict(),
                    "style_z": style_z,
                    "loss": loss.item(),
                },
                checkpoint_path,
            )

            # Load into new model
            model2 = GlyphNetwork()
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model2.load_state_dict(checkpoint["model_state_dict"])

            # Compare outputs
            model1.eval()
            model2.eval()
            with torch.no_grad():
                out1 = model1(prev_char, curr_char, style_z.unsqueeze(0).expand(4, -1))
                out2 = model2(prev_char, curr_char, style_z.unsqueeze(0).expand(4, -1))

            assert torch.allclose(out1, out2)

    def test_checkpoint_contains_style_z(self):
        """Test checkpoint includes style_z."""
        model = GlyphNetwork()
        style_z = torch.randn(64)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "epoch": 0,
                    "model_state_dict": model.state_dict(),
                    "style_z": style_z,
                    "loss": 0.5,
                },
                checkpoint_path,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            loaded_style_z = checkpoint["style_z"]

            assert torch.allclose(style_z, loaded_style_z)


class TestFullTraining:
    """Integration tests for full training loop."""

    @pytest.mark.slow
    def test_train_runs_one_epoch(self):
        """Test train function runs without error for 1 epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # This requires the actual dataset to exist
            try:
                model, style_z = train(
                    epochs=1,
                    checkpoint_dir=Path(tmpdir),
                    device=torch.device("cpu"),
                )

                assert model is not None
                assert style_z is not None
                assert style_z.shape == (CONFIG.style_dim,)

                # Check checkpoint was created
                assert (Path(tmpdir) / "final_checkpoint.pt").exists()
            except FileNotFoundError:
                pytest.skip("Dataset not found - run generate_dataset.py first")
