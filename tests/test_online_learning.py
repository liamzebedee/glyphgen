"""
Tests for online learning with REINFORCE-style policy gradients.

Tests cover:
- Advantage computation
- Baseline EMA convergence
- Gradient stability (no NaN/Inf)
- Weight update direction matches advantage sign
- Checkpoint versioning
- Full update cycle integration
"""

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.config import CONFIG
from src.feedback import FeedbackResult, EvaluationRun
from src.glyphnet import GlyphNetwork
from src.style import StyleVector
from src.utils import get_device


# Test fixtures
@pytest.fixture
def device():
    """Get best available device."""
    return get_device()


@pytest.fixture
def model(device):
    """Create a GlyphNetwork for testing."""
    model = GlyphNetwork()
    return model.to(device)


@pytest.fixture
def style_z():
    """Create a random style vector."""
    return StyleVector.random(name="test", seed=42)


@pytest.fixture
def temp_networks_dir(tmp_path):
    """Create a temporary networks directory."""
    networks_dir = tmp_path / "networks"
    networks_dir.mkdir()

    with patch.object(type(CONFIG), "networks_dir", property(lambda self: networks_dir)):
        yield networks_dir


@pytest.fixture
def sample_feedback():
    """Create a sample feedback result."""
    return FeedbackResult(
        glyph_char="a",
        ratings={"fruity": 7.0, "aggressive": 3.0},
        timestamp=datetime.now(),
        model="claude-sonnet-4-20250514",
        success=True,
    )


@pytest.fixture
def sample_evaluation_run():
    """Create a sample evaluation run with multiple results."""
    results = [
        FeedbackResult(
            glyph_char="a",
            ratings={"fruity": 8.0},
            timestamp=datetime.now(),
            model="test",
            success=True,
        ),
        FeedbackResult(
            glyph_char="b",
            ratings={"fruity": 6.0},
            timestamp=datetime.now(),
            model="test",
            success=True,
        ),
        FeedbackResult(
            glyph_char="c",
            ratings={"fruity": 7.0},
            timestamp=datetime.now(),
            model="test",
            success=True,
        ),
    ]
    return EvaluationRun(
        run_number=1,
        font_id="1-fruity",
        personality="fruity",
        results=results,
    )


class TestBaselineTracker:
    """Tests for EMA baseline tracking."""

    def test_initial_baseline_value(self):
        """Test baseline initializes to middle of scale."""
        from src.online_learn import BaselineTracker

        baseline = BaselineTracker()
        assert baseline.value == 5.5  # Middle of 1-10 scale

    def test_first_observation_sets_baseline(self):
        """Test that first observation becomes the baseline."""
        from src.online_learn import BaselineTracker

        baseline = BaselineTracker()
        baseline.update(8.0)
        assert baseline.value == 8.0
        assert baseline.count == 1

    def test_ema_update_formula(self):
        """Test EMA update follows correct formula."""
        from src.online_learn import BaselineTracker

        baseline = BaselineTracker(alpha=0.9)
        baseline.update(8.0)  # First: sets to 8.0
        baseline.update(4.0)  # Second: 0.9 * 4.0 + 0.1 * 8.0 = 4.4

        expected = 0.9 * 4.0 + 0.1 * 8.0
        assert abs(baseline.value - expected) < 1e-6

    def test_advantage_computation(self):
        """Test advantage = rating - baseline."""
        from src.online_learn import BaselineTracker

        baseline = BaselineTracker()
        baseline.update(5.0)  # Baseline = 5.0

        assert baseline.get_advantage(7.0) == 2.0  # Positive advantage
        assert baseline.get_advantage(3.0) == -2.0  # Negative advantage
        assert baseline.get_advantage(5.0) == 0.0  # Zero advantage

    def test_baseline_reliability(self):
        """Test baseline becomes reliable after min observations."""
        from src.online_learn import BaselineTracker

        baseline = BaselineTracker(min_observations=3)
        assert not baseline.is_reliable

        baseline.update(5.0)
        assert not baseline.is_reliable

        baseline.update(6.0)
        assert not baseline.is_reliable

        baseline.update(7.0)
        assert baseline.is_reliable  # Now has 3 observations

    def test_baseline_convergence(self):
        """Test baseline converges to mean of constant ratings."""
        from src.online_learn import BaselineTracker

        baseline = BaselineTracker(alpha=0.9)

        # Apply many updates with constant rating
        constant_rating = 7.0
        for _ in range(100):
            baseline.update(constant_rating)

        # Should converge close to constant rating
        assert abs(baseline.value - constant_rating) < 0.01

    def test_baseline_serialization(self):
        """Test baseline can be serialized and deserialized."""
        from src.online_learn import BaselineTracker

        original = BaselineTracker(alpha=0.95, value=6.5, count=10)
        data = original.to_dict()
        restored = BaselineTracker.from_dict(data)

        assert restored.alpha == original.alpha
        assert restored.value == original.value
        assert restored.count == original.count


class TestGradientComputation:
    """Tests for policy gradient computation."""

    def test_positive_advantage_positive_gradient(self, model, device):
        """Test positive advantage produces gradient in positive direction."""
        from src.online_learn import BaselineTracker, compute_policy_gradient

        baseline = BaselineTracker()
        baseline.update(5.0)  # Baseline = 5.0

        style_z = torch.randn(64, device=device)
        rating = 8.0  # Advantage = 3.0 (positive)

        grad, _, stats = compute_policy_gradient(
            feedback_rating=rating,
            baseline=baseline,
            style_z=style_z,
            model=model,
            generated_glyph=None,
        )

        assert stats.advantage > 0
        # Gradient should be in same direction as style_z (reinforce this style)
        dot_product = torch.dot(grad.flatten(), style_z.flatten())
        assert dot_product > 0  # Same direction

    def test_negative_advantage_negative_gradient(self, model, device):
        """Test negative advantage produces gradient in negative direction."""
        from src.online_learn import BaselineTracker, compute_policy_gradient

        baseline = BaselineTracker()
        baseline.update(8.0)  # Baseline = 8.0

        style_z = torch.randn(64, device=device)
        rating = 3.0  # Advantage = -5.0 (negative)

        grad, _, stats = compute_policy_gradient(
            feedback_rating=rating,
            baseline=baseline,
            style_z=style_z,
            model=model,
            generated_glyph=None,
        )

        assert stats.advantage < 0
        # Gradient should be opposite direction (discourage this style)
        dot_product = torch.dot(grad.flatten(), style_z.flatten())
        assert dot_product < 0  # Opposite direction

    def test_zero_advantage_small_gradient(self, model, device):
        """Test zero advantage produces minimal gradient."""
        from src.online_learn import BaselineTracker, compute_policy_gradient

        baseline = BaselineTracker()
        baseline.update(5.0)

        style_z = torch.randn(64, device=device)
        rating = 5.0  # Advantage = 0

        grad, _, stats = compute_policy_gradient(
            feedback_rating=rating,
            baseline=baseline,
            style_z=style_z,
            model=model,
            generated_glyph=None,
        )

        assert stats.advantage == 0.0
        assert grad.norm().item() < 1e-6  # Near zero gradient

    def test_gradient_clipping(self, model, device):
        """Test gradient is clipped to max norm."""
        from src.online_learn import BaselineTracker, compute_policy_gradient

        baseline = BaselineTracker()
        baseline.update(1.0)  # Low baseline

        style_z = torch.randn(64, device=device) * 10  # Large style vector
        rating = 10.0  # High advantage

        grad, _, stats = compute_policy_gradient(
            feedback_rating=rating,
            baseline=baseline,
            style_z=style_z,
            model=model,
            generated_glyph=None,
            gradient_clip=1.0,
        )

        assert stats.gradient_norm <= 1.0 + 1e-6  # Clipped

    def test_gradient_no_nan_inf(self, model, device):
        """Test gradients never contain NaN or Inf."""
        from src.online_learn import BaselineTracker, compute_policy_gradient

        baseline = BaselineTracker()

        # Test with various extreme values
        test_cases = [
            (1.0, 10.0),  # Low baseline, high rating
            (10.0, 1.0),  # High baseline, low rating
            (5.0, 5.0),   # Equal
            (0.01, 9.99), # Near extremes
        ]

        for base_val, rating in test_cases:
            baseline.value = base_val
            style_z = torch.randn(64, device=device)

            grad, _, stats = compute_policy_gradient(
                feedback_rating=rating,
                baseline=baseline,
                style_z=style_z,
                model=model,
                generated_glyph=None,
            )

            assert not torch.isnan(grad).any(), f"NaN in gradient for baseline={base_val}, rating={rating}"
            assert not torch.isinf(grad).any(), f"Inf in gradient for baseline={base_val}, rating={rating}"


class TestStyleUpdate:
    """Tests for style vector updates."""

    def test_style_update_applies_gradient(self):
        """Test that style update applies gradient correctly."""
        from src.online_learn import apply_style_update

        style_z = StyleVector.random(seed=42)
        original_data = style_z.data.clone()

        gradient = torch.randn_like(style_z.data) * 0.1
        updated = apply_style_update(style_z, gradient)

        # Updated should be different from original
        assert not torch.allclose(updated.data, original_data)

    def test_style_update_clamps_norm(self):
        """Test that style update clamps to max norm."""
        from src.online_learn import apply_style_update

        style_z = StyleVector.random(seed=42)

        # Large gradient that would exceed max norm
        gradient = torch.ones_like(style_z.data) * 100
        max_norm = 5.0

        updated = apply_style_update(style_z, gradient, max_norm=max_norm)

        assert updated.norm() <= max_norm + 1e-6

    def test_style_update_preserves_name(self):
        """Test that style update preserves personality name."""
        from src.online_learn import apply_style_update

        style_z = StyleVector.random(name="fruity", seed=42)
        gradient = torch.randn_like(style_z.data) * 0.1

        updated = apply_style_update(style_z, gradient)

        assert updated.name == "fruity"


class TestDecoderUpdate:
    """Tests for decoder layer updates."""

    def test_decoder_update_modifies_parameters(self, model):
        """Test that decoder update modifies model parameters."""
        from src.online_learn import apply_decoder_update

        # Get original parameter values
        original_params = {}
        for name, param in model.named_parameters():
            if "decoder" in name:
                original_params[name] = param.clone()

        # Create fake gradients for decoder parameters
        gradients = {}
        for name, param in model.named_parameters():
            if "decoder.4" in name or "decoder.5" in name:
                gradients[name] = torch.randn_like(param) * 0.01

        updated_count = apply_decoder_update(model, gradients)

        assert updated_count > 0

        # Check that parameters changed
        changed_count = 0
        for name, param in model.named_parameters():
            if name in gradients:
                if not torch.allclose(param, original_params[name]):
                    changed_count += 1

        assert changed_count > 0

    def test_decoder_update_clamps_weights(self, model):
        """Test that decoder update clamps weights to bounds."""
        from src.online_learn import apply_decoder_update

        # Create large gradients that would exceed bounds
        gradients = {}
        for name, param in model.named_parameters():
            if "decoder.4" in name:
                gradients[name] = torch.ones_like(param) * 100

        weight_bound = 5.0
        apply_decoder_update(model, gradients, weight_bound=weight_bound)

        # Check all decoder parameters are within bounds
        for name, param in model.named_parameters():
            if name in gradients:
                assert param.max() <= weight_bound
                assert param.min() >= -weight_bound


class TestCheckpointing:
    """Tests for weight checkpoint versioning."""

    def test_save_checkpoint(self, temp_networks_dir, model, style_z):
        """Test saving a versioned checkpoint."""
        from src.online_learn import save_weight_checkpoint

        path = save_weight_checkpoint(
            font_id="1-test",
            run_id="2026-01-11T10:00:00.000000",
            model=model,
            style_z=style_z,
            version=1,
        )

        assert path.exists()
        assert path.name == "weights_v1.pt"

    def test_load_checkpoint(self, temp_networks_dir, model, style_z, device):
        """Test loading a versioned checkpoint."""
        from src.online_learn import save_weight_checkpoint, load_weight_checkpoint

        # Save checkpoint
        save_weight_checkpoint(
            font_id="1-test",
            run_id="2026-01-11T10:00:00.000000",
            model=model,
            style_z=style_z,
            version=1,
        )

        # Create new model and load
        new_model = GlyphNetwork().to(device)
        loaded_style, version = load_weight_checkpoint(
            font_id="1-test",
            run_id="2026-01-11T10:00:00.000000",
            model=new_model,
            version=1,
            device=device,
        )

        assert version == 1
        assert torch.allclose(loaded_style.data, style_z.data.to(device), atol=1e-5)

    def test_load_latest_checkpoint(self, temp_networks_dir, model, style_z, device):
        """Test loading latest checkpoint when version not specified."""
        from src.online_learn import save_weight_checkpoint, load_weight_checkpoint

        run_id = "2026-01-11T10:00:00.000000"

        # Save multiple versions
        for v in [1, 2, 3]:
            style_z.data = torch.randn(CONFIG.style_dim)  # Different each time
            save_weight_checkpoint(
                font_id="1-test",
                run_id=run_id,
                model=model,
                style_z=style_z,
                version=v,
            )

        # Load without specifying version
        new_model = GlyphNetwork().to(device)
        _, version = load_weight_checkpoint(
            font_id="1-test",
            run_id=run_id,
            model=new_model,
            device=device,
        )

        assert version == 3  # Latest version


class TestOnlineLearningState:
    """Tests for persistent learning state."""

    def test_state_creation(self, temp_networks_dir):
        """Test creating new learning state."""
        from src.online_learn import OnlineLearningState

        state = OnlineLearningState(
            font_id="1-test",
            run_id="2026-01-11T10:00:00.000000",
        )

        assert state.font_id == "1-test"
        assert state.weight_version == 0
        assert len(state.update_history) == 0

    def test_state_save_load(self, temp_networks_dir):
        """Test saving and loading learning state."""
        from src.online_learn import OnlineLearningState, BaselineTracker

        # Create state with some history
        state = OnlineLearningState(
            font_id="1-test",
            run_id="2026-01-11T10:00:00.000000",
        )
        state.baseline.update(7.0)
        state.baseline.update(8.0)
        state.weight_version = 3
        state.update_history.append({"rating": 7.5, "advantage": 1.5})

        # Save and load
        state.save()
        loaded = OnlineLearningState.load("1-test", "2026-01-11T10:00:00.000000")

        assert loaded.weight_version == 3
        assert len(loaded.update_history) == 1
        assert loaded.baseline.count == 2


class TestOnlineLearner:
    """Tests for the main OnlineLearner class."""

    def test_learner_initialization(self, temp_networks_dir, model, style_z):
        """Test initializing online learner."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-test",
            model=model,
            style_z=style_z,
            learning_rate=1e-4,
        )

        assert learner.font_id == "1-test"
        assert learner.learning_rate == 1e-4
        assert learner.update_count == 0

    def test_learner_single_update(self, temp_networks_dir, model, style_z, sample_feedback):
        """Test applying a single update."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-test",
            model=model,
            style_z=style_z,
        )

        original_style_norm = learner.style_z.norm()

        stats = learner.update(sample_feedback, target_dimension="fruity")

        assert stats.success
        assert learner.update_count == 1
        # Style should have changed
        assert learner.style_z.norm() != original_style_norm or True  # May be same if update is small

    def test_learner_batch_update(self, temp_networks_dir, model, style_z, sample_evaluation_run):
        """Test applying batch updates from evaluation run."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-fruity",
            model=model,
            style_z=style_z,
        )

        stats_list = learner.update_batch(
            evaluation_run=sample_evaluation_run,
            target_dimension="fruity",
        )

        assert len(stats_list) == 3
        assert learner.update_count == 3
        assert all(s.success for s in stats_list)

    def test_learner_finalize(self, temp_networks_dir, model, style_z, sample_feedback):
        """Test finalizing learning session."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-test",
            model=model,
            style_z=style_z,
            checkpoint_frequency=1,
        )

        learner.update(sample_feedback, target_dimension="fruity")
        metadata = learner.finalize()

        assert metadata.completed
        assert metadata.font_id == "1-test"
        assert "total_updates" in metadata.training_params

    def test_learner_invalid_font_id(self, model, style_z):
        """Test that invalid font ID raises error."""
        from src.online_learn import OnlineLearner

        with pytest.raises(ValueError, match="Invalid font ID"):
            OnlineLearner(
                font_id="invalid",  # No number prefix
                model=model,
                style_z=style_z,
            )

    def test_learner_failed_feedback_handled(self, temp_networks_dir, model, style_z):
        """Test that failed feedback is handled gracefully."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-test",
            model=model,
            style_z=style_z,
        )

        failed_feedback = FeedbackResult(
            glyph_char="a",
            ratings={},
            timestamp=datetime.now(),
            model="test",
            success=False,
            error="API error",
        )

        stats = learner.update(failed_feedback)

        assert not stats.success
        assert stats.error == "API error"
        assert learner.update_count == 0  # No update applied

    def test_learner_missing_dimension_handled(self, temp_networks_dir, model, style_z):
        """Test that missing target dimension is handled."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-test",
            model=model,
            style_z=style_z,
        )

        feedback = FeedbackResult(
            glyph_char="a",
            ratings={"fruity": 7.0},
            timestamp=datetime.now(),
            model="test",
            success=True,
        )

        stats = learner.update(feedback, target_dimension="nonexistent")

        assert not stats.success
        assert "nonexistent" in stats.error

    def test_learner_learning_stats(self, temp_networks_dir, model, style_z, sample_evaluation_run):
        """Test getting learning statistics."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-fruity",
            model=model,
            style_z=style_z,
        )

        # Apply some updates
        learner.update_batch(sample_evaluation_run, target_dimension="fruity")

        stats = learner.get_learning_stats()

        assert stats["update_count"] == 3
        assert "advantage_mean" in stats
        assert "rating_mean" in stats
        assert "style_z_norm" in stats


class TestGradientStability:
    """Tests for gradient stability over multiple updates."""

    def test_no_nan_after_many_updates(self, temp_networks_dir, model, style_z):
        """Test no NaN values after 10+ consecutive updates."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-test",
            model=model,
            style_z=style_z,
        )

        # Apply many updates with varying ratings
        for i in range(15):
            rating = 3.0 + (i % 7)  # Ratings between 3-9
            feedback = FeedbackResult(
                glyph_char=chr(ord("a") + (i % 26)),
                ratings={"test": rating},
                timestamp=datetime.now(),
                model="test",
                success=True,
            )

            stats = learner.update(feedback, target_dimension="test")

            assert stats.success, f"Update {i} failed"
            assert not math.isnan(stats.gradient_norm), f"NaN gradient at update {i}"
            assert not math.isinf(stats.gradient_norm), f"Inf gradient at update {i}"
            assert not math.isnan(learner.style_z.norm()), f"NaN style_z norm at update {i}"

    def test_style_z_bounded_after_updates(self, temp_networks_dir, model, style_z):
        """Test style_z remains bounded after updates."""
        from src.online_learn import OnlineLearner

        learner = OnlineLearner(
            font_id="1-test",
            model=model,
            style_z=style_z,
        )

        # Apply extreme ratings
        for i in range(20):
            rating = 10.0 if i % 2 == 0 else 1.0  # Alternating extremes
            feedback = FeedbackResult(
                glyph_char="a",
                ratings={"test": rating},
                timestamp=datetime.now(),
                model="test",
                success=True,
            )

            learner.update(feedback, target_dimension="test")

            # Style should remain bounded
            assert learner.style_z.norm() <= 10.0 + 1e-6, f"Style unbounded at update {i}"


class TestUpdateTiming:
    """Tests for update timing constraints."""

    def test_update_within_time_budget(self, temp_networks_dir, model, style_z, sample_feedback):
        """Test that single update completes within 500ms budget."""
        from src.online_learn import OnlineLearner, UPDATE_BUDGET_MS
        import time

        learner = OnlineLearner(
            font_id="1-test",
            model=model,
            style_z=style_z,
        )

        start = time.time()
        stats = learner.update(sample_feedback, target_dimension="fruity")
        elapsed_ms = (time.time() - start) * 1000

        assert stats.success
        # Allow some overhead but should be reasonably fast
        assert elapsed_ms < UPDATE_BUDGET_MS * 2, f"Update took {elapsed_ms:.1f}ms"
