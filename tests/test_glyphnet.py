"""
Tests for GlyphNetwork.

Tests cover:
- Input/output shapes
- Output bounds [0, 1]
- Style variation produces different outputs
- Character variation produces different outputs
- Determinism with fixed seed
- Parameter count < 2.9M
- Batch processing consistency
- Input validation
"""

import pytest
import torch
import time

from src.glyphnet import GlyphNetwork, count_parameters, create_model


class TestGlyphNetworkBasic:
    """Basic shape and type tests."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return GlyphNetwork()

    def test_output_shape_single(self, model):
        """Test output shape for single sample."""
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        output = model(prev_char, curr_char, style_z)

        assert output.shape == (1, 1, 128, 128)

    def test_output_shape_batch(self, model):
        """Test output shape for batch."""
        batch_size = 8
        prev_char = torch.randint(0, 27, (batch_size,))
        curr_char = torch.randint(1, 27, (batch_size,))
        style_z = torch.randn(batch_size, 64)

        output = model(prev_char, curr_char, style_z)

        assert output.shape == (batch_size, 1, 128, 128)

    def test_output_bounds(self, model):
        """Test output values are in [0, 1]."""
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        output = model(prev_char, curr_char, style_z)

        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_output_bounds_extreme_style(self, model):
        """Test output bounds with extreme style vectors."""
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])

        # Large style vector
        style_z = torch.randn(1, 64) * 10
        output = model(prev_char, curr_char, style_z)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

        # Zero style vector
        style_z = torch.zeros(1, 64)
        output = model(prev_char, curr_char, style_z)
        assert output.min() >= 0.0
        assert output.max() <= 1.0


class TestGlyphNetworkInputValidation:
    """Input validation tests."""

    @pytest.fixture
    def model(self):
        return GlyphNetwork()

    def test_accepts_valid_prev_char_range(self, model):
        """Test prev_char accepts [0, 26]."""
        style_z = torch.randn(1, 64)

        # Test boundary values
        for prev_val in [0, 13, 26]:
            prev_char = torch.tensor([prev_val])
            curr_char = torch.tensor([1])
            output = model(prev_char, curr_char, style_z)
            assert output.shape == (1, 1, 128, 128)

    def test_accepts_valid_curr_char_range(self, model):
        """Test curr_char accepts [1, 26]."""
        style_z = torch.randn(1, 64)

        # Test boundary values
        for curr_val in [1, 13, 26]:
            prev_char = torch.tensor([0])
            curr_char = torch.tensor([curr_val])
            output = model(prev_char, curr_char, style_z)
            assert output.shape == (1, 1, 128, 128)

    def test_rejects_invalid_prev_char(self, model):
        """Test prev_char rejects values outside [0, 26]."""
        style_z = torch.randn(1, 64)
        curr_char = torch.tensor([1])

        with pytest.raises(ValueError, match="prev_char"):
            prev_char = torch.tensor([27])
            model(prev_char, curr_char, style_z)

    def test_rejects_invalid_curr_char_zero(self, model):
        """Test curr_char rejects 0."""
        style_z = torch.randn(1, 64)
        prev_char = torch.tensor([0])

        with pytest.raises(ValueError, match="curr_char"):
            curr_char = torch.tensor([0])
            model(prev_char, curr_char, style_z)

    def test_rejects_invalid_curr_char_high(self, model):
        """Test curr_char rejects values > 26."""
        style_z = torch.randn(1, 64)
        prev_char = torch.tensor([0])

        with pytest.raises(ValueError, match="curr_char"):
            curr_char = torch.tensor([27])
            model(prev_char, curr_char, style_z)

    def test_rejects_wrong_style_dim(self, model):
        """Test style_z rejects wrong dimension."""
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])

        with pytest.raises(ValueError, match="style_z"):
            style_z = torch.randn(1, 32)
            model(prev_char, curr_char, style_z)

    def test_handles_scalar_inputs(self, model):
        """Test model handles scalar tensor inputs."""
        prev_char = torch.tensor(0)
        curr_char = torch.tensor(1)
        style_z = torch.randn(64)

        output = model(prev_char, curr_char, style_z)
        assert output.shape == (1, 1, 128, 128)


class TestGlyphNetworkVariation:
    """Tests for output variation based on inputs."""

    @pytest.fixture
    def model(self):
        return GlyphNetwork()

    def test_style_variation(self, model):
        """Different style vectors produce different outputs."""
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])

        torch.manual_seed(42)
        z1 = torch.randn(1, 64)
        z2 = torch.randn(1, 64)

        out1 = model(prev_char, curr_char, z1)
        out2 = model(prev_char, curr_char, z2)

        assert not torch.allclose(out1, out2)

    def test_character_variation(self, model):
        """Different characters produce different outputs."""
        prev_char = torch.tensor([0])
        style_z = torch.randn(1, 64)

        out_a = model(prev_char, torch.tensor([1]), style_z)
        out_b = model(prev_char, torch.tensor([2]), style_z)

        assert not torch.allclose(out_a, out_b)

    def test_all_characters_different(self, model):
        """All 26 characters produce distinct outputs."""
        prev_char = torch.tensor([0])
        style_z = torch.randn(1, 64)

        outputs = []
        for char_idx in range(1, 27):
            curr_char = torch.tensor([char_idx])
            out = model(prev_char, curr_char, style_z)
            outputs.append(out)

        # Check that all outputs are pairwise different
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), \
                    f"Characters {i+1} and {j+1} produced identical outputs"

    def test_prev_char_affects_output(self, model):
        """Different prev_char values affect output."""
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        out1 = model(torch.tensor([0]), curr_char, style_z)
        out2 = model(torch.tensor([1]), curr_char, style_z)

        assert not torch.allclose(out1, out2)


class TestGlyphNetworkDeterminism:
    """Tests for deterministic behavior."""

    def test_deterministic_same_seed(self):
        """Same seed produces identical outputs."""
        torch.manual_seed(42)
        model1 = GlyphNetwork()

        torch.manual_seed(42)
        model2 = GlyphNetwork()

        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        # Need same style_z for both
        torch.manual_seed(123)
        style_z1 = torch.randn(1, 64)
        torch.manual_seed(123)
        style_z2 = torch.randn(1, 64)

        out1 = model1(prev_char, curr_char, style_z1)
        out2 = model2(prev_char, curr_char, style_z2)

        assert torch.allclose(out1, out2)

    def test_eval_mode_deterministic(self):
        """Eval mode produces deterministic outputs."""
        model = GlyphNetwork()
        model.eval()

        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        out1 = model(prev_char, curr_char, style_z)
        out2 = model(prev_char, curr_char, style_z)

        assert torch.allclose(out1, out2)


class TestGlyphNetworkBatchConsistency:
    """Tests for batch processing consistency."""

    @pytest.fixture
    def model(self):
        model = GlyphNetwork()
        model.eval()
        return model

    def test_batch_equals_individual(self, model):
        """Batch processing equals individual processing."""
        style_z = torch.randn(3, 64)
        prev_chars = torch.tensor([0, 5, 10])
        curr_chars = torch.tensor([1, 10, 20])

        # Batch inference
        batch_out = model(prev_chars, curr_chars, style_z)

        # Individual inference
        individual_outs = []
        for i in range(3):
            out = model(
                prev_chars[i:i+1],
                curr_chars[i:i+1],
                style_z[i:i+1]
            )
            individual_outs.append(out)

        individual_concat = torch.cat(individual_outs, dim=0)

        assert torch.allclose(batch_out, individual_concat, atol=1e-6)


class TestGlyphNetworkParameterCount:
    """Tests for model size constraints."""

    def test_parameter_count_under_limit(self):
        """Model has < 2.9M parameters."""
        model = GlyphNetwork()
        param_count = count_parameters(model)

        assert param_count < 2_900_000, \
            f"Model has {param_count:,} params, exceeds 2.9M limit"

    def test_parameter_count_reasonable(self):
        """Model has reasonable parameter count."""
        model = GlyphNetwork()
        param_count = count_parameters(model)

        # Expected ~1.5M (within 2.9M budget)
        assert 500_000 < param_count < 2_000_000, \
            f"Model has {param_count:,} params, expected 500K-2M"


class TestGlyphNetworkPerformance:
    """Performance tests."""

    def test_cpu_inference_under_100ms(self):
        """Single forward pass < 100ms on CPU."""
        model = GlyphNetwork()
        model.eval()

        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        # Warmup
        for _ in range(3):
            model(prev_char, curr_char, style_z)

        # Measure
        start = time.time()
        for _ in range(10):
            model(prev_char, curr_char, style_z)
        elapsed = (time.time() - start) / 10

        assert elapsed < 0.1, f"Inference took {elapsed*1000:.1f}ms, exceeds 100ms"


class TestCreateModel:
    """Tests for create_model helper."""

    def test_create_model_default(self):
        """create_model returns valid model."""
        model = create_model()
        assert isinstance(model, GlyphNetwork)

    def test_create_model_with_device(self):
        """create_model places model on device."""
        model = create_model(device=torch.device("cpu"))
        # Check model is on CPU
        assert next(model.parameters()).device.type == "cpu"


class TestGlyphNetworkEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def model(self):
        return GlyphNetwork()

    def test_start_token(self, model):
        """prev_char=0 (start token) works correctly."""
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        output = model(prev_char, curr_char, style_z)
        assert output.shape == (1, 1, 128, 128)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_same_char_repetition(self, model):
        """Same prev and curr char (e.g., 'aa') produces valid output."""
        for char_idx in [1, 13, 26]:
            prev_char = torch.tensor([char_idx])
            curr_char = torch.tensor([char_idx])
            style_z = torch.randn(1, 64)

            output = model(prev_char, curr_char, style_z)
            assert output.shape == (1, 1, 128, 128)

    def test_output_not_blank(self, model):
        """Output is not blank (all zeros)."""
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        output = model(prev_char, curr_char, style_z)

        # Output should have some variation
        assert output.std() > 0.01, "Output appears blank"
        assert output.max() > 0.1, "Output max pixel too low"

    def test_output_not_saturated(self, model):
        """Output is not saturated (all ones)."""
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])
        style_z = torch.randn(1, 64)

        output = model(prev_char, curr_char, style_z)

        # Output should not be all white
        assert output.min() < 0.9, "Output appears saturated"
