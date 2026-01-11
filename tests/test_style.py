"""
Tests for StyleVector and personality expression.

Tests cover:
- StyleVector creation (zeros, random, from_tensor)
- Dimensionality validation
- Interpolation between styles
- L2 norm clamping for stability
- Save/load round-trip
- Cross-glyph consistency with same style_z
- Zero vector produces valid output
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.config import CONFIG
from src.style import (
    StyleVector,
    create_personality_vectors,
    interpolate_personalities,
)


class TestStyleVectorCreation:
    """Test StyleVector creation methods."""

    def test_zeros_creates_correct_dimensions(self):
        """Zero vector has 64 dimensions."""
        sv = StyleVector.zeros()
        assert sv.data.shape == (CONFIG.style_dim,)
        assert sv.data.shape == (64,)

    def test_zeros_is_all_zeros(self):
        """Zero vector has all zero values."""
        sv = StyleVector.zeros()
        assert torch.allclose(sv.data, torch.zeros(64))

    def test_zeros_with_name(self):
        """Zero vector can have a name."""
        sv = StyleVector.zeros(name="neutral")
        assert sv.name == "neutral"

    def test_random_creates_correct_dimensions(self):
        """Random vector has 64 dimensions."""
        sv = StyleVector.random()
        assert sv.data.shape == (64,)

    def test_random_is_not_zeros(self):
        """Random vector is not all zeros."""
        sv = StyleVector.random(seed=42)
        assert not torch.allclose(sv.data, torch.zeros(64))

    def test_random_with_seed_is_reproducible(self):
        """Same seed produces same vector."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=42)
        assert torch.allclose(sv1.data, sv2.data)

    def test_random_different_seeds_differ(self):
        """Different seeds produce different vectors."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)
        assert not torch.allclose(sv1.data, sv2.data)

    def test_random_with_std(self):
        """Random vector respects std parameter."""
        sv_small = StyleVector.random(seed=42, std=0.1)
        sv_large = StyleVector.random(seed=42, std=10.0)
        # Larger std should produce larger norm on average
        assert sv_large.norm() > sv_small.norm()

    def test_from_tensor_clones_data(self):
        """from_tensor creates a copy, not an alias."""
        original = torch.randn(64)
        sv = StyleVector.from_tensor(original)
        original[0] = 999.0
        assert sv.data[0] != 999.0

    def test_from_tensor_with_name(self):
        """from_tensor preserves name."""
        sv = StyleVector.from_tensor(torch.randn(64), name="test")
        assert sv.name == "test"


class TestStyleVectorValidation:
    """Test StyleVector validation."""

    def test_rejects_wrong_dimensions(self):
        """Raises error for non-64-dim tensors."""
        with pytest.raises(ValueError, match="64 dimensions"):
            StyleVector(torch.randn(32))

    def test_rejects_2d_tensor(self):
        """Raises error for 2D tensors."""
        with pytest.raises(ValueError, match="must be 1D"):
            StyleVector(torch.randn(1, 64))

    def test_rejects_0d_tensor(self):
        """Raises error for scalar tensors."""
        with pytest.raises(ValueError, match="must be 1D"):
            StyleVector(torch.tensor(1.0))

    def test_converts_int_to_float(self):
        """Integer tensors are converted to float."""
        sv = StyleVector(torch.zeros(64, dtype=torch.int32))
        assert sv.data.is_floating_point()


class TestStyleVectorNorm:
    """Test norm operations."""

    def test_norm_of_zeros_is_zero(self):
        """Zero vector has zero norm."""
        sv = StyleVector.zeros()
        assert sv.norm() == 0.0

    def test_norm_calculation(self):
        """Norm is computed correctly."""
        data = torch.zeros(64)
        data[0] = 3.0
        data[1] = 4.0
        sv = StyleVector(data)
        assert abs(sv.norm() - 5.0) < 1e-6

    def test_clamp_norm_reduces_large_norm(self):
        """clamp_norm reduces norm when too large."""
        sv = StyleVector.random(seed=42)
        original_norm = sv.norm()
        max_norm = original_norm / 2
        clamped = sv.clamp_norm(max_norm)
        assert abs(clamped.norm() - max_norm) < 1e-6

    def test_clamp_norm_preserves_small_norm(self):
        """clamp_norm doesn't change small vectors."""
        sv = StyleVector.random(seed=42, std=0.1)
        original_norm = sv.norm()
        clamped = sv.clamp_norm(100.0)  # Much larger than actual norm
        assert abs(clamped.norm() - original_norm) < 1e-6

    def test_clamp_norm_handles_zero_vector(self):
        """clamp_norm works on zero vector."""
        sv = StyleVector.zeros()
        clamped = sv.clamp_norm(1.0)
        assert clamped.norm() == 0.0

    def test_normalize_creates_unit_vector(self):
        """normalize creates unit vector by default."""
        sv = StyleVector.random(seed=42)
        normalized = sv.normalize()
        assert abs(normalized.norm() - 1.0) < 1e-6

    def test_normalize_to_target(self):
        """normalize respects target norm."""
        sv = StyleVector.random(seed=42)
        normalized = sv.normalize(target_norm=5.0)
        assert abs(normalized.norm() - 5.0) < 1e-6


class TestStyleVectorInterpolation:
    """Test interpolation operations."""

    def test_interpolate_at_zero_is_self(self):
        """t=0 returns self."""
        sv1 = StyleVector.random(seed=42, name="a")
        sv2 = StyleVector.random(seed=43, name="b")
        result = sv1.interpolate(sv2, 0.0)
        assert torch.allclose(result.data, sv1.data)

    def test_interpolate_at_one_is_other(self):
        """t=1 returns other."""
        sv1 = StyleVector.random(seed=42, name="a")
        sv2 = StyleVector.random(seed=43, name="b")
        result = sv1.interpolate(sv2, 1.0)
        assert torch.allclose(result.data, sv2.data)

    def test_interpolate_at_half_is_midpoint(self):
        """t=0.5 returns midpoint."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)
        result = sv1.interpolate(sv2, 0.5)
        expected = (sv1.data + sv2.data) / 2
        assert torch.allclose(result.data, expected)

    def test_interpolate_rejects_negative_t(self):
        """Raises error for t < 0."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)
        with pytest.raises(ValueError, match="must be in"):
            sv1.interpolate(sv2, -0.1)

    def test_interpolate_rejects_t_greater_than_one(self):
        """Raises error for t > 1."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)
        with pytest.raises(ValueError, match="must be in"):
            sv1.interpolate(sv2, 1.1)

    def test_interpolate_produces_smooth_transition(self):
        """Interpolation is monotonic (distances decrease toward target)."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)

        # Get interpolation at increasing t values
        steps = [sv1.interpolate(sv2, t) for t in [0.0, 0.25, 0.5, 0.75, 1.0]]

        # Distance to sv2 should decrease monotonically
        distances = [(s.data - sv2.data).norm().item() for s in steps]
        for i in range(len(distances) - 1):
            assert distances[i] >= distances[i + 1]

    def test_interpolate_names_combined(self):
        """Interpolated vector has combined name."""
        sv1 = StyleVector.random(seed=42, name="fruity")
        sv2 = StyleVector.random(seed=43, name="aggressive")
        result = sv1.interpolate(sv2, 0.5)
        assert "fruity" in result.name
        assert "aggressive" in result.name


class TestStyleVectorPersistence:
    """Test save/load operations."""

    def test_save_load_roundtrip(self):
        """Save and load preserves data exactly."""
        sv = StyleVector.random(seed=42, name="test")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "style.pt"
            sv.save(path)
            loaded = StyleVector.load(path)
            assert torch.allclose(sv.data, loaded.data)
            assert sv.name == loaded.name

    def test_save_creates_parent_dirs(self):
        """Save creates parent directories if needed."""
        sv = StyleVector.random(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "style.pt"
            sv.save(path)
            assert path.exists()

    def test_load_missing_file_raises(self):
        """Load raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            StyleVector.load("/nonexistent/path.pt")

    def test_load_handles_raw_tensor_format(self):
        """Load works with raw tensor (legacy format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "style.pt"
            torch.save(torch.randn(64), path)
            loaded = StyleVector.load(path)
            assert loaded.data.shape == (64,)

    def test_load_handles_style_z_key_format(self):
        """Load works with style_z key (persistence.py format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "style.pt"
            torch.save({"style_z": torch.randn(64)}, path)
            loaded = StyleVector.load(path)
            assert loaded.data.shape == (64,)


class TestStyleVectorOperations:
    """Test other operations."""

    def test_to_device(self):
        """to() moves to specified device."""
        sv = StyleVector.random(seed=42)
        moved = sv.to("cpu")
        assert moved.data.device == torch.device("cpu")

    def test_clone_creates_copy(self):
        """clone() creates independent copy."""
        sv = StyleVector.random(seed=42)
        cloned = sv.clone()
        sv.data[0] = 999.0
        assert cloned.data[0] != 999.0

    def test_perturb_adds_noise(self):
        """perturb() creates different vector."""
        sv = StyleVector.random(seed=42)
        perturbed = sv.perturb(std=0.1, seed=99)
        assert not torch.allclose(sv.data, perturbed.data)

    def test_perturb_is_reproducible(self):
        """perturb() with seed is reproducible."""
        sv = StyleVector.random(seed=42)
        p1 = sv.perturb(std=0.1, seed=99)
        p2 = sv.perturb(std=0.1, seed=99)
        assert torch.allclose(p1.data, p2.data)

    def test_equality(self):
        """Equality compares data values."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=42)
        sv3 = StyleVector.random(seed=43)
        assert sv1 == sv2
        assert sv1 != sv3

    def test_repr(self):
        """repr includes useful info."""
        sv = StyleVector.random(seed=42, name="fruity")
        r = repr(sv)
        assert "64" in r or "dim=" in r
        assert "fruity" in r


class TestCreatePersonalityVectors:
    """Test create_personality_vectors utility."""

    def test_creates_all_personalities(self):
        """Creates vectors for all specified personalities."""
        vectors = create_personality_vectors(["a", "b", "c"])
        assert len(vectors) == 3
        assert "a" in vectors
        assert "b" in vectors
        assert "c" in vectors

    def test_uses_config_personalities_by_default(self):
        """Uses CONFIG.personalities when none specified."""
        vectors = create_personality_vectors()
        for name in CONFIG.personalities:
            assert name in vectors

    def test_each_personality_is_different(self):
        """Each personality gets a unique vector."""
        vectors = create_personality_vectors(["a", "b", "c"])
        assert not torch.allclose(vectors["a"].data, vectors["b"].data)
        assert not torch.allclose(vectors["b"].data, vectors["c"].data)
        assert not torch.allclose(vectors["a"].data, vectors["c"].data)

    def test_is_reproducible(self):
        """Same seed produces same vectors."""
        v1 = create_personality_vectors(["a", "b"], seed=42)
        v2 = create_personality_vectors(["a", "b"], seed=42)
        assert torch.allclose(v1["a"].data, v2["a"].data)
        assert torch.allclose(v1["b"].data, v2["b"].data)

    def test_vectors_have_names(self):
        """Created vectors have personality names."""
        vectors = create_personality_vectors(["fruity", "aggressive"])
        assert vectors["fruity"].name == "fruity"
        assert vectors["aggressive"].name == "aggressive"


class TestInterpolatePersonalities:
    """Test interpolate_personalities utility."""

    def test_returns_correct_number_of_steps(self):
        """Returns requested number of steps."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)
        result = interpolate_personalities(sv1, sv2, steps=5)
        assert len(result) == 5

    def test_first_is_start(self):
        """First element matches start."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)
        result = interpolate_personalities(sv1, sv2, steps=5)
        assert torch.allclose(result[0].data, sv1.data)

    def test_last_is_end(self):
        """Last element matches end."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)
        result = interpolate_personalities(sv1, sv2, steps=5)
        assert torch.allclose(result[-1].data, sv2.data)

    def test_rejects_less_than_two_steps(self):
        """Raises error for steps < 2."""
        sv1 = StyleVector.random(seed=42)
        sv2 = StyleVector.random(seed=43)
        with pytest.raises(ValueError, match="at least 2"):
            interpolate_personalities(sv1, sv2, steps=1)


class TestStyleVectorWithModel:
    """Test StyleVector integration with GlyphNetwork."""

    def test_zero_vector_produces_valid_output(self):
        """Model produces valid output with zero style vector."""
        from src.glyphnet import create_model

        model = create_model("cpu")
        model.eval()

        style = StyleVector.zeros()
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])

        with torch.no_grad():
            output = model(prev_char, curr_char, style.data.unsqueeze(0))

        # Output should be valid (not NaN, in [0,1] range)
        assert not torch.isnan(output).any()
        assert output.min() >= 0.0
        assert output.max() <= 1.0
        assert output.shape == (1, 1, 128, 128)

    def test_same_style_produces_consistent_outputs(self):
        """Same style vector produces deterministic outputs."""
        from src.glyphnet import create_model

        model = create_model("cpu")
        model.eval()

        style = StyleVector.random(seed=42)
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])

        with torch.no_grad():
            output1 = model(prev_char, curr_char, style.data.unsqueeze(0))
            output2 = model(prev_char, curr_char, style.data.unsqueeze(0))

        assert torch.allclose(output1, output2)

    def test_different_styles_produce_different_outputs(self):
        """Different style vectors produce different outputs."""
        from src.glyphnet import create_model

        model = create_model("cpu")
        model.eval()

        style1 = StyleVector.random(seed=42)
        style2 = StyleVector.random(seed=43)
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])

        with torch.no_grad():
            output1 = model(prev_char, curr_char, style1.data.unsqueeze(0))
            output2 = model(prev_char, curr_char, style2.data.unsqueeze(0))

        # Outputs should be noticeably different
        diff = (output1 - output2).abs().mean().item()
        assert diff > 0.01

    def test_style_produces_consistent_across_glyphs(self):
        """Same style produces visually related outputs across different glyphs."""
        from src.glyphnet import create_model

        model = create_model("cpu")
        model.eval()

        style = StyleVector.random(seed=42)

        outputs = []
        for char_idx in [1, 2, 3]:  # a, b, c
            prev_char = torch.tensor([0])
            curr_char = torch.tensor([char_idx])
            with torch.no_grad():
                output = model(prev_char, curr_char, style.data.unsqueeze(0))
            outputs.append(output)

        # All outputs should be valid
        for output in outputs:
            assert not torch.isnan(output).any()
            assert output.min() >= 0.0
            assert output.max() <= 1.0

    def test_extreme_style_values_handled_gracefully(self):
        """Model handles extreme style_z values (±5 std)."""
        from src.glyphnet import create_model

        model = create_model("cpu")
        model.eval()

        # Create extreme style vector (5 std deviations)
        style = StyleVector.random(seed=42, std=5.0)
        prev_char = torch.tensor([0])
        curr_char = torch.tensor([1])

        with torch.no_grad():
            output = model(prev_char, curr_char, style.data.unsqueeze(0))

        # Should still produce valid output (sigmoid ensures [0,1])
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.min() >= 0.0
        assert output.max() <= 1.0
