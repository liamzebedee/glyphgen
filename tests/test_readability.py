"""
Tests for readability validation.

Tests cover:
- Edge sharpness metric computation
- Glyph metrics (x-height, stroke width, etc.)
- Blank and saturated glyph detection
- Reference deviation calculation
- Single and batch validation
- Summary statistics
"""

import pytest
import torch

from src.readability import (
    BLANK_THRESHOLD,
    EDGE_SHARPNESS_THRESHOLD,
    METRIC_VARIANCE_THRESHOLD,
    OPEN_SANS_REFERENCE,
    SATURATED_THRESHOLD,
    BatchReadabilityResult,
    GlyphMetrics,
    ReadabilityResult,
    check_is_blank,
    check_is_saturated,
    compute_edge_sharpness,
    compute_edge_sharpness_batch,
    compute_glyph_metrics,
    compute_gradient_magnitude,
    compute_reference_deviation,
    get_readability_summary,
    validate_batch_readability,
    validate_glyph_readability,
)


class TestGradientMagnitude:
    """Tests for gradient magnitude computation."""

    def test_gradient_shape_2d(self):
        """Test gradient output shape matches input for 2D tensor."""
        image = torch.rand(128, 128)
        grad = compute_gradient_magnitude(image)
        assert grad.shape == image.shape

    def test_gradient_shape_3d(self):
        """Test gradient output shape matches input for 3D tensor."""
        image = torch.rand(1, 128, 128)
        grad = compute_gradient_magnitude(image)
        assert grad.shape == image.shape

    def test_gradient_shape_4d(self):
        """Test gradient output shape matches input for 4D tensor."""
        image = torch.rand(4, 1, 128, 128)
        grad = compute_gradient_magnitude(image)
        assert grad.shape == image.shape

    def test_gradient_non_negative(self):
        """Test gradient magnitude is always non-negative."""
        image = torch.rand(4, 1, 128, 128)
        grad = compute_gradient_magnitude(image)
        assert (grad >= 0).all()

    def test_gradient_zero_for_constant(self):
        """Test gradient is near zero for constant image interior."""
        image = torch.ones(1, 128, 128) * 0.5
        grad = compute_gradient_magnitude(image)
        # Interior should be near zero (exclude border due to padding effects)
        interior = grad[:, 2:-2, 2:-2]
        assert interior.mean().item() < 0.01

    def test_gradient_detects_edges(self):
        """Test gradient is higher at edges."""
        # Create image with a sharp vertical edge
        image = torch.zeros(1, 128, 128)
        image[:, :, 64:] = 1.0  # right half is white

        grad = compute_gradient_magnitude(image)

        # Gradient should be highest at the edge column (63-65)
        edge_region = grad[:, :, 62:66]
        non_edge_region = grad[:, :, :60]
        assert edge_region.mean() > non_edge_region.mean() * 2


class TestEdgeSharpness:
    """Tests for edge sharpness metric."""

    def test_edge_sharpness_returns_float(self):
        """Test edge sharpness returns a float."""
        image = torch.rand(1, 128, 128)
        sharpness = compute_edge_sharpness(image)
        assert isinstance(sharpness, float)

    def test_edge_sharpness_range(self):
        """Test edge sharpness is in [0, 1] range."""
        # Test various images
        for _ in range(5):
            image = torch.rand(1, 128, 128)
            sharpness = compute_edge_sharpness(image)
            assert 0 <= sharpness <= 1

    def test_edge_sharpness_low_for_smooth(self):
        """Test smooth images have low edge sharpness."""
        image = torch.ones(128, 128) * 0.5
        sharpness = compute_edge_sharpness(image)
        assert sharpness < 0.1

    def test_edge_sharpness_high_for_edges(self):
        """Test images with sharp edges have high sharpness."""
        image = torch.zeros(1, 128, 128)
        # Create a checkerboard pattern for lots of edges
        for i in range(0, 128, 8):
            for j in range(0, 128, 8):
                if (i // 8 + j // 8) % 2 == 0:
                    image[:, i:i+8, j:j+8] = 1.0

        sharpness = compute_edge_sharpness(image)
        assert sharpness > 0.3  # Should have significant edge content

    def test_edge_sharpness_batch(self):
        """Test batch edge sharpness computation."""
        images = torch.rand(4, 1, 128, 128)
        sharpness_list = compute_edge_sharpness_batch(images)

        assert len(sharpness_list) == 4
        for s in sharpness_list:
            assert isinstance(s, float)
            assert 0 <= s <= 1

    def test_edge_sharpness_batch_invalid_dim(self):
        """Test batch computation rejects invalid dimensions."""
        images = torch.rand(128, 128)  # 2D instead of 4D
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            compute_edge_sharpness_batch(images)


class TestGlyphMetrics:
    """Tests for glyph metric computation."""

    def test_metrics_returns_dataclass(self):
        """Test compute_glyph_metrics returns GlyphMetrics."""
        image = torch.rand(1, 128, 128)
        metrics = compute_glyph_metrics(image)
        assert isinstance(metrics, GlyphMetrics)

    def test_metrics_for_blank_image(self):
        """Test metrics for blank (all black) image."""
        image = torch.zeros(1, 128, 128)
        metrics = compute_glyph_metrics(image)

        assert metrics.ink_coverage == pytest.approx(0.0, abs=1e-6)
        assert metrics.x_height == 0.0
        assert metrics.stroke_width == 0.0

    def test_metrics_ink_coverage(self):
        """Test ink coverage calculation."""
        # Create image with 25% ink coverage
        image = torch.zeros(1, 128, 128)
        image[:, :64, :64] = 1.0  # top-left quarter is white

        metrics = compute_glyph_metrics(image)
        assert metrics.ink_coverage == pytest.approx(0.25, abs=0.01)

    def test_metrics_cap_height(self):
        """Test cap height measures top of glyph."""
        # Create a glyph that spans from row 20 to row 100
        image = torch.zeros(128, 128)
        image[20:100, 60:68] = 1.0  # vertical bar

        metrics = compute_glyph_metrics(image)
        # Cap height should be ~0.84 (1 - 20/128)
        assert metrics.cap_height == pytest.approx(0.84, abs=0.05)

    def test_metrics_stroke_width_positive(self):
        """Test stroke width is positive for non-blank image."""
        image = torch.zeros(128, 128)
        image[30:80, 60:70] = 1.0  # 10-pixel wide vertical bar

        metrics = compute_glyph_metrics(image)
        assert metrics.stroke_width > 0

    def test_metrics_serialization(self):
        """Test GlyphMetrics can be serialized and deserialized."""
        original = GlyphMetrics(
            x_height=0.45,
            cap_height=0.70,
            stroke_width=8.0,
            descender_depth=0.15,
            ink_coverage=0.12,
        )

        data = original.to_dict()
        restored = GlyphMetrics.from_dict(data)

        assert restored.x_height == original.x_height
        assert restored.cap_height == original.cap_height
        assert restored.stroke_width == original.stroke_width
        assert restored.descender_depth == original.descender_depth
        assert restored.ink_coverage == original.ink_coverage


class TestBlankSaturatedDetection:
    """Tests for blank and saturated glyph detection."""

    def test_check_is_blank_true_for_zeros(self):
        """Test blank detection for zero image."""
        image = torch.zeros(1, 128, 128)
        assert check_is_blank(image) is True

    def test_check_is_blank_false_for_content(self):
        """Test blank detection returns false for image with content."""
        image = torch.zeros(1, 128, 128)
        image[:, 64, 64] = 0.5
        assert check_is_blank(image) is False

    def test_check_is_blank_threshold(self):
        """Test blank detection respects threshold."""
        image = torch.ones(1, 128, 128) * (BLANK_THRESHOLD - 0.001)
        assert check_is_blank(image) is True

        image = torch.ones(1, 128, 128) * (BLANK_THRESHOLD + 0.001)
        assert check_is_blank(image) is False

    def test_check_is_saturated_true_for_ones(self):
        """Test saturated detection for all-white image."""
        image = torch.ones(1, 128, 128)
        assert check_is_saturated(image) is True

    def test_check_is_saturated_false_for_content(self):
        """Test saturated detection returns false for image with variation."""
        image = torch.ones(1, 128, 128)
        image[:, 64, 64] = 0.5
        assert check_is_saturated(image) is False

    def test_check_is_saturated_threshold(self):
        """Test saturated detection respects threshold."""
        image = torch.ones(1, 128, 128) * SATURATED_THRESHOLD
        # Min is at threshold, so should be saturated
        assert check_is_saturated(image) is True

        image = torch.ones(1, 128, 128) * (SATURATED_THRESHOLD - 0.01)
        assert check_is_saturated(image) is False


class TestReferenceDeviation:
    """Tests for reference deviation computation."""

    def test_deviation_zero_for_matching_metrics(self):
        """Test deviation is zero when metrics match reference."""
        deviation = compute_reference_deviation(OPEN_SANS_REFERENCE)
        for key, value in deviation.items():
            assert value == pytest.approx(0.0, abs=1e-6)

    def test_deviation_positive_for_different_metrics(self):
        """Test deviation is positive for different metrics."""
        different = GlyphMetrics(
            x_height=0.30,  # Different from 0.45
            cap_height=0.50,  # Different from 0.70
            stroke_width=4.0,  # Different from 8.0
            descender_depth=0.05,  # Different from 0.15
            ink_coverage=0.06,  # Different from 0.12
        )

        deviation = compute_reference_deviation(different)

        assert deviation["x_height"] > 0
        assert deviation["cap_height"] > 0
        assert deviation["stroke_width"] > 0

    def test_deviation_calculation_accuracy(self):
        """Test deviation is calculated correctly."""
        # X-height: reference is 0.45, test with 0.54 (20% higher)
        metrics = GlyphMetrics(
            x_height=0.54,
            cap_height=0.70,
            stroke_width=8.0,
            descender_depth=0.15,
            ink_coverage=0.12,
        )

        deviation = compute_reference_deviation(metrics)
        # Deviation should be 0.09/0.45 = 0.2 (20%)
        assert deviation["x_height"] == pytest.approx(0.2, abs=0.01)

    def test_deviation_custom_reference(self):
        """Test deviation with custom reference metrics."""
        custom_ref = GlyphMetrics(
            x_height=0.50,
            cap_height=0.80,
            stroke_width=10.0,
            descender_depth=0.20,
            ink_coverage=0.15,
        )

        measured = GlyphMetrics(
            x_height=0.55,  # 10% higher
            cap_height=0.80,  # Same
            stroke_width=10.0,  # Same
            descender_depth=0.20,  # Same
            ink_coverage=0.15,  # Same
        )

        deviation = compute_reference_deviation(measured, reference=custom_ref)
        assert deviation["x_height"] == pytest.approx(0.1, abs=0.01)
        assert deviation["cap_height"] == pytest.approx(0.0, abs=0.01)


class TestReadabilityResult:
    """Tests for ReadabilityResult dataclass."""

    def test_is_readable_all_pass(self):
        """Test is_readable is True when all checks pass."""
        result = ReadabilityResult(
            edge_sharpness=0.8,
            metrics=OPEN_SANS_REFERENCE,
            is_blank=False,
            is_saturated=False,
            passes_edge_threshold=True,
            passes_metrics_threshold=True,
        )
        assert result.is_readable is True

    def test_is_readable_fails_if_blank(self):
        """Test is_readable is False if glyph is blank."""
        result = ReadabilityResult(
            edge_sharpness=0.8,
            metrics=OPEN_SANS_REFERENCE,
            is_blank=True,  # Fail condition
            is_saturated=False,
            passes_edge_threshold=True,
            passes_metrics_threshold=True,
        )
        assert result.is_readable is False

    def test_is_readable_fails_if_saturated(self):
        """Test is_readable is False if glyph is saturated."""
        result = ReadabilityResult(
            edge_sharpness=0.8,
            metrics=OPEN_SANS_REFERENCE,
            is_blank=False,
            is_saturated=True,  # Fail condition
            passes_edge_threshold=True,
            passes_metrics_threshold=True,
        )
        assert result.is_readable is False

    def test_is_readable_fails_if_edge_fails(self):
        """Test is_readable is False if edge threshold not met."""
        result = ReadabilityResult(
            edge_sharpness=0.5,
            metrics=OPEN_SANS_REFERENCE,
            is_blank=False,
            is_saturated=False,
            passes_edge_threshold=False,  # Fail condition
            passes_metrics_threshold=True,
        )
        assert result.is_readable is False

    def test_is_readable_fails_if_metrics_fail(self):
        """Test is_readable is False if metrics threshold not met."""
        result = ReadabilityResult(
            edge_sharpness=0.8,
            metrics=OPEN_SANS_REFERENCE,
            is_blank=False,
            is_saturated=False,
            passes_edge_threshold=True,
            passes_metrics_threshold=False,  # Fail condition
        )
        assert result.is_readable is False

    def test_result_serialization(self):
        """Test ReadabilityResult can be serialized to dict."""
        result = ReadabilityResult(
            edge_sharpness=0.8,
            metrics=OPEN_SANS_REFERENCE,
            is_blank=False,
            is_saturated=False,
            passes_edge_threshold=True,
            passes_metrics_threshold=True,
            reference_deviation={"x_height": 0.05},
        )

        data = result.to_dict()
        assert data["edge_sharpness"] == 0.8
        assert data["is_readable"] is True
        assert "metrics" in data
        assert data["reference_deviation"]["x_height"] == 0.05


class TestValidateGlyphReadability:
    """Tests for single glyph validation."""

    def test_validate_returns_result(self):
        """Test validate_glyph_readability returns ReadabilityResult."""
        image = torch.rand(1, 128, 128)
        result = validate_glyph_readability(image)
        assert isinstance(result, ReadabilityResult)

    def test_validate_blank_glyph(self):
        """Test validation correctly identifies blank glyph."""
        image = torch.zeros(1, 128, 128)
        result = validate_glyph_readability(image)

        assert result.is_blank is True
        assert result.is_readable is False

    def test_validate_saturated_glyph(self):
        """Test validation correctly identifies saturated glyph."""
        image = torch.ones(1, 128, 128)
        result = validate_glyph_readability(image)

        assert result.is_saturated is True
        assert result.is_readable is False

    def test_validate_custom_thresholds(self):
        """Test validation with custom thresholds."""
        image = torch.rand(1, 128, 128)

        # Very strict edge threshold
        result_strict = validate_glyph_readability(image, edge_threshold=0.99)
        # Very lenient edge threshold
        result_lenient = validate_glyph_readability(image, edge_threshold=0.01)

        # Lenient should be more likely to pass
        assert result_lenient.passes_edge_threshold

    def test_validate_computes_deviation(self):
        """Test validation computes reference deviation."""
        image = torch.rand(1, 128, 128)
        result = validate_glyph_readability(image)

        assert result.reference_deviation is not None
        assert "x_height" in result.reference_deviation

    def test_validate_2d_input(self):
        """Test validation handles 2D input."""
        image = torch.rand(128, 128)
        result = validate_glyph_readability(image)
        assert isinstance(result, ReadabilityResult)


class TestBatchValidation:
    """Tests for batch glyph validation."""

    def test_batch_validate_returns_result(self):
        """Test batch validation returns BatchReadabilityResult."""
        images = torch.rand(4, 1, 128, 128)
        result = validate_batch_readability(images)
        assert isinstance(result, BatchReadabilityResult)

    def test_batch_validate_counts_correctly(self):
        """Test batch validation counts total correctly."""
        images = torch.rand(8, 1, 128, 128)
        result = validate_batch_readability(images)

        assert result.total_count == 8
        assert len(result.results) == 8

    def test_batch_validate_readable_ratio(self):
        """Test batch readable_ratio is in valid range."""
        images = torch.rand(4, 1, 128, 128)
        result = validate_batch_readability(images)

        assert 0 <= result.readable_ratio <= 1

    def test_batch_validate_mean_sharpness(self):
        """Test batch computes mean sharpness correctly."""
        images = torch.rand(4, 1, 128, 128)
        result = validate_batch_readability(images)

        # Verify mean is computed correctly
        manual_mean = sum(r.edge_sharpness for r in result.results) / len(result.results)
        assert result.mean_edge_sharpness == pytest.approx(manual_mean, abs=1e-6)

    def test_batch_validate_invalid_dim(self):
        """Test batch validation rejects invalid dimensions."""
        images = torch.rand(128, 128)  # 2D
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            validate_batch_readability(images)

    def test_batch_serialization(self):
        """Test BatchReadabilityResult can be serialized."""
        images = torch.rand(2, 1, 128, 128)
        result = validate_batch_readability(images)

        data = result.to_dict()
        assert "results" in data
        assert len(data["results"]) == 2
        assert "readable_ratio" in data


class TestReadabilitySummary:
    """Tests for readability summary statistics."""

    def test_summary_empty_list(self):
        """Test summary handles empty list."""
        summary = get_readability_summary([])

        assert summary["mean_edge_sharpness"] == 0.0
        assert summary["readable_ratio"] == 0.0
        assert summary["blank_count"] == 0

    def test_summary_computes_stats(self):
        """Test summary computes correct statistics."""
        results = [
            ReadabilityResult(
                edge_sharpness=0.6,
                metrics=OPEN_SANS_REFERENCE,
                is_blank=False,
                is_saturated=False,
                passes_edge_threshold=False,
                passes_metrics_threshold=True,
            ),
            ReadabilityResult(
                edge_sharpness=0.8,
                metrics=OPEN_SANS_REFERENCE,
                is_blank=False,
                is_saturated=False,
                passes_edge_threshold=True,
                passes_metrics_threshold=True,
            ),
            ReadabilityResult(
                edge_sharpness=0.7,
                metrics=OPEN_SANS_REFERENCE,
                is_blank=True,
                is_saturated=False,
                passes_edge_threshold=True,
                passes_metrics_threshold=True,
            ),
        ]

        summary = get_readability_summary(results)

        assert summary["mean_edge_sharpness"] == pytest.approx(0.7, abs=0.01)
        assert summary["min_edge_sharpness"] == 0.6
        assert summary["max_edge_sharpness"] == 0.8
        assert summary["blank_count"] == 1
        assert summary["readable_ratio"] == pytest.approx(1/3, abs=0.01)  # Only 1 readable

    def test_summary_counts_saturated(self):
        """Test summary counts saturated glyphs."""
        results = [
            ReadabilityResult(
                edge_sharpness=0.8,
                metrics=OPEN_SANS_REFERENCE,
                is_blank=False,
                is_saturated=True,
                passes_edge_threshold=True,
                passes_metrics_threshold=True,
            ),
            ReadabilityResult(
                edge_sharpness=0.8,
                metrics=OPEN_SANS_REFERENCE,
                is_blank=False,
                is_saturated=True,
                passes_edge_threshold=True,
                passes_metrics_threshold=True,
            ),
        ]

        summary = get_readability_summary(results)
        assert summary["saturated_count"] == 2


class TestThresholdConstants:
    """Tests for threshold constant values."""

    def test_edge_sharpness_threshold_value(self):
        """Test edge sharpness threshold matches spec."""
        # From specs/readability-guarantees.md: Mean gradient > 0.7
        assert EDGE_SHARPNESS_THRESHOLD == 0.7

    def test_metric_variance_threshold_value(self):
        """Test metric variance threshold matches spec."""
        # From specs/readability-guarantees.md: Within ±15% of reference
        assert METRIC_VARIANCE_THRESHOLD == 0.15

    def test_blank_threshold_reasonable(self):
        """Test blank threshold is reasonable."""
        assert 0 < BLANK_THRESHOLD < 0.1

    def test_saturated_threshold_reasonable(self):
        """Test saturated threshold is reasonable."""
        assert 0.9 < SATURATED_THRESHOLD < 1.0


class TestOpenSansReference:
    """Tests for Open Sans reference metrics."""

    def test_reference_values_reasonable(self):
        """Test Open Sans reference values are reasonable."""
        ref = OPEN_SANS_REFERENCE

        # X-height should be less than cap-height
        assert ref.x_height < ref.cap_height

        # All values should be positive
        assert ref.x_height > 0
        assert ref.cap_height > 0
        assert ref.stroke_width > 0
        assert ref.ink_coverage > 0

        # Ink coverage should be reasonable for a font (not too high)
        assert ref.ink_coverage < 0.5
