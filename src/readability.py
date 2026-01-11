"""
Readability validation for generated glyphs.

Implements metrics and validation for ensuring generated glyphs remain legible:
- Edge sharpness via gradient magnitude
- Glyph metrics (x-height, stroke width) comparison to Open Sans reference
- Blank/saturated glyph detection

Usage:
    from src.readability import (
        compute_edge_sharpness,
        compute_glyph_metrics,
        validate_glyph_readability,
    )

    # Single glyph validation
    sharpness = compute_edge_sharpness(glyph_tensor)
    metrics = compute_glyph_metrics(glyph_tensor)
    result = validate_glyph_readability(glyph_tensor)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# Thresholds from specs/readability-guarantees.md
EDGE_SHARPNESS_THRESHOLD = 0.7  # Mean gradient > 0.7
METRIC_VARIANCE_THRESHOLD = 0.15  # Within ±15% of reference
BLANK_THRESHOLD = 0.01  # Max value < threshold means blank
SATURATED_THRESHOLD = 0.99  # Min value > threshold means saturated


@dataclass
class GlyphMetrics:
    """Measured metrics for a single glyph."""

    x_height: float  # Vertical extent of main glyph body (0-1 normalized)
    cap_height: float  # Top of glyph bounding box (0-1 normalized)
    stroke_width: float  # Average stroke width in pixels
    descender_depth: float  # How far below baseline (0-1 normalized)
    ink_coverage: float  # Percentage of pixels with ink (0-1)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "x_height": self.x_height,
            "cap_height": self.cap_height,
            "stroke_width": self.stroke_width,
            "descender_depth": self.descender_depth,
            "ink_coverage": self.ink_coverage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "GlyphMetrics":
        """Create from dictionary."""
        return cls(
            x_height=data["x_height"],
            cap_height=data["cap_height"],
            stroke_width=data["stroke_width"],
            descender_depth=data["descender_depth"],
            ink_coverage=data["ink_coverage"],
        )


@dataclass
class ReadabilityResult:
    """Result of readability validation for a single glyph."""

    edge_sharpness: float
    metrics: GlyphMetrics
    is_blank: bool
    is_saturated: bool
    passes_edge_threshold: bool
    passes_metrics_threshold: bool
    reference_deviation: Optional[Dict[str, float]] = None  # Deviation from reference

    @property
    def is_readable(self) -> bool:
        """Check if glyph passes all readability checks."""
        return (
            not self.is_blank
            and not self.is_saturated
            and self.passes_edge_threshold
            and self.passes_metrics_threshold
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "edge_sharpness": self.edge_sharpness,
            "metrics": self.metrics.to_dict(),
            "is_blank": self.is_blank,
            "is_saturated": self.is_saturated,
            "passes_edge_threshold": self.passes_edge_threshold,
            "passes_metrics_threshold": self.passes_metrics_threshold,
            "reference_deviation": self.reference_deviation,
            "is_readable": self.is_readable,
        }


@dataclass
class BatchReadabilityResult:
    """Aggregated readability results for a batch of glyphs."""

    results: List[ReadabilityResult] = field(default_factory=list)
    mean_edge_sharpness: float = 0.0
    readable_count: int = 0
    total_count: int = 0

    @property
    def readable_ratio(self) -> float:
        """Ratio of readable glyphs (0-1)."""
        if self.total_count == 0:
            return 0.0
        return self.readable_count / self.total_count

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "mean_edge_sharpness": self.mean_edge_sharpness,
            "readable_count": self.readable_count,
            "total_count": self.total_count,
            "readable_ratio": self.readable_ratio,
        }


# Reference metrics for Open Sans (approximate values for 128x128 rendered glyphs)
# These are normalized to 0-1 range relative to image height
OPEN_SANS_REFERENCE = GlyphMetrics(
    x_height=0.45,  # Lowercase letters occupy ~45% of em-square height
    cap_height=0.70,  # Capitals occupy ~70% of em-square height
    stroke_width=8.0,  # Average stroke width in pixels at 128x128
    descender_depth=0.15,  # Descenders extend ~15% below baseline
    ink_coverage=0.12,  # ~12% of pixels have ink on average
)


def _create_sobel_kernels(
    dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create Sobel kernels for edge detection."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)

    return sobel_x, sobel_y


def compute_gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient magnitude using Sobel filters.

    Args:
        image: Glyph image tensor (B, 1, H, W) or (1, H, W) or (H, W)

    Returns:
        Gradient magnitude tensor, same spatial shape as input
    """
    # Normalize input shape to (B, 1, H, W)
    original_shape = image.shape
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    sobel_x, sobel_y = _create_sobel_kernels(image.dtype, image.device)

    # Compute gradients
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)

    # Gradient magnitude with epsilon for numerical stability
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    # Restore original batch dimension if needed
    if len(original_shape) == 2:
        grad_mag = grad_mag.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        grad_mag = grad_mag.squeeze(0)

    return grad_mag


def compute_edge_sharpness(image: torch.Tensor) -> float:
    """
    Compute edge sharpness score for a glyph.

    The sharpness score is the mean gradient magnitude normalized to [0, 1].
    Higher values indicate sharper, more defined edges.

    Args:
        image: Glyph image tensor (1, H, W) or (H, W), values in [0, 1]

    Returns:
        Edge sharpness score (0-1), higher is sharper
    """
    grad_mag = compute_gradient_magnitude(image)

    # Normalize: Sobel max theoretical output is 4.0 for a step edge
    # In practice, we normalize by dividing by a reasonable max (2.0)
    normalized = grad_mag / 2.0
    normalized = torch.clamp(normalized, 0, 1)

    # Mean gradient magnitude as sharpness score
    return normalized.mean().item()


def compute_edge_sharpness_batch(images: torch.Tensor) -> List[float]:
    """
    Compute edge sharpness for a batch of glyphs.

    Args:
        images: Batch of glyph images (B, 1, H, W)

    Returns:
        List of edge sharpness scores
    """
    if images.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B, 1, H, W), got {images.dim()}D")

    grad_mag = compute_gradient_magnitude(images)
    normalized = torch.clamp(grad_mag / 2.0, 0, 1)

    # Mean per image
    scores = normalized.mean(dim=[1, 2, 3])
    return scores.tolist()


def compute_glyph_metrics(image: torch.Tensor, ink_threshold: float = 0.5) -> GlyphMetrics:
    """
    Compute structural metrics for a glyph.

    Args:
        image: Glyph image tensor (1, H, W) or (H, W), values in [0, 1]
        ink_threshold: Threshold above which a pixel is considered "ink"

    Returns:
        GlyphMetrics with measured values
    """
    # Normalize shape
    if image.dim() == 3:
        image = image.squeeze(0)

    h, w = image.shape

    # Binary mask of ink pixels
    ink_mask = image > ink_threshold

    # Ink coverage
    ink_coverage = ink_mask.float().mean().item()

    # If blank, return zeros
    if ink_coverage < 1e-6:
        return GlyphMetrics(
            x_height=0.0,
            cap_height=0.0,
            stroke_width=0.0,
            descender_depth=0.0,
            ink_coverage=0.0,
        )

    # Find bounding box of ink
    ink_rows = ink_mask.any(dim=1)

    if not ink_rows.any():
        # No ink found
        return GlyphMetrics(
            x_height=0.0,
            cap_height=0.0,
            stroke_width=0.0,
            descender_depth=0.0,
            ink_coverage=ink_coverage,
        )

    # Find first and last row with ink
    row_indices = torch.where(ink_rows)[0]
    top_row = row_indices[0].item()
    bottom_row = row_indices[-1].item()

    # Cap height: distance from top of image to top of glyph
    cap_height = 1.0 - (top_row / h)

    # Assume baseline is at ~75% of image height (standard typography)
    baseline_row = int(h * 0.75)

    # X-height: main body above baseline
    main_body_rows = (row_indices >= int(h * 0.25)) & (row_indices <= baseline_row)
    if main_body_rows.any():
        main_top = row_indices[main_body_rows][0].item()
        main_bottom = min(row_indices[main_body_rows][-1].item(), baseline_row)
        x_height = (main_bottom - main_top) / h
    else:
        x_height = (baseline_row - top_row) / h

    # Descender depth: how far below baseline
    if bottom_row > baseline_row:
        descender_depth = (bottom_row - baseline_row) / h
    else:
        descender_depth = 0.0

    # Stroke width: estimate from vertical runs
    stroke_widths = []
    for col in range(w):
        col_ink = ink_mask[:, col]
        if col_ink.any():
            # Find runs of ink
            runs = torch.where(col_ink)[0]
            if len(runs) > 0:
                # Count consecutive sequences
                diffs = runs[1:] - runs[:-1]
                run_lengths = []
                current_run = 1
                for d in diffs:
                    if d == 1:
                        current_run += 1
                    else:
                        run_lengths.append(current_run)
                        current_run = 1
                run_lengths.append(current_run)
                stroke_widths.extend(run_lengths)

    if stroke_widths:
        stroke_width = sum(stroke_widths) / len(stroke_widths)
    else:
        stroke_width = 0.0

    return GlyphMetrics(
        x_height=x_height,
        cap_height=cap_height,
        stroke_width=stroke_width,
        descender_depth=descender_depth,
        ink_coverage=ink_coverage,
    )


def check_is_blank(image: torch.Tensor) -> bool:
    """Check if glyph is blank (all pixels near zero)."""
    return image.max().item() < BLANK_THRESHOLD


def check_is_saturated(image: torch.Tensor) -> bool:
    """Check if glyph is saturated (all pixels near one)."""
    return image.min().item() > SATURATED_THRESHOLD


def compute_reference_deviation(
    metrics: GlyphMetrics,
    reference: Optional[GlyphMetrics] = None,
) -> Dict[str, float]:
    """
    Compute deviation from reference metrics.

    Args:
        metrics: Measured glyph metrics
        reference: Reference metrics (defaults to Open Sans)

    Returns:
        Dictionary of percentage deviations from reference
    """
    if reference is None:
        reference = OPEN_SANS_REFERENCE

    deviations = {}

    # X-height deviation
    if reference.x_height > 0:
        deviations["x_height"] = abs(metrics.x_height - reference.x_height) / reference.x_height
    else:
        deviations["x_height"] = 0.0

    # Cap height deviation
    if reference.cap_height > 0:
        deviations["cap_height"] = (
            abs(metrics.cap_height - reference.cap_height) / reference.cap_height
        )
    else:
        deviations["cap_height"] = 0.0

    # Stroke width deviation
    if reference.stroke_width > 0:
        deviations["stroke_width"] = (
            abs(metrics.stroke_width - reference.stroke_width) / reference.stroke_width
        )
    else:
        deviations["stroke_width"] = 0.0

    # Ink coverage deviation
    if reference.ink_coverage > 0:
        deviations["ink_coverage"] = (
            abs(metrics.ink_coverage - reference.ink_coverage) / reference.ink_coverage
        )
    else:
        deviations["ink_coverage"] = 0.0

    return deviations


def validate_glyph_readability(
    image: torch.Tensor,
    reference: Optional[GlyphMetrics] = None,
    edge_threshold: float = EDGE_SHARPNESS_THRESHOLD,
    metric_threshold: float = METRIC_VARIANCE_THRESHOLD,
) -> ReadabilityResult:
    """
    Validate readability of a single glyph.

    Args:
        image: Glyph image tensor (1, H, W) or (H, W), values in [0, 1]
        reference: Reference metrics for comparison (defaults to Open Sans)
        edge_threshold: Minimum edge sharpness (default 0.7)
        metric_threshold: Maximum deviation from reference (default 0.15)

    Returns:
        ReadabilityResult with all validation details
    """
    # Compute edge sharpness
    edge_sharpness = compute_edge_sharpness(image)

    # Compute metrics
    metrics = compute_glyph_metrics(image)

    # Check blank/saturated
    is_blank = check_is_blank(image)
    is_saturated = check_is_saturated(image)

    # Edge threshold check
    passes_edge = edge_sharpness >= edge_threshold

    # Metrics deviation check
    deviation = compute_reference_deviation(metrics, reference)
    # Average deviation across all metrics
    avg_deviation = sum(deviation.values()) / len(deviation) if deviation else 0.0
    passes_metrics = avg_deviation <= metric_threshold

    return ReadabilityResult(
        edge_sharpness=edge_sharpness,
        metrics=metrics,
        is_blank=is_blank,
        is_saturated=is_saturated,
        passes_edge_threshold=passes_edge,
        passes_metrics_threshold=passes_metrics,
        reference_deviation=deviation,
    )


def validate_batch_readability(
    images: torch.Tensor,
    reference: Optional[GlyphMetrics] = None,
    edge_threshold: float = EDGE_SHARPNESS_THRESHOLD,
    metric_threshold: float = METRIC_VARIANCE_THRESHOLD,
) -> BatchReadabilityResult:
    """
    Validate readability for a batch of glyphs.

    Args:
        images: Batch of glyph images (B, 1, H, W)
        reference: Reference metrics for comparison
        edge_threshold: Minimum edge sharpness
        metric_threshold: Maximum deviation from reference

    Returns:
        BatchReadabilityResult with aggregated results
    """
    if images.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B, 1, H, W), got {images.dim()}D")

    results = []
    sharpness_sum = 0.0
    readable_count = 0

    for i in range(images.size(0)):
        result = validate_glyph_readability(
            images[i],
            reference=reference,
            edge_threshold=edge_threshold,
            metric_threshold=metric_threshold,
        )
        results.append(result)
        sharpness_sum += result.edge_sharpness
        if result.is_readable:
            readable_count += 1

    return BatchReadabilityResult(
        results=results,
        mean_edge_sharpness=sharpness_sum / len(results) if results else 0.0,
        readable_count=readable_count,
        total_count=len(results),
    )


def get_readability_summary(results: List[ReadabilityResult]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of readability results.

    Args:
        results: List of ReadabilityResult objects

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            "mean_edge_sharpness": 0.0,
            "min_edge_sharpness": 0.0,
            "max_edge_sharpness": 0.0,
            "readable_ratio": 0.0,
            "blank_count": 0,
            "saturated_count": 0,
        }

    sharpness_values = [r.edge_sharpness for r in results]

    return {
        "mean_edge_sharpness": sum(sharpness_values) / len(sharpness_values),
        "min_edge_sharpness": min(sharpness_values),
        "max_edge_sharpness": max(sharpness_values),
        "readable_ratio": sum(1 for r in results if r.is_readable) / len(results),
        "blank_count": sum(1 for r in results if r.is_blank),
        "saturated_count": sum(1 for r in results if r.is_saturated),
    }
