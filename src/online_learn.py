"""
Online learning system for REINFORCE-style policy gradient updates.

Enables real-time weight updates based on Claude feedback without full model retraining.
Implements:
- REINFORCE policy gradient computation
- EMA baseline tracking for variance reduction
- Selective decoder fine-tuning
- Weight persistence with versioning
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.glyphnet import GlyphNetwork

from src.config import CONFIG
from src.feedback import EvaluationRun, FeedbackResult
from src.persistence import (
    generate_run_id,
    get_runs_dir,
    validate_font_id,
    RunMetadata,
)
from src.style import StyleVector

# Configure logging
logger = logging.getLogger(__name__)


# Constants
UPDATE_BUDGET_MS = 500  # Max time for single feedback cycle
MEMORY_BUDGET_MB = 50   # Max memory overhead per network
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_WEIGHT_BOUND = 5.0


@dataclass
class BaselineTracker:
    """
    Exponential moving average baseline for variance reduction in REINFORCE.

    Tracks expected rating to compute advantage = rating - baseline.

    Attributes:
        alpha: EMA coefficient (high = slow adaptation, typical 0.9-0.99)
        value: Current baseline value
        count: Number of observations
        min_observations: Minimum observations before baseline is reliable
    """

    alpha: float = 0.9
    value: float = 5.5  # Initialize to middle of 1-10 scale
    count: int = 0
    min_observations: int = 3

    def update(self, rating: float) -> float:
        """
        Update baseline with new rating.

        Args:
            rating: New rating value (1-10 scale)

        Returns:
            Updated baseline value
        """
        if self.count == 0:
            # First observation: set baseline to rating
            self.value = rating
        else:
            # EMA update: baseline_t = alpha * rating + (1 - alpha) * baseline_{t-1}
            self.value = self.alpha * rating + (1 - self.alpha) * self.value

        self.count += 1
        return self.value

    def get_advantage(self, rating: float) -> float:
        """
        Compute advantage for a rating.

        Args:
            rating: Current rating value

        Returns:
            Advantage (rating - baseline)
        """
        return rating - self.value

    @property
    def is_reliable(self) -> bool:
        """Check if baseline has enough observations to be reliable."""
        return self.count >= self.min_observations

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alpha": self.alpha,
            "value": self.value,
            "count": self.count,
            "min_observations": self.min_observations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineTracker":
        """Deserialize from dictionary."""
        return cls(
            alpha=data.get("alpha", 0.9),
            value=data.get("value", 5.5),
            count=data.get("count", 0),
            min_observations=data.get("min_observations", 3),
        )


@dataclass
class GradientStats:
    """Statistics from a gradient computation/update cycle."""

    advantage: float
    gradient_norm: float
    style_update_norm: float
    decoder_update_norm: float
    update_time_ms: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "advantage": self.advantage,
            "gradient_norm": self.gradient_norm,
            "style_update_norm": self.style_update_norm,
            "decoder_update_norm": self.decoder_update_norm,
            "update_time_ms": self.update_time_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class OnlineLearningState:
    """
    Persistent state for online learning session.

    Tracks baseline, update history, and weight versions.
    """

    font_id: str
    run_id: str
    baseline: BaselineTracker = field(default_factory=BaselineTracker)
    weight_version: int = 0
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    style_z_history: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def state_dir(self) -> Path:
        """Get directory for this learning state."""
        return get_runs_dir(self.font_id) / self.run_id

    def save(self) -> Path:
        """
        Save learning state to disk.

        Returns:
            Path to saved state file
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.state_dir / "online_state.json"

        state_data = {
            "font_id": self.font_id,
            "run_id": self.run_id,
            "baseline": self.baseline.to_dict(),
            "weight_version": self.weight_version,
            "update_history": self.update_history,
            "style_z_history": self.style_z_history,
            "start_time": self.start_time.isoformat(),
        }

        with open(state_path, "w") as f:
            json.dump(state_data, f, indent=2)

        return state_path

    @classmethod
    def load(cls, font_id: str, run_id: str) -> "OnlineLearningState":
        """
        Load learning state from disk.

        Args:
            font_id: Font identifier
            run_id: Run identifier

        Returns:
            Loaded state or new state if not found
        """
        state_path = get_runs_dir(font_id) / run_id / "online_state.json"

        if not state_path.exists():
            return cls(font_id=font_id, run_id=run_id)

        try:
            with open(state_path) as f:
                data = json.load(f)

            return cls(
                font_id=data["font_id"],
                run_id=data["run_id"],
                baseline=BaselineTracker.from_dict(data["baseline"]),
                weight_version=data.get("weight_version", 0),
                update_history=data.get("update_history", []),
                style_z_history=data.get("style_z_history", []),
                start_time=datetime.fromisoformat(data["start_time"]),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load online state: {e}, creating new state")
            return cls(font_id=font_id, run_id=run_id)


def compute_policy_gradient(
    feedback_rating: float,
    baseline: BaselineTracker,
    style_z: torch.Tensor,
    model: nn.Module,
    generated_glyph: Optional[torch.Tensor],
    learning_rate: float = CONFIG.online_lr,
    gradient_clip: float = DEFAULT_GRADIENT_CLIP,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], GradientStats]:
    """
    Compute REINFORCE-style policy gradient from feedback.

    The REINFORCE algorithm uses advantage to scale gradients:
    - Positive advantage (rating > baseline) -> increase probability of style_z
    - Negative advantage (rating < baseline) -> decrease probability of style_z

    Args:
        feedback_rating: Claude rating (1-10 scale)
        baseline: Baseline tracker for variance reduction
        style_z: Style vector that produced the glyph (requires_grad=True for gradient)
        model: GlyphNetwork for decoder gradient computation
        generated_glyph: The generated glyph tensor
        learning_rate: Step size for updates
        gradient_clip: Max gradient L2 norm

    Returns:
        Tuple of (style_z_gradient, decoder_gradients_dict, GradientStats)
    """
    start_time = time.time()

    # Compute advantage
    advantage = baseline.get_advantage(feedback_rating)

    # For REINFORCE, we compute gradient of log-probability scaled by advantage
    # Since we're treating style_z as the action, we use a simple gradient direction
    # proportional to how the glyph was generated

    # The gradient direction for style_z: advantage * style_z / ||style_z||
    # This encourages similar styles for positive feedback, dissimilar for negative
    style_norm = style_z.norm(p=2).clamp(min=1e-8)
    style_z_grad = advantage * style_z / style_norm

    # Scale by learning rate
    style_z_grad = style_z_grad * learning_rate

    # Clip gradient norm
    grad_norm = style_z_grad.norm(p=2).item()
    if grad_norm > gradient_clip:
        style_z_grad = style_z_grad * (gradient_clip / grad_norm)
        grad_norm = gradient_clip

    # For decoder gradients, we compute gradients w.r.t. output
    # This allows fine-tuning the last layers to better express style
    decoder_grad_norm = 0.0
    decoder_grads = {}

    # Only compute decoder gradients if we have a generated glyph
    if generated_glyph is not None and generated_glyph.requires_grad:
        # Use the advantage-weighted reconstruction as target
        # Positive advantage -> reinforce current output
        # Negative advantage -> move away from current output

        # Compute gradient of decoder parameters
        for name, param in model.named_parameters():
            if "decoder" in name and param.requires_grad:
                if param.grad is not None:
                    # Scale existing gradients by advantage
                    grad = param.grad * advantage * learning_rate
                    decoder_grads[name] = grad.clone()
                    decoder_grad_norm += grad.norm(p=2).item() ** 2

        decoder_grad_norm = math.sqrt(decoder_grad_norm)

    elapsed_ms = (time.time() - start_time) * 1000

    stats = GradientStats(
        advantage=advantage,
        gradient_norm=grad_norm,
        style_update_norm=style_z_grad.norm(p=2).item(),
        decoder_update_norm=decoder_grad_norm,
        update_time_ms=elapsed_ms,
        success=True,
    )

    return style_z_grad, decoder_grads, stats


def apply_style_update(
    style_z: StyleVector,
    gradient: torch.Tensor,
    max_norm: float = 10.0,
) -> StyleVector:
    """
    Apply gradient update to style vector with stability constraints.

    Args:
        style_z: Current style vector
        gradient: Gradient to apply (positive = increase, negative = decrease)
        max_norm: Maximum L2 norm for resulting style vector

    Returns:
        Updated StyleVector with clamped norm
    """
    # Apply gradient (REINFORCE: add scaled advantage direction)
    updated_data = style_z.data + gradient.to(style_z.data.device)

    # Create new style vector and clamp norm for stability
    updated_style = StyleVector(updated_data, name=style_z.name)
    return updated_style.clamp_norm(max_norm)


def apply_decoder_update(
    model: nn.Module,
    gradients: Dict[str, torch.Tensor],
    weight_bound: float = DEFAULT_WEIGHT_BOUND,
    target_layers: Optional[List[str]] = None,
) -> int:
    """
    Apply gradient updates to decoder layers.

    Args:
        model: GlyphNetwork to update
        gradients: Dict mapping parameter names to gradients
        weight_bound: Max absolute value for weights
        target_layers: Optional list of layer names to update (defaults to last 2)

    Returns:
        Number of parameters updated
    """
    updated_count = 0

    # Default: update last 2 decoder layers
    if target_layers is None:
        target_layers = ["decoder.3", "decoder.4", "decoder.5", "decoder.6"]

    for name, param in model.named_parameters():
        # Check if this parameter should be updated
        should_update = any(layer in name for layer in target_layers)
        if not should_update:
            continue

        if name in gradients:
            grad = gradients[name]

            # Apply gradient update
            with torch.no_grad():
                param.add_(grad)

                # Clamp weights for stability
                param.clamp_(-weight_bound, weight_bound)

            updated_count += 1

    return updated_count


def save_weight_checkpoint(
    font_id: str,
    run_id: str,
    model: nn.Module,
    style_z: StyleVector,
    version: int,
) -> Path:
    """
    Save versioned weight checkpoint.

    Args:
        font_id: Font identifier
        run_id: Run identifier
        model: Model to save
        style_z: Style vector to save
        version: Weight version number

    Returns:
        Path to saved checkpoint
    """
    run_dir = get_runs_dir(font_id) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / f"weights_v{version}.pt"

    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "style_z": style_z.data.detach().cpu(),
        "version": version,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


def load_weight_checkpoint(
    font_id: str,
    run_id: str,
    model: nn.Module,
    version: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[StyleVector, int]:
    """
    Load versioned weight checkpoint.

    Args:
        font_id: Font identifier
        run_id: Run identifier
        model: Model to load weights into
        version: Specific version to load (defaults to latest)
        device: Device to load to

    Returns:
        Tuple of (StyleVector, version number)
    """
    run_dir = get_runs_dir(font_id) / run_id

    if version is None:
        # Find latest version
        versions = []
        for p in run_dir.glob("weights_v*.pt"):
            try:
                v = int(p.stem.split("_v")[1])
                versions.append(v)
            except (ValueError, IndexError):
                continue

        if not versions:
            raise FileNotFoundError(f"No weight checkpoints found in {run_dir}")
        version = max(versions)

    checkpoint_path = run_dir / f"weights_v{version}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint version {version} not found")

    checkpoint_data = torch.load(checkpoint_path, weights_only=False, map_location=device)

    model.load_state_dict(checkpoint_data["model_state_dict"])

    style_data = checkpoint_data["style_z"]
    if device is not None:
        style_data = style_data.to(device)
    style_z = StyleVector(style_data)

    return style_z, checkpoint_data.get("version", version)


class OnlineLearner:
    """
    Main class for online learning with REINFORCE.

    Orchestrates the full feedback → gradient → update → checkpoint cycle.

    Usage:
        learner = OnlineLearner(font_id="1-fruity", model=model, style_z=style)
        for feedback in feedback_results:
            learner.update(feedback)
        learner.finalize()
    """

    def __init__(
        self,
        font_id: str,
        model: nn.Module,
        style_z: StyleVector,
        run_id: Optional[str] = None,
        learning_rate: Optional[float] = None,
        baseline_alpha: Optional[float] = None,
        gradient_clip: float = DEFAULT_GRADIENT_CLIP,
        checkpoint_frequency: int = 1,  # Checkpoint every N updates
        device: Optional[torch.device] = None,
    ):
        """
        Initialize online learner.

        Args:
            font_id: Font identifier
            model: GlyphNetwork to update
            style_z: Initial style vector
            run_id: Optional run ID (generated if not provided)
            learning_rate: Learning rate for updates
            baseline_alpha: EMA coefficient for baseline
            gradient_clip: Max gradient norm
            checkpoint_frequency: How often to save checkpoints
            device: Device for computation
        """
        if not validate_font_id(font_id):
            raise ValueError(f"Invalid font ID: {font_id}")

        self.font_id = font_id
        self.model = model
        self.style_z = style_z
        self.run_id = run_id or generate_run_id()
        self.learning_rate = learning_rate or CONFIG.online_lr
        self.gradient_clip = gradient_clip
        self.checkpoint_frequency = checkpoint_frequency
        self.device = device

        # Initialize or load state
        self.state = OnlineLearningState.load(font_id, self.run_id)

        # Update baseline alpha if specified
        if baseline_alpha is not None:
            self.state.baseline.alpha = baseline_alpha

        # Track updates since last checkpoint
        self._updates_since_checkpoint = 0

        logger.info(
            f"OnlineLearner initialized: font={font_id}, run={self.run_id}, "
            f"lr={self.learning_rate}, baseline={self.state.baseline.value:.2f}"
        )

    def update(
        self,
        feedback: FeedbackResult,
        generated_glyph: Optional[torch.Tensor] = None,
        target_dimension: Optional[str] = None,
    ) -> GradientStats:
        """
        Apply single feedback update.

        Args:
            feedback: FeedbackResult from Claude evaluation
            generated_glyph: Optional generated glyph tensor for decoder updates
            target_dimension: Specific dimension to use (defaults to first)

        Returns:
            GradientStats with update information
        """
        if not feedback.success:
            return GradientStats(
                advantage=0.0,
                gradient_norm=0.0,
                style_update_norm=0.0,
                decoder_update_norm=0.0,
                update_time_ms=0.0,
                success=False,
                error=feedback.error,
            )

        # Get rating for target dimension
        if target_dimension is None:
            # Use first available rating
            if not feedback.ratings:
                return GradientStats(
                    advantage=0.0,
                    gradient_norm=0.0,
                    style_update_norm=0.0,
                    decoder_update_norm=0.0,
                    update_time_ms=0.0,
                    success=False,
                    error="No ratings in feedback",
                )
            target_dimension = list(feedback.ratings.keys())[0]

        rating = feedback.ratings.get(target_dimension)
        if rating is None:
            return GradientStats(
                advantage=0.0,
                gradient_norm=0.0,
                style_update_norm=0.0,
                decoder_update_norm=0.0,
                update_time_ms=0.0,
                success=False,
                error=f"No rating for dimension '{target_dimension}'",
            )

        # Update baseline
        self.state.baseline.update(rating)

        # Compute gradients
        style_z_tensor = self.style_z.data.clone().requires_grad_(True)
        style_grad, decoder_grads, stats = compute_policy_gradient(
            feedback_rating=rating,
            baseline=self.state.baseline,
            style_z=style_z_tensor,
            model=self.model,
            generated_glyph=generated_glyph,
            learning_rate=self.learning_rate,
            gradient_clip=self.gradient_clip,
        )

        # Apply style update
        self.style_z = apply_style_update(self.style_z, style_grad)

        # Apply decoder updates
        if decoder_grads:
            apply_decoder_update(self.model, decoder_grads)

        # Record update history
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "glyph_char": feedback.glyph_char,
            "rating": rating,
            "dimension": target_dimension,
            "stats": stats.to_dict(),
        }
        self.state.update_history.append(update_record)

        # Record style_z history
        self.state.style_z_history.append({
            "timestamp": datetime.now().isoformat(),
            "norm": self.style_z.norm(),
            "mean": self.style_z.data.mean().item(),
            "std": self.style_z.data.std().item(),
        })

        # Increment version and maybe checkpoint
        self.state.weight_version += 1
        self._updates_since_checkpoint += 1

        if self._updates_since_checkpoint >= self.checkpoint_frequency:
            self._checkpoint()

        # Save state
        self.state.save()

        logger.info(
            f"Update applied: char={feedback.glyph_char}, "
            f"rating={rating:.1f}, advantage={stats.advantage:.2f}, "
            f"style_norm={self.style_z.norm():.2f}"
        )

        return stats

    def update_batch(
        self,
        evaluation_run: EvaluationRun,
        generated_glyphs: Optional[Dict[str, torch.Tensor]] = None,
        target_dimension: Optional[str] = None,
    ) -> List[GradientStats]:
        """
        Apply updates from a complete evaluation run.

        Args:
            evaluation_run: EvaluationRun with multiple feedback results
            generated_glyphs: Optional dict of char -> tensor
            target_dimension: Specific dimension to use

        Returns:
            List of GradientStats for each update
        """
        stats_list = []

        for result in evaluation_run.results:
            glyph = None
            if generated_glyphs and result.glyph_char in generated_glyphs:
                glyph = generated_glyphs[result.glyph_char]

            stats = self.update(result, glyph, target_dimension)
            stats_list.append(stats)

        return stats_list

    def _checkpoint(self) -> None:
        """Save weight checkpoint."""
        save_weight_checkpoint(
            font_id=self.font_id,
            run_id=self.run_id,
            model=self.model,
            style_z=self.style_z,
            version=self.state.weight_version,
        )
        self._updates_since_checkpoint = 0
        logger.info(f"Checkpoint saved: version {self.state.weight_version}")

    def finalize(self) -> RunMetadata:
        """
        Finalize learning session.

        Saves final checkpoint and metadata.

        Returns:
            RunMetadata for the completed run
        """
        # Force final checkpoint
        self._checkpoint()
        self.state.save()

        # Create run metadata
        metadata = RunMetadata(
            run_id=self.run_id,
            font_id=self.font_id,
            start_time=self.state.start_time,
            end_time=datetime.now(),
            completed=True,
            training_params={
                "learning_rate": self.learning_rate,
                "baseline_alpha": self.state.baseline.alpha,
                "gradient_clip": self.gradient_clip,
                "total_updates": len(self.state.update_history),
                "final_baseline": self.state.baseline.value,
            },
        )

        # Save metadata
        metadata_path = self.state.state_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(
            f"Online learning finalized: {len(self.state.update_history)} updates, "
            f"final baseline={self.state.baseline.value:.2f}"
        )

        return metadata

    @property
    def current_baseline(self) -> float:
        """Get current baseline value."""
        return self.state.baseline.value

    @property
    def update_count(self) -> int:
        """Get total number of updates applied."""
        return len(self.state.update_history)

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the learning session.

        Returns:
            Dict with learning statistics
        """
        if not self.state.update_history:
            return {
                "update_count": 0,
                "baseline": self.state.baseline.value,
                "baseline_reliable": self.state.baseline.is_reliable,
            }

        # Compute statistics from history
        advantages = [u["stats"]["advantage"] for u in self.state.update_history]
        grad_norms = [u["stats"]["gradient_norm"] for u in self.state.update_history]
        ratings = [u["rating"] for u in self.state.update_history]

        return {
            "update_count": len(self.state.update_history),
            "baseline": self.state.baseline.value,
            "baseline_reliable": self.state.baseline.is_reliable,
            "weight_version": self.state.weight_version,
            "advantage_mean": sum(advantages) / len(advantages),
            "advantage_std": (sum((a - sum(advantages)/len(advantages))**2 for a in advantages) / len(advantages)) ** 0.5,
            "gradient_norm_mean": sum(grad_norms) / len(grad_norms),
            "rating_mean": sum(ratings) / len(ratings),
            "rating_range": (min(ratings), max(ratings)),
            "style_z_norm": self.style_z.norm(),
        }


def run_online_learning(
    font_id: str,
    model: "GlyphNetwork",
    style_z: StyleVector,
    num_runs: int = 10,
    glyphs_per_run: int = 3,
    personality: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[StyleVector, RunMetadata]:
    """
    Run complete online learning loop with Claude feedback.

    This is the main entry point for running online learning.

    Args:
        font_id: Font identifier
        model: GlyphNetwork to train
        style_z: Initial style vector
        num_runs: Number of feedback collection runs
        glyphs_per_run: Glyphs to sample per run
        personality: Target personality (extracted from font_id if not provided)
        device: Device for computation

    Returns:
        Tuple of (updated StyleVector, RunMetadata)
    """
    # Import here to avoid circular dependency
    from src.feedback import collect_feedback
    from src.inference import InferenceEngine

    # Extract personality from font_id if not provided
    if personality is None:
        personality = font_id.split("-", 1)[1] if "-" in font_id else "unknown"

    # Create inference engine
    engine = InferenceEngine(model, device=device, compile_model=False)

    # Create online learner
    learner = OnlineLearner(
        font_id=font_id,
        model=model,
        style_z=style_z,
        device=device,
    )

    for run_num in range(num_runs):
        logger.info(f"Starting feedback run {run_num + 1}/{num_runs}")

        # Generate glyphs with current style
        chars = "abcdefghijklmnopqrstuvwxyz"
        glyphs = {}
        for char in chars:
            char_idx = ord(char) - ord('a') + 1  # 1-26
            glyph = engine.generate_glyph(
                prev_char=0,  # Start token
                curr_char=char_idx,
                style_z=learner.style_z.data,  # Pass tensor, not StyleVector
            )
            glyphs[char] = glyph

        # Collect feedback from Claude
        evaluation_run = collect_feedback(
            font_id=font_id,
            glyphs=glyphs,
            personality=personality,
            sample_size=glyphs_per_run,
            dimensions=[personality],
        )

        # Apply updates
        stats_list = learner.update_batch(
            evaluation_run=evaluation_run,
            generated_glyphs=glyphs,
            target_dimension=personality,
        )

        # Log progress
        successful = sum(1 for s in stats_list if s.success)
        logger.info(
            f"Run {run_num + 1} complete: {successful}/{len(stats_list)} updates, "
            f"baseline={learner.current_baseline:.2f}"
        )

    # Finalize and return results
    metadata = learner.finalize()
    return learner.style_z, metadata


if __name__ == "__main__":
    import argparse
    from src.glyphnet import GlyphNetwork
    from src.utils import get_device

    parser = argparse.ArgumentParser(description="Run online learning")
    parser.add_argument("--personality", type=str, required=True, help="Personality name")
    parser.add_argument("--runs", type=int, default=10, help="Number of feedback runs")
    parser.add_argument("--checkpoint", type=str, help="Path to base checkpoint")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create model
    model = GlyphNetwork().to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Create style vector
    style_z = StyleVector.random(name=args.personality, seed=42)

    # Create font_id
    font_id = f"1-{args.personality}"

    # Run online learning
    updated_style, metadata = run_online_learning(
        font_id=font_id,
        model=model,
        style_z=style_z,
        num_runs=args.runs,
        personality=args.personality,
        device=device,
    )

    logger.info("Online learning complete!")
    logger.info(f"Run ID: {metadata.run_id}")
    logger.info(f"Style vector norm: {updated_style.norm():.4f}")
