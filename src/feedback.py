"""
Claude Vision API feedback collection for glyph personality evaluation.

Collects qualitative personality ratings (1-10 scale) from Claude for generated glyphs.
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import CONFIG
from src.image_encoding import get_media_type, tensor_to_base64

# Configure logging
logger = logging.getLogger(__name__)


# Constants
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2
INITIAL_BACKOFF = 1.0  # seconds

# Personality dimension prompts
PERSONALITY_PROMPTS = {
    "fruity": (
        "Rate how 'fruity' this glyph is (1=minimal, 10=extremely fruity). "
        "Consider curves, roundness, playful characteristics, and organic flow."
    ),
    "aggressive": (
        "Rate how 'aggressive' this glyph is (1=minimal, 10=extremely aggressive). "
        "Consider sharp angles, boldness, tension, and confrontational energy."
    ),
    "dumb": (
        "Rate how 'dumb' or goofy this glyph is (1=minimal, 10=extremely dumb). "
        "Consider irregularity, comic-like qualities, and naive charm."
    ),
    "elegant": (
        "Rate how 'elegant' this glyph is (1=minimal, 10=extremely elegant). "
        "Consider refinement, balance, sophistication, and graceful proportions."
    ),
}


@dataclass
class FeedbackResult:
    """Result of a single glyph feedback evaluation."""

    glyph_char: str
    ratings: Dict[str, float]
    timestamp: datetime
    model: str
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "glyph_char": self.glyph_char,
            "ratings": self.ratings,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "success": self.success,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackResult":
        """Create from dictionary."""
        return cls(
            glyph_char=data["glyph_char"],
            ratings=data["ratings"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model=data["model"],
            success=data["success"],
            error=data.get("error"),
        )


@dataclass
class EvaluationRun:
    """Results of a complete evaluation run with multiple glyphs."""

    run_number: int
    font_id: str
    personality: str
    results: List[FeedbackResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_number": self.run_number,
            "font_id": self.font_id,
            "personality": self.personality,
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRun":
        """Create from dictionary."""
        return cls(
            run_number=data["run_number"],
            font_id=data["font_id"],
            personality=data["personality"],
            results=[FeedbackResult.from_dict(r) for r in data["results"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class ClaudeFeedbackClient:
    """
    Client for collecting personality feedback from Claude Vision API.

    Handles image encoding, API calls with retry logic, and response parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize the Claude feedback client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use (must support vision)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient failures
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it or pass api_key to the constructor."
            )

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Import anthropic here to allow module import without API key
        import anthropic

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _build_prompt(self, personality: str, dimensions: Optional[List[str]] = None) -> str:
        """
        Build the evaluation prompt for a given personality.

        Args:
            personality: Target personality being evaluated
            dimensions: Specific dimensions to rate (defaults to all)

        Returns:
            Formatted prompt string
        """
        if dimensions is None:
            dimensions = list(PERSONALITY_PROMPTS.keys())

        dimension_prompts = []
        for dim in dimensions:
            if dim in PERSONALITY_PROMPTS:
                dimension_prompts.append(f"- {dim.upper()}: {PERSONALITY_PROMPTS[dim]}")
            else:
                # Generic prompt for unknown dimensions
                dimension_prompts.append(
                    f"- {dim.upper()}: Rate how '{dim}' this glyph is (1=minimal, 10=extreme)."
                )

        return f"""You are evaluating a generated glyph image for personality traits.
The target personality for this font is: {personality}

Please rate this glyph on the following dimensions (1-10 scale):
{chr(10).join(dimension_prompts)}

IMPORTANT: Respond ONLY with a JSON object containing the ratings. No other text.
Example response format:
{{"fruity": 7, "aggressive": 3, "dumb": 5, "elegant": 6}}

Rate based on the visual characteristics of the glyph you see in the image."""

    def _parse_response(self, response_text: str) -> Dict[str, float]:
        """
        Parse Claude's response into ratings dictionary.

        Args:
            response_text: Raw response text from Claude

        Returns:
            Dictionary mapping dimension names to ratings

        Raises:
            ValueError: If response cannot be parsed
        """
        # Try to extract JSON from response
        text = response_text.strip()

        # Handle potential markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Try to find JSON object
        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]

        try:
            ratings = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}\nResponse: {response_text}")

        # Validate ratings
        validated = {}
        for key, value in ratings.items():
            key_lower = key.lower()
            try:
                rating = float(value)
                if rating < 1 or rating > 10:
                    logger.warning(f"Rating for {key} out of range: {rating}, clamping to [1, 10]")
                    rating = max(1, min(10, rating))
                validated[key_lower] = rating
            except (ValueError, TypeError):
                logger.warning(f"Invalid rating value for {key}: {value}, skipping")

        if not validated:
            raise ValueError(f"No valid ratings found in response: {response_text}")

        return validated

    def evaluate_glyph(
        self,
        glyph_tensor,
        glyph_char: str,
        personality: str,
        dimensions: Optional[List[str]] = None,
    ) -> FeedbackResult:
        """
        Evaluate a single glyph for personality traits.

        Args:
            glyph_tensor: Glyph tensor with shape (1, H, W) or (H, W), values in [0, 1]
            glyph_char: Character this glyph represents
            personality: Target personality being evaluated
            dimensions: Specific dimensions to rate (defaults to all)

        Returns:
            FeedbackResult with ratings or error information
        """
        timestamp = datetime.now()

        # Encode image
        try:
            b64_image = tensor_to_base64(glyph_tensor)
            media_type = get_media_type("PNG")
        except Exception as e:
            logger.error(f"Failed to encode glyph '{glyph_char}': {e}")
            return FeedbackResult(
                glyph_char=glyph_char,
                ratings={},
                timestamp=timestamp,
                model=self.model,
                success=False,
                error=f"Encoding error: {e}",
            )

        # Build prompt
        prompt = self._build_prompt(personality, dimensions)

        # Call API with retry logic
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_image,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )

                # Parse response
                response_text = response.content[0].text
                ratings = self._parse_response(response_text)

                return FeedbackResult(
                    glyph_char=glyph_char,
                    ratings=ratings,
                    timestamp=timestamp,
                    model=self.model,
                    success=True,
                )

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Check for non-retryable errors
                if "AuthenticationError" in error_type or "401" in str(e):
                    logger.error(f"Authentication failed: {e}")
                    return FeedbackResult(
                        glyph_char=glyph_char,
                        ratings={},
                        timestamp=timestamp,
                        model=self.model,
                        success=False,
                        error=f"Authentication error: {e}",
                    )

                # Rate limit - check for Retry-After header
                if "RateLimitError" in error_type or "429" in str(e):
                    retry_after = getattr(e, "retry_after", None) or (INITIAL_BACKOFF * (2**attempt))
                    logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                    time.sleep(retry_after)
                    continue

                # Timeout or other transient error - retry with backoff
                if attempt < self.max_retries:
                    backoff = INITIAL_BACKOFF * (2**attempt)
                    logger.warning(f"API error (attempt {attempt + 1}): {e}, retrying in {backoff}s")
                    time.sleep(backoff)
                    continue

        # All retries exhausted
        logger.error(f"Failed to evaluate glyph '{glyph_char}' after {self.max_retries + 1} attempts: {last_error}")
        return FeedbackResult(
            glyph_char=glyph_char,
            ratings={},
            timestamp=timestamp,
            model=self.model,
            success=False,
            error=f"API error after retries: {last_error}",
        )

    def evaluate_batch(
        self,
        glyphs: List[Tuple[Any, str]],
        personality: str,
        dimensions: Optional[List[str]] = None,
    ) -> List[FeedbackResult]:
        """
        Evaluate multiple glyphs for personality traits.

        Args:
            glyphs: List of (tensor, char) tuples
            personality: Target personality being evaluated
            dimensions: Specific dimensions to rate

        Returns:
            List of FeedbackResult objects
        """
        results = []
        for tensor, char in glyphs:
            result = self.evaluate_glyph(tensor, char, personality, dimensions)
            results.append(result)

            # Small delay between requests to avoid rate limits
            if result.success:
                time.sleep(0.5)

        return results


def sample_glyphs(
    glyphs: Dict[str, Any],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: Optional[int] = None,
) -> List[Tuple[Any, str]]:
    """
    Randomly sample glyphs for evaluation.

    Args:
        glyphs: Dictionary mapping characters to tensors
        sample_size: Number of glyphs to sample
        seed: Optional random seed for reproducibility

    Returns:
        List of (tensor, char) tuples

    Raises:
        ValueError: If population is smaller than sample size
    """
    if len(glyphs) < sample_size:
        logger.warning(
            f"Population ({len(glyphs)}) smaller than sample size ({sample_size}), "
            f"sampling all available glyphs"
        )
        sample_size = len(glyphs)

    if seed is not None:
        random.seed(seed)

    chars = random.sample(list(glyphs.keys()), sample_size)
    return [(glyphs[char], char) for char in chars]


def get_feedback_dir(font_id: str) -> Path:
    """Get the feedback directory for a font."""
    return CONFIG.networks_dir / font_id / "feedback"


def get_feedback_file_path(font_id: str, run_number: int, sample_index: int, glyph_char: str) -> Path:
    """
    Get the path for a feedback file.

    Path format: networks/[id]/feedback/N_eval_X_glyph-[char]_feedback.txt
    """
    feedback_dir = get_feedback_dir(font_id)
    return feedback_dir / f"{run_number}_eval_{sample_index}_glyph-{glyph_char}_feedback.txt"


def save_feedback(
    font_id: str,
    run_number: int,
    sample_index: int,
    result: FeedbackResult,
) -> Path:
    """
    Save feedback for a single glyph evaluation.

    Args:
        font_id: Font identifier
        run_number: Evaluation run number
        sample_index: Index of this sample within the run (0-2)
        result: FeedbackResult to save

    Returns:
        Path to the saved feedback file
    """
    feedback_dir = get_feedback_dir(font_id)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    file_path = get_feedback_file_path(font_id, run_number, sample_index, result.glyph_char)

    # Write feedback as JSON
    with open(file_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return file_path


def save_evaluation_run(font_id: str, run: EvaluationRun) -> Path:
    """
    Save a complete evaluation run.

    Args:
        font_id: Font identifier
        run: EvaluationRun to save

    Returns:
        Path to the saved run file
    """
    feedback_dir = get_feedback_dir(font_id)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    file_path = feedback_dir / f"run_{run.run_number}_summary.json"

    with open(file_path, "w") as f:
        json.dump(run.to_dict(), f, indent=2)

    return file_path


def load_evaluation_run(font_id: str, run_number: int) -> Optional[EvaluationRun]:
    """
    Load a saved evaluation run.

    Args:
        font_id: Font identifier
        run_number: Run number to load

    Returns:
        EvaluationRun or None if not found
    """
    file_path = get_feedback_dir(font_id) / f"run_{run_number}_summary.json"

    if not file_path.exists():
        return None

    try:
        with open(file_path) as f:
            data = json.load(f)
        return EvaluationRun.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load evaluation run {run_number}: {e}")
        return None


def get_next_run_number(font_id: str) -> int:
    """
    Get the next available run number for a font.

    Args:
        font_id: Font identifier

    Returns:
        Next run number (1-indexed)
    """
    feedback_dir = get_feedback_dir(font_id)
    if not feedback_dir.exists():
        return 1

    max_run = 0
    for file_path in feedback_dir.glob("run_*_summary.json"):
        try:
            run_num = int(file_path.stem.split("_")[1])
            max_run = max(max_run, run_num)
        except (IndexError, ValueError):
            continue

    return max_run + 1


def collect_feedback(
    font_id: str,
    glyphs: Dict[str, Any],
    personality: str,
    client: Optional[ClaudeFeedbackClient] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    dimensions: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> EvaluationRun:
    """
    Collect personality feedback for a set of glyphs.

    This is the main entry point for feedback collection.

    Args:
        font_id: Font identifier
        glyphs: Dictionary mapping characters to tensors
        personality: Target personality being evaluated
        client: Optional ClaudeFeedbackClient (created if not provided)
        sample_size: Number of glyphs to sample
        dimensions: Specific dimensions to rate
        seed: Optional random seed for reproducibility

    Returns:
        EvaluationRun with all results
    """
    # Create client if not provided
    if client is None:
        try:
            client = ClaudeFeedbackClient()
        except ValueError as e:
            logger.error(f"Cannot create feedback client: {e}")
            # Return empty run with error
            run_number = get_next_run_number(font_id)
            return EvaluationRun(
                run_number=run_number,
                font_id=font_id,
                personality=personality,
                results=[],
            )

    # Get run number
    run_number = get_next_run_number(font_id)

    # Sample glyphs
    sampled = sample_glyphs(glyphs, sample_size, seed)

    # Evaluate glyphs
    results = []
    for idx, (tensor, char) in enumerate(sampled):
        result = client.evaluate_glyph(tensor, char, personality, dimensions)
        results.append(result)

        # Save individual feedback file
        save_feedback(font_id, run_number, idx, result)

        logger.info(
            f"Evaluated glyph '{char}': "
            f"{'success' if result.success else 'failed'} "
            f"ratings={result.ratings if result.success else result.error}"
        )

    # Create and save evaluation run
    run = EvaluationRun(
        run_number=run_number,
        font_id=font_id,
        personality=personality,
        results=results,
    )
    save_evaluation_run(font_id, run)

    # Log summary
    successful = sum(1 for r in results if r.success)
    logger.info(
        f"Feedback collection complete: {successful}/{len(results)} successful "
        f"for font '{font_id}' personality '{personality}'"
    )

    return run


def list_evaluation_runs(font_id: str) -> List[int]:
    """
    List all evaluation run numbers for a font.

    Args:
        font_id: Font identifier

    Returns:
        List of run numbers, sorted ascending
    """
    feedback_dir = get_feedback_dir(font_id)
    if not feedback_dir.exists():
        return []

    runs = []
    for file_path in feedback_dir.glob("run_*_summary.json"):
        try:
            run_num = int(file_path.stem.split("_")[1])
            runs.append(run_num)
        except (IndexError, ValueError):
            continue

    return sorted(runs)


def compute_average_ratings(runs: List[EvaluationRun]) -> Dict[str, float]:
    """
    Compute average ratings across multiple evaluation runs.

    Args:
        runs: List of EvaluationRun objects

    Returns:
        Dictionary mapping dimension names to average ratings
    """
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for run in runs:
        for result in run.results:
            if result.success:
                for dim, rating in result.ratings.items():
                    totals[dim] = totals.get(dim, 0) + rating
                    counts[dim] = counts.get(dim, 0) + 1

    return {dim: totals[dim] / counts[dim] for dim in totals if counts[dim] > 0}
