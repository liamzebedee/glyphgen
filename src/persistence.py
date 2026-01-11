"""
Model persistence layer for Generative Font Renderer.

Manages storage, retrieval, and versioning of trained models organized by font ID.
Directory structure: networks/[id]/runs/[timestamp]/
"""

import json
import re
import shutil
import tempfile
import torch
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image

from src.config import CONFIG


# Constants
FONT_ID_PATTERN = re.compile(r"^[0-9]+-[a-z][a-z0-9-]*$")
WEIGHTS_VERSION = "v1"
FEEDBACK_SCHEMA = "1.0"


@dataclass
class RunMetadata:
    """Metadata for a single training run."""

    run_id: str
    font_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    completed: bool = False
    training_params: Dict[str, Any] = field(default_factory=dict)
    weights_version: str = WEIGHTS_VERSION
    feedback_schema: str = FEEDBACK_SCHEMA

    @property
    def run_dir(self) -> Path:
        """Get the directory path for this run."""
        return CONFIG.networks_dir / self.font_id / "runs" / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "font_id": self.font_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "completed": self.completed,
            "training_params": self.training_params,
            "weights_version": self.weights_version,
            "feedback_schema": self.feedback_schema,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMetadata":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            font_id=data["font_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
            ),
            completed=data.get("completed", False),
            training_params=data.get("training_params", {}),
            weights_version=data.get("weights_version", "v1"),
            feedback_schema=data.get("feedback_schema", "1.0"),
        )


def validate_font_id(font_id: str) -> bool:
    """
    Validate font ID format: {number}-{name}.

    Args:
        font_id: Font ID to validate (e.g., "1-fruity", "2-aggressive-sans")

    Returns:
        True if valid, False otherwise
    """
    return bool(FONT_ID_PATTERN.match(font_id))


def generate_run_id() -> str:
    """
    Generate a unique run ID based on ISO 8601 timestamp with microseconds.

    Returns:
        Run ID string (e.g., "2026-01-11T08:30:00.123456")
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


def get_font_dir(font_id: str) -> Path:
    """Get the directory path for a font ID."""
    if not validate_font_id(font_id):
        raise ValueError(
            f"Invalid font ID '{font_id}'. Must match pattern {{number}}-{{name}} "
            "(e.g., '1-fruity', '2-aggressive-sans')"
        )
    return CONFIG.networks_dir / font_id


def get_runs_dir(font_id: str) -> Path:
    """Get the runs directory for a font ID."""
    return get_font_dir(font_id) / "runs"


def ensure_font_dirs(font_id: str) -> Path:
    """
    Ensure font directories exist, creating if necessary.

    Args:
        font_id: Font ID (e.g., "1-fruity")

    Returns:
        Path to the runs directory
    """
    runs_dir = get_runs_dir(font_id)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def list_fonts() -> List[str]:
    """
    List all font IDs in the networks directory.

    Returns:
        List of font ID strings, sorted alphabetically
    """
    if not CONFIG.networks_dir.exists():
        return []

    fonts = []
    for item in CONFIG.networks_dir.iterdir():
        if item.is_dir() and validate_font_id(item.name):
            fonts.append(item.name)

    return sorted(fonts)


def list_runs(font_id: str) -> List[RunMetadata]:
    """
    List all runs for a font ID in chronological order.

    Args:
        font_id: Font ID (e.g., "1-fruity")

    Returns:
        List of RunMetadata objects, oldest first
    """
    runs_dir = get_runs_dir(font_id)
    if not runs_dir.exists():
        return []

    runs = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                runs.append(RunMetadata.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                # Handle corrupted metadata - create minimal metadata from directory name
                try:
                    start_time = datetime.fromisoformat(run_dir.name)
                except ValueError:
                    start_time = datetime.now()
                runs.append(
                    RunMetadata(
                        run_id=run_dir.name,
                        font_id=font_id,
                        start_time=start_time,
                        completed=False,
                    )
                )
        else:
            # No metadata file - infer from directory name
            try:
                start_time = datetime.fromisoformat(run_dir.name)
            except ValueError:
                start_time = datetime.now()
            runs.append(
                RunMetadata(
                    run_id=run_dir.name,
                    font_id=font_id,
                    start_time=start_time,
                    completed=(run_dir / "weights").exists(),
                )
            )

    # Sort chronologically by start_time
    return sorted(runs, key=lambda r: r.start_time)


def get_run_info(font_id: str, run_id: str) -> RunMetadata:
    """
    Get metadata for a specific run.

    Args:
        font_id: Font ID (e.g., "1-fruity")
        run_id: Run ID (timestamp string)

    Returns:
        RunMetadata object

    Raises:
        FileNotFoundError: If run does not exist
    """
    run_dir = get_runs_dir(font_id) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run '{run_id}' not found for font '{font_id}'")

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return RunMetadata.from_dict(json.load(f))

    # Fallback: create metadata from directory
    try:
        start_time = datetime.fromisoformat(run_id)
    except ValueError:
        start_time = datetime.now()

    return RunMetadata(
        run_id=run_id,
        font_id=font_id,
        start_time=start_time,
        completed=(run_dir / "weights").exists(),
    )


def get_latest_run(font_id: str) -> Optional[RunMetadata]:
    """
    Get the most recent run for a font ID.

    Args:
        font_id: Font ID (e.g., "1-fruity")

    Returns:
        RunMetadata for the latest run, or None if no runs exist
    """
    runs = list_runs(font_id)
    return runs[-1] if runs else None


def _atomic_save(data: Any, path: Path, save_func) -> None:
    """
    Save data atomically using temp file + rename.

    Args:
        data: Data to save
        path: Target path
        save_func: Function to call with (data, temp_path)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename
    fd, temp_path_str = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_path_str)

    try:
        # Close the file descriptor since we'll use the path
        import os

        os.close(fd)

        # Save to temp file
        save_func(data, temp_path)

        # Atomic rename
        temp_path.replace(path)
    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def save_model(
    font_id: str,
    model: torch.nn.Module,
    style_z: torch.Tensor,
    sample_image: Optional[Image.Image] = None,
    eval_glyphs: Optional[Dict[str, Image.Image]] = None,
    feedback: Optional[Dict[str, Any]] = None,
    training_params: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    Save model state for a training run.

    Args:
        font_id: Font ID (e.g., "1-fruity")
        model: PyTorch model to save
        style_z: Personality vector (64-dim tensor)
        sample_image: Optional sample output image
        eval_glyphs: Optional dict of char -> PIL Image for evaluation glyphs
        feedback: Optional feedback data
        training_params: Optional training parameters for metadata
        run_id: Optional run ID (generated if not provided)

    Returns:
        Run ID of the saved model

    Raises:
        ValueError: If font_id is invalid
    """
    if not validate_font_id(font_id):
        raise ValueError(
            f"Invalid font ID '{font_id}'. Must match pattern {{number}}-{{name}}"
        )

    # Generate or use provided run ID
    if run_id is None:
        run_id = generate_run_id()
        start_time = datetime.now()
    else:
        # Parse start_time from run_id if it's a valid ISO timestamp
        try:
            start_time = datetime.fromisoformat(run_id)
        except ValueError:
            start_time = datetime.now()

    # Create run directory
    run_dir = get_runs_dir(font_id) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata
    metadata = RunMetadata(
        run_id=run_id,
        font_id=font_id,
        start_time=start_time,
        end_time=datetime.now(),
        completed=True,
        training_params=training_params or {},
    )

    # Save weights with personality vector
    weights_data = {
        "model_state_dict": model.state_dict(),
        "style_z": style_z.detach().cpu(),
        "weights_version": WEIGHTS_VERSION,
        "timestamp": datetime.now().isoformat(),
    }

    def save_weights(data, path):
        torch.save(data, path)

    _atomic_save(weights_data, run_dir / "weights", save_weights)

    # Save sample image if provided
    if sample_image is not None:
        sample_path = run_dir / "sample.png"
        sample_image.save(sample_path, "PNG")

    # Save eval glyphs if provided
    if eval_glyphs:
        eval_dir = run_dir / "eval-glyphs"
        eval_dir.mkdir(exist_ok=True)
        for char, img in eval_glyphs.items():
            img.save(eval_dir / f"{char}.png", "PNG")

    # Save feedback if provided
    if feedback is not None:
        feedback_data = {
            "feedback_schema": FEEDBACK_SCHEMA,
            "data": feedback,
            "timestamp": datetime.now().isoformat(),
        }

        def save_json(data, path):
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        _atomic_save(feedback_data, run_dir / "feedback.json", save_json)

    # Save metadata
    def save_metadata(data, path):
        with open(path, "w") as f:
            json.dump(data.to_dict(), f, indent=2)

    _atomic_save(metadata, run_dir / "metadata.json", save_metadata)

    return run_id


def load_model(
    font_id: str,
    model: torch.nn.Module,
    run_id: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, RunMetadata]:
    """
    Load model state from a training run.

    Args:
        font_id: Font ID (e.g., "1-fruity")
        model: PyTorch model to load weights into
        run_id: Optional run ID (defaults to latest run)
        device: Optional device to load weights to

    Returns:
        Tuple of (style_z tensor, RunMetadata)

    Raises:
        FileNotFoundError: If font or run not found
        RuntimeError: If weights file is corrupted
    """
    # Get run to load
    if run_id is None:
        latest = get_latest_run(font_id)
        if latest is None:
            raise FileNotFoundError(f"No runs found for font '{font_id}'")
        run_id = latest.run_id

    run_dir = get_runs_dir(font_id) / run_id
    weights_path = run_dir / "weights"

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found for run '{run_id}'")

    # Load weights with corruption detection
    try:
        weights_data = torch.load(weights_path, weights_only=False, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Corrupted weights file for run '{run_id}': {e}. "
            "The file may be damaged or incompatible. "
            "Try loading a previous run or regenerating the model."
        )

    # Validate weights structure
    required_keys = ["model_state_dict", "style_z"]
    missing_keys = [k for k in required_keys if k not in weights_data]
    if missing_keys:
        raise RuntimeError(
            f"Invalid weights file for run '{run_id}': missing keys {missing_keys}"
        )

    # Load model state
    model.load_state_dict(weights_data["model_state_dict"])

    # Get style_z
    style_z = weights_data["style_z"]
    if device is not None:
        style_z = style_z.to(device)

    # Get metadata
    metadata = get_run_info(font_id, run_id)

    return style_z, metadata


def load_feedback(font_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    """
    Load feedback data for a run.

    Args:
        font_id: Font ID
        run_id: Run ID

    Returns:
        Feedback data dict, or None if no feedback file exists
    """
    feedback_path = get_runs_dir(font_id) / run_id / "feedback.json"
    if not feedback_path.exists():
        return None

    try:
        with open(feedback_path) as f:
            data = json.load(f)
        return data.get("data", data)  # Handle both schema versions
    except json.JSONDecodeError:
        return None


def load_sample_image(font_id: str, run_id: str) -> Optional[Image.Image]:
    """
    Load sample image for a run.

    Args:
        font_id: Font ID
        run_id: Run ID

    Returns:
        PIL Image, or None if no sample exists
    """
    sample_path = get_runs_dir(font_id) / run_id / "sample.png"
    if not sample_path.exists():
        return None
    return Image.open(sample_path)


def load_eval_glyphs(font_id: str, run_id: str) -> Dict[str, Image.Image]:
    """
    Load evaluation glyphs for a run.

    Args:
        font_id: Font ID
        run_id: Run ID

    Returns:
        Dict of char -> PIL Image
    """
    eval_dir = get_runs_dir(font_id) / run_id / "eval-glyphs"
    if not eval_dir.exists():
        return {}

    glyphs = {}
    for img_path in eval_dir.glob("*.png"):
        char = img_path.stem
        glyphs[char] = Image.open(img_path)

    return glyphs


def delete_run(font_id: str, run_id: str) -> None:
    """
    Delete a run and all its files.

    Args:
        font_id: Font ID
        run_id: Run ID

    Raises:
        FileNotFoundError: If run does not exist
    """
    run_dir = get_runs_dir(font_id) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run '{run_id}' not found for font '{font_id}'")

    shutil.rmtree(run_dir)


def delete_font(font_id: str) -> None:
    """
    Delete a font and all its runs.

    Args:
        font_id: Font ID

    Raises:
        FileNotFoundError: If font does not exist
    """
    font_dir = get_font_dir(font_id)
    if not font_dir.exists():
        raise FileNotFoundError(f"Font '{font_id}' not found")

    shutil.rmtree(font_dir)


def get_run_files(font_id: str, run_id: str) -> Dict[str, bool]:
    """
    Check which files exist for a run.

    Args:
        font_id: Font ID
        run_id: Run ID

    Returns:
        Dict mapping file names to existence status
    """
    run_dir = get_runs_dir(font_id) / run_id
    return {
        "weights": (run_dir / "weights").exists(),
        "sample.png": (run_dir / "sample.png").exists(),
        "eval-glyphs": (run_dir / "eval-glyphs").is_dir(),
        "feedback.json": (run_dir / "feedback.json").exists(),
        "metadata.json": (run_dir / "metadata.json").exists(),
    }


def is_run_complete(font_id: str, run_id: str) -> bool:
    """
    Check if a run has all required files.

    Args:
        font_id: Font ID
        run_id: Run ID

    Returns:
        True if weights file exists (minimum requirement for usable run)
    """
    run_dir = get_runs_dir(font_id) / run_id
    return (run_dir / "weights").exists()


def export_style_z(font_id: str, run_id: str, path: Path) -> None:
    """
    Export personality vector to a separate file.

    Args:
        font_id: Font ID
        run_id: Run ID
        path: Output path for the style_z tensor
    """
    weights_path = get_runs_dir(font_id) / run_id / "weights"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found for run '{run_id}'")

    weights_data = torch.load(weights_path, weights_only=False)
    style_z = weights_data["style_z"]
    torch.save({"style_z": style_z, "font_id": font_id, "run_id": run_id}, path)
