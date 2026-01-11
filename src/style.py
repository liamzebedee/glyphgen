"""
Style vector management for personality expression.

Provides the StyleVector class for managing 64-dimensional personality vectors
with support for interpolation, normalization, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .config import CONFIG


@dataclass
class StyleVector:
    """
    64-dimensional style latent vector for personality expression.

    Encapsulates a style_z tensor with operations for initialization,
    interpolation, normalization, and persistence.

    Attributes:
        data: The underlying 64-dimensional float tensor
        name: Optional personality name (e.g., "fruity", "aggressive")
    """

    data: torch.Tensor
    name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the style vector after initialization."""
        if self.data.dim() != 1:
            raise ValueError(f"StyleVector must be 1D, got {self.data.dim()}D")
        if self.data.size(0) != CONFIG.style_dim:
            raise ValueError(
                f"StyleVector must have {CONFIG.style_dim} dimensions, "
                f"got {self.data.size(0)}"
            )
        # Ensure float dtype
        if not self.data.is_floating_point():
            self.data = self.data.float()

    @classmethod
    def zeros(cls, name: Optional[str] = None) -> StyleVector:
        """
        Create a zero-initialized style vector (neutral personality).

        Args:
            name: Optional personality name

        Returns:
            StyleVector with all zeros
        """
        return cls(torch.zeros(CONFIG.style_dim), name=name)

    @classmethod
    def random(
        cls,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        std: float = 1.0,
    ) -> StyleVector:
        """
        Create a randomly initialized style vector.

        Args:
            name: Optional personality name
            seed: Random seed for reproducibility
            std: Standard deviation for normal distribution

        Returns:
            StyleVector with random values from N(0, std^2)
        """
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            data = torch.randn(CONFIG.style_dim, generator=generator) * std
        else:
            data = torch.randn(CONFIG.style_dim) * std
        return cls(data, name=name)

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, name: Optional[str] = None
    ) -> StyleVector:
        """
        Create a StyleVector from an existing tensor.

        Args:
            tensor: 64-dimensional tensor
            name: Optional personality name

        Returns:
            StyleVector wrapping the tensor (cloned to avoid aliasing)
        """
        return cls(tensor.detach().clone(), name=name)

    @classmethod
    def load(cls, path: Union[str, Path]) -> StyleVector:
        """
        Load a style vector from disk.

        Args:
            path: Path to saved style vector (.pt file)

        Returns:
            Loaded StyleVector

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Style vector not found: {path}")

        data = torch.load(path, weights_only=True)

        if isinstance(data, torch.Tensor):
            # Raw tensor format
            return cls(data)
        elif isinstance(data, dict):
            # Dict format with metadata
            if "data" in data:
                tensor = data["data"]
            elif "style_z" in data:
                tensor = data["style_z"]
            else:
                raise ValueError("Invalid style vector format: missing 'data' or 'style_z' key")
            name = data.get("name")
            return cls(tensor, name=name)
        else:
            raise ValueError(f"Invalid style vector format: expected Tensor or dict, got {type(data)}")

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the style vector to disk.

        Args:
            path: Output path for .pt file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "data": self.data.detach().cpu(),
            "name": self.name,
            "dim": CONFIG.style_dim,
        }
        torch.save(data, path)

    def to(self, device: Union[str, torch.device]) -> StyleVector:
        """
        Move style vector to specified device.

        Args:
            device: Target device

        Returns:
            New StyleVector on target device
        """
        return StyleVector(self.data.to(device), name=self.name)

    def clone(self) -> StyleVector:
        """
        Create a deep copy of this style vector.

        Returns:
            New StyleVector with cloned data
        """
        return StyleVector(self.data.clone(), name=self.name)

    def norm(self) -> float:
        """
        Compute L2 norm of the style vector.

        Returns:
            L2 norm as float
        """
        return self.data.norm(p=2).item()

    def clamp_norm(self, max_norm: float) -> StyleVector:
        """
        Clamp L2 norm to maximum value for stability.

        If the current norm exceeds max_norm, scales the vector down.
        Otherwise, returns the vector unchanged.

        Args:
            max_norm: Maximum allowed L2 norm

        Returns:
            New StyleVector with clamped norm
        """
        current_norm = self.norm()
        if current_norm > max_norm and current_norm > 0:
            scale = max_norm / current_norm
            return StyleVector(self.data * scale, name=self.name)
        return self.clone()

    def normalize(self, target_norm: float = 1.0) -> StyleVector:
        """
        Normalize to specified L2 norm.

        Args:
            target_norm: Target L2 norm (default 1.0 for unit vector)

        Returns:
            New StyleVector with specified norm
        """
        current_norm = self.norm()
        if current_norm > 0:
            scale = target_norm / current_norm
            return StyleVector(self.data * scale, name=self.name)
        return self.clone()

    def interpolate(self, other: StyleVector, t: float) -> StyleVector:
        """
        Linear interpolation between this vector and another.

        Args:
            other: Target style vector
            t: Interpolation factor (0.0 = self, 1.0 = other)

        Returns:
            New StyleVector at interpolated position

        Raises:
            ValueError: If t is not in [0, 1]
        """
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"Interpolation factor must be in [0, 1], got {t}")

        # Linear interpolation
        interpolated = (1.0 - t) * self.data + t * other.data

        # Create name for interpolated result
        name: Optional[str] = None
        if self.name and other.name and self.name != other.name:
            if t == 0.0:
                name = self.name
            elif t == 1.0:
                name = other.name
            else:
                name = f"{self.name}-{other.name}-{t:.2f}"
        else:
            name = self.name or other.name

        return StyleVector(interpolated, name=name)

    def perturb(self, std: float = 0.1, seed: Optional[int] = None) -> StyleVector:
        """
        Create a perturbed copy by adding Gaussian noise.

        Useful for exploration during online learning.

        Args:
            std: Standard deviation of noise
            seed: Random seed for reproducibility

        Returns:
            New StyleVector with added noise
        """
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(CONFIG.style_dim, generator=generator) * std
        else:
            noise = torch.randn_like(self.data) * std

        return StyleVector(self.data + noise, name=self.name)

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return f"StyleVector(dim={CONFIG.style_dim}, norm={self.norm():.4f}{name_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StyleVector):
            return NotImplemented
        return torch.allclose(self.data, other.data, atol=1e-6)


def create_personality_vectors(
    personalities: Optional[list[str]] = None,
    seed: int = 42,
) -> dict[str, StyleVector]:
    """
    Create initial random style vectors for a list of personalities.

    Each personality gets a deterministically seeded random vector
    to ensure reproducibility.

    Args:
        personalities: List of personality names (defaults to CONFIG.personalities)
        seed: Base random seed

    Returns:
        Dictionary mapping personality names to StyleVectors
    """
    if personalities is None:
        personalities = CONFIG.personalities

    vectors = {}
    for i, name in enumerate(personalities):
        # Use personality index as part of seed for uniqueness
        personality_seed = seed + i * 1000
        vectors[name] = StyleVector.random(name=name, seed=personality_seed)

    return vectors


def interpolate_personalities(
    style1: StyleVector,
    style2: StyleVector,
    steps: int = 5,
) -> list[StyleVector]:
    """
    Create a sequence of interpolated style vectors between two personalities.

    Args:
        style1: Starting style vector
        style2: Ending style vector
        steps: Number of interpolation steps (including endpoints)

    Returns:
        List of StyleVectors from style1 to style2
    """
    if steps < 2:
        raise ValueError(f"Steps must be at least 2, got {steps}")

    result = []
    for i in range(steps):
        t = i / (steps - 1)
        result.append(style1.interpolate(style2, t))

    return result
