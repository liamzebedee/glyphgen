"""
Image encoding utilities for Claude Vision API integration.

Converts glyph tensors to base64-encoded PNG images for API transmission.
"""

import base64
import io

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a glyph tensor to a PIL Image.

    Args:
        tensor: Glyph tensor with shape (1, H, W) or (H, W), values in [0, 1].
                Grayscale float tensor from model output.

    Returns:
        PIL Image in L mode (grayscale, 8-bit).

    Raises:
        ValueError: If tensor shape or values are invalid.
    """
    # Validate input
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

    # Handle different tensor shapes
    if tensor.dim() == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected single-channel tensor (1, H, W), got {tensor.shape}"
            )
        tensor = tensor.squeeze(0)  # (H, W)
    elif tensor.dim() != 2:
        raise ValueError(
            f"Expected tensor with 2 or 3 dimensions, got {tensor.dim()}"
        )

    # Move to CPU and convert to numpy
    array = tensor.detach().cpu().numpy()

    # Validate value range
    if array.min() < -0.01 or array.max() > 1.01:
        raise ValueError(
            f"Tensor values must be in [0, 1], got [{array.min():.3f}, {array.max():.3f}]"
        )

    # Clip to valid range and convert to uint8
    array = np.clip(array, 0.0, 1.0)
    array_uint8 = (array * 255).astype(np.uint8)

    # Create PIL Image
    return Image.fromarray(array_uint8, mode="L")


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL Image to a base64-encoded string.

    Args:
        image: PIL Image to encode.
        format: Image format for encoding (default: PNG for lossless).

    Returns:
        Base64-encoded string of the image bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def tensor_to_base64(tensor: torch.Tensor, format: str = "PNG") -> str:
    """
    Convert a glyph tensor directly to a base64-encoded PNG string.

    This is the main function for preparing glyph images for Claude Vision API.

    Args:
        tensor: Glyph tensor with shape (1, H, W) or (H, W), values in [0, 1].
        format: Image format for encoding (default: PNG for lossless).

    Returns:
        Base64-encoded string of the PNG image.

    Example:
        >>> glyph = model(prev_char, curr_char, style_z)  # (1, 1, 128, 128)
        >>> b64 = tensor_to_base64(glyph[0])  # Encode single glyph
        >>> # Use b64 in Claude Vision API call
    """
    image = tensor_to_pil(tensor)
    return pil_to_base64(image, format=format)


def base64_to_pil(b64_string: str) -> Image.Image:
    """
    Decode a base64 string back to a PIL Image.

    Useful for validating round-trip encoding fidelity.

    Args:
        b64_string: Base64-encoded image string.

    Returns:
        PIL Image decoded from the string.
    """
    image_bytes = base64.b64decode(b64_string)
    buffer = io.BytesIO(image_bytes)
    return Image.open(buffer)


def base64_to_tensor(b64_string: str) -> torch.Tensor:
    """
    Decode a base64 string back to a tensor.

    Useful for validating round-trip encoding fidelity.

    Args:
        b64_string: Base64-encoded image string.

    Returns:
        Tensor with shape (1, H, W), values in [0, 1].
    """
    image = base64_to_pil(b64_string)
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)  # (1, H, W)


def batch_to_base64(batch: torch.Tensor, format: str = "PNG") -> list[str]:
    """
    Convert a batch of glyph tensors to base64-encoded strings.

    Args:
        batch: Batch tensor with shape (B, 1, H, W), values in [0, 1].
        format: Image format for encoding (default: PNG).

    Returns:
        List of base64-encoded strings, one per glyph in the batch.

    Raises:
        ValueError: If batch shape is invalid.
    """
    if batch.dim() != 4:
        raise ValueError(
            f"Expected 4D batch tensor (B, 1, H, W), got {batch.dim()}D"
        )
    if batch.shape[1] != 1:
        raise ValueError(
            f"Expected single-channel batch (B, 1, H, W), got {batch.shape}"
        )

    return [tensor_to_base64(batch[i], format=format) for i in range(batch.shape[0])]


def validate_encoding_fidelity(
    original: torch.Tensor,
    encoded: str,
    max_error: float = 1.0 / 255.0,
) -> bool:
    """
    Validate that encoding/decoding preserves image fidelity.

    Due to uint8 quantization, some small error is expected.
    Default threshold is 1/255 (one quantization level).

    Args:
        original: Original tensor with shape (1, H, W) or (H, W).
        encoded: Base64-encoded string of the image.
        max_error: Maximum allowed per-pixel error (default: 1/255).

    Returns:
        True if round-trip error is within tolerance.
    """
    decoded = base64_to_tensor(encoded)

    # Handle shape differences
    orig = original.detach().cpu()
    if orig.dim() == 2:
        orig = orig.unsqueeze(0)

    # Check shapes match
    if orig.shape != decoded.shape:
        return False

    # Check values are close
    max_diff = (orig - decoded).abs().max().item()
    return max_diff <= max_error + 1e-6  # Small epsilon for float precision


def get_media_type(format: str = "PNG") -> str:
    """
    Get the MIME media type for an image format.

    Used for Claude Vision API content type specification.

    Args:
        format: Image format (PNG, JPEG, etc.).

    Returns:
        MIME type string (e.g., "image/png").
    """
    format_lower = format.lower()
    media_types = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return media_types.get(format_lower, f"image/{format_lower}")
