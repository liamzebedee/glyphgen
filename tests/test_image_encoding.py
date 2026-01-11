"""
Tests for image encoding utilities.

Tests cover:
- tensor_to_pil conversion with various tensor shapes
- pil_to_base64 encoding
- tensor_to_base64 direct conversion
- Round-trip encoding/decoding fidelity
- Batch encoding
- Error handling for invalid inputs
- Media type mapping
"""

import base64
import io
from typing import Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from src.image_encoding import (
    base64_to_pil,
    base64_to_tensor,
    batch_to_base64,
    get_media_type,
    pil_to_base64,
    tensor_to_base64,
    tensor_to_pil,
    validate_encoding_fidelity,
)
from src.config import CONFIG


class TestTensorToPil:
    """Test tensor to PIL Image conversion."""

    def test_converts_2d_tensor(self):
        """2D tensor (H, W) converts to PIL Image."""
        tensor = torch.rand(128, 128)
        image = tensor_to_pil(tensor)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert image.size == (128, 128)

    def test_converts_3d_tensor(self):
        """3D tensor (1, H, W) converts to PIL Image."""
        tensor = torch.rand(1, 128, 128)
        image = tensor_to_pil(tensor)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert image.size == (128, 128)

    def test_handles_model_output_shape(self):
        """Handles typical model output shape (1, 128, 128)."""
        tensor = torch.rand(1, CONFIG.image_size, CONFIG.image_size)
        image = tensor_to_pil(tensor)
        assert image.size == (CONFIG.image_size, CONFIG.image_size)

    def test_preserves_black_pixels(self):
        """Black pixels (0.0) convert correctly."""
        tensor = torch.zeros(1, 128, 128)
        image = tensor_to_pil(tensor)
        array = np.array(image)
        assert array.min() == 0
        assert array.max() == 0

    def test_preserves_white_pixels(self):
        """White pixels (1.0) convert correctly."""
        tensor = torch.ones(1, 128, 128)
        image = tensor_to_pil(tensor)
        array = np.array(image)
        assert array.min() == 255
        assert array.max() == 255

    def test_preserves_gradient(self):
        """Gradient values convert correctly."""
        tensor = torch.linspace(0, 1, 128).unsqueeze(0).expand(128, -1)
        image = tensor_to_pil(tensor)
        array = np.array(image)
        # First column should be near black, last near white
        assert array[:, 0].mean() < 10
        assert array[:, -1].mean() > 245

    def test_rejects_invalid_type(self):
        """Raises TypeError for non-tensor input."""
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            tensor_to_pil(np.zeros((128, 128)))

    def test_rejects_4d_tensor(self):
        """Raises ValueError for 4D tensor (batch)."""
        tensor = torch.rand(2, 1, 128, 128)
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            tensor_to_pil(tensor)

    def test_rejects_1d_tensor(self):
        """Raises ValueError for 1D tensor."""
        tensor = torch.rand(128)
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            tensor_to_pil(tensor)

    def test_rejects_multi_channel(self):
        """Raises ValueError for multi-channel tensor."""
        tensor = torch.rand(3, 128, 128)
        with pytest.raises(ValueError, match="single-channel"):
            tensor_to_pil(tensor)

    def test_rejects_out_of_range_negative(self):
        """Raises ValueError for negative values."""
        tensor = torch.rand(1, 128, 128) - 0.5
        with pytest.raises(ValueError, match="must be in"):
            tensor_to_pil(tensor)

    def test_rejects_out_of_range_positive(self):
        """Raises ValueError for values > 1."""
        tensor = torch.rand(1, 128, 128) + 0.5
        with pytest.raises(ValueError, match="must be in"):
            tensor_to_pil(tensor)

    def test_tolerates_small_floating_point_errors(self):
        """Tolerates small floating point errors beyond [0, 1]."""
        # Values very slightly outside range due to float precision
        tensor = torch.tensor([[[-0.001, 1.001]]])
        # Should not raise, will clip
        image = tensor_to_pil(tensor)
        assert image is not None

    def test_handles_gpu_tensor(self):
        """GPU tensors convert correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        tensor = torch.rand(1, 128, 128, device="cuda")
        image = tensor_to_pil(tensor)
        assert isinstance(image, Image.Image)


class TestPilToBase64:
    """Test PIL to base64 encoding."""

    def test_returns_string(self):
        """Encoding returns a string."""
        image = Image.new("L", (128, 128), color=128)
        result = pil_to_base64(image)
        assert isinstance(result, str)

    def test_returns_valid_base64(self):
        """Encoded string is valid base64."""
        image = Image.new("L", (128, 128), color=128)
        result = pil_to_base64(image)
        # Should not raise
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_default_format_is_png(self):
        """Default format produces PNG."""
        image = Image.new("L", (128, 128), color=128)
        result = pil_to_base64(image)
        decoded = base64.b64decode(result)
        # PNG magic bytes
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_jpeg_format(self):
        """JPEG format produces JPEG."""
        image = Image.new("L", (128, 128), color=128)
        result = pil_to_base64(image, format="JPEG")
        decoded = base64.b64decode(result)
        # JPEG magic bytes
        assert decoded[:2] == b"\xff\xd8"

    def test_different_sizes_produce_different_lengths(self):
        """Larger images produce longer base64 strings."""
        small = Image.new("L", (32, 32), color=128)
        large = Image.new("L", (256, 256), color=128)
        small_b64 = pil_to_base64(small)
        large_b64 = pil_to_base64(large)
        assert len(large_b64) > len(small_b64)


class TestTensorToBase64:
    """Test direct tensor to base64 conversion."""

    def test_returns_string(self):
        """Conversion returns a string."""
        tensor = torch.rand(1, 128, 128)
        result = tensor_to_base64(tensor)
        assert isinstance(result, str)

    def test_returns_valid_base64(self):
        """Conversion produces valid base64."""
        tensor = torch.rand(1, 128, 128)
        result = tensor_to_base64(tensor)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_produces_png_by_default(self):
        """Default produces PNG format."""
        tensor = torch.rand(1, 128, 128)
        result = tensor_to_base64(tensor)
        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_produces_jpeg_when_specified(self):
        """JPEG format when specified."""
        tensor = torch.rand(1, 128, 128)
        result = tensor_to_base64(tensor, format="JPEG")
        decoded = base64.b64decode(result)
        assert decoded[:2] == b"\xff\xd8"


class TestBase64ToPil:
    """Test base64 to PIL decoding."""

    def test_decodes_valid_base64(self):
        """Decodes valid base64 image."""
        image = Image.new("L", (128, 128), color=128)
        b64 = pil_to_base64(image)
        decoded = base64_to_pil(b64)
        assert isinstance(decoded, Image.Image)

    def test_preserves_dimensions(self):
        """Decoded image has correct dimensions."""
        image = Image.new("L", (64, 128), color=128)
        b64 = pil_to_base64(image)
        decoded = base64_to_pil(b64)
        assert decoded.size == (64, 128)

    def test_round_trip_preserves_content(self):
        """Round-trip preserves image content."""
        # Create image with known pattern
        original = Image.new("L", (128, 128))
        pixels = original.load()
        for y in range(128):
            for x in range(128):
                pixels[x, y] = (x + y) % 256
        b64 = pil_to_base64(original)
        decoded = base64_to_pil(b64)
        assert np.array_equal(np.array(original), np.array(decoded))


class TestBase64ToTensor:
    """Test base64 to tensor decoding."""

    def test_returns_tensor(self):
        """Decoding returns a tensor."""
        tensor = torch.rand(1, 128, 128)
        b64 = tensor_to_base64(tensor)
        result = base64_to_tensor(b64)
        assert isinstance(result, torch.Tensor)

    def test_correct_shape(self):
        """Decoded tensor has shape (1, H, W)."""
        tensor = torch.rand(1, 64, 128)
        b64 = tensor_to_base64(tensor)
        result = base64_to_tensor(b64)
        assert result.shape == (1, 64, 128)

    def test_values_in_range(self):
        """Decoded tensor values are in [0, 1]."""
        tensor = torch.rand(1, 128, 128)
        b64 = tensor_to_base64(tensor)
        result = base64_to_tensor(b64)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestRoundTripFidelity:
    """Test encoding/decoding round-trip fidelity."""

    def test_validate_fidelity_passes_for_valid(self):
        """Validation passes for proper round-trip."""
        tensor = torch.rand(1, 128, 128)
        b64 = tensor_to_base64(tensor)
        assert validate_encoding_fidelity(tensor, b64)

    def test_fidelity_within_quantization_error(self):
        """Round-trip error is within quantization tolerance."""
        tensor = torch.rand(1, 128, 128)
        b64 = tensor_to_base64(tensor)
        decoded = base64_to_tensor(b64)
        max_error = (tensor - decoded).abs().max().item()
        # Max error should be at most 1/255 (one quantization level)
        assert max_error <= 1.0 / 255.0 + 1e-6

    def test_fidelity_for_known_values(self):
        """Known values round-trip correctly."""
        # Test exact quantization boundaries
        values = [0.0, 0.5, 1.0, 0.25, 0.75]
        for val in values:
            tensor = torch.full((1, 32, 32), val)
            b64 = tensor_to_base64(tensor)
            decoded = base64_to_tensor(b64)
            # Allow 1/255 error due to quantization
            assert (tensor - decoded).abs().max().item() <= 1.0 / 255.0 + 1e-6

    def test_validate_fidelity_handles_2d_tensor(self):
        """Validation works with 2D original tensor."""
        tensor = torch.rand(128, 128)
        b64 = tensor_to_base64(tensor)
        assert validate_encoding_fidelity(tensor, b64)

    def test_validate_fidelity_fails_for_wrong_size(self):
        """Validation fails when sizes don't match."""
        tensor1 = torch.rand(1, 64, 64)
        tensor2 = torch.rand(1, 128, 128)
        b64 = tensor_to_base64(tensor2)
        assert not validate_encoding_fidelity(tensor1, b64)


class TestBatchToBase64:
    """Test batch encoding."""

    def test_returns_list(self):
        """Batch encoding returns a list."""
        batch = torch.rand(4, 1, 128, 128)
        result = batch_to_base64(batch)
        assert isinstance(result, list)

    def test_correct_count(self):
        """Returns one encoding per batch item."""
        batch = torch.rand(5, 1, 128, 128)
        result = batch_to_base64(batch)
        assert len(result) == 5

    def test_each_is_valid_base64(self):
        """Each batch item produces valid base64."""
        batch = torch.rand(3, 1, 128, 128)
        result = batch_to_base64(batch)
        for b64 in result:
            decoded = base64.b64decode(b64)
            assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_batch_items_are_independent(self):
        """Different batch items produce different encodings."""
        batch = torch.rand(2, 1, 128, 128)
        batch[0] = 0.0  # All black
        batch[1] = 1.0  # All white
        result = batch_to_base64(batch)
        assert result[0] != result[1]

    def test_single_item_batch(self):
        """Single-item batch works correctly."""
        batch = torch.rand(1, 1, 128, 128)
        result = batch_to_base64(batch)
        assert len(result) == 1

    def test_rejects_3d_input(self):
        """Raises ValueError for 3D input."""
        tensor = torch.rand(1, 128, 128)
        with pytest.raises(ValueError, match="4D batch tensor"):
            batch_to_base64(tensor)

    def test_rejects_multi_channel(self):
        """Raises ValueError for multi-channel batch."""
        batch = torch.rand(2, 3, 128, 128)
        with pytest.raises(ValueError, match="single-channel"):
            batch_to_base64(batch)


class TestGetMediaType:
    """Test media type mapping."""

    def test_png(self):
        """PNG maps to image/png."""
        assert get_media_type("PNG") == "image/png"
        assert get_media_type("png") == "image/png"

    def test_jpeg(self):
        """JPEG maps to image/jpeg."""
        assert get_media_type("JPEG") == "image/jpeg"
        assert get_media_type("jpeg") == "image/jpeg"
        assert get_media_type("JPG") == "image/jpeg"
        assert get_media_type("jpg") == "image/jpeg"

    def test_gif(self):
        """GIF maps to image/gif."""
        assert get_media_type("GIF") == "image/gif"
        assert get_media_type("gif") == "image/gif"

    def test_webp(self):
        """WebP maps to image/webp."""
        assert get_media_type("WEBP") == "image/webp"
        assert get_media_type("webp") == "image/webp"

    def test_unknown_format(self):
        """Unknown format uses lowercase."""
        assert get_media_type("TIFF") == "image/tiff"
        assert get_media_type("BMP") == "image/bmp"


class TestIntegrationWithModel:
    """Integration tests with model output shapes."""

    def test_model_output_batch(self):
        """Handles typical model batch output (B, 1, 128, 128)."""
        # Simulate model output
        batch_size = 8
        output = torch.rand(batch_size, 1, CONFIG.image_size, CONFIG.image_size)
        result = batch_to_base64(output)
        assert len(result) == batch_size
        for b64 in result:
            assert isinstance(b64, str)
            assert len(b64) > 100  # Reasonable length for PNG

    def test_single_glyph_from_batch(self):
        """Single glyph extraction and encoding."""
        batch = torch.rand(4, 1, 128, 128)
        single = batch[0]  # Shape: (1, 128, 128)
        b64 = tensor_to_base64(single)
        decoded = base64_to_tensor(b64)
        assert decoded.shape == (1, 128, 128)

    def test_encoding_for_api_content_type(self):
        """Encoding produces valid content for API."""
        tensor = torch.rand(1, 128, 128)
        b64 = tensor_to_base64(tensor)
        media_type = get_media_type("PNG")
        # Simulate API content structure
        content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }
        assert content["source"]["media_type"] == "image/png"
        assert len(content["source"]["data"]) > 0
