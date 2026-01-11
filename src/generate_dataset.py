"""
Training dataset generation for Generative Font Renderer.

Generates 14,040 character samples from Open Sans with augmentations:
- 26 characters (a-z)
- 27 context variations (start token + 26 previous characters)
- 20 augmentations per character-context pair

Output: data/train_dataset.pt with keys: images, prev_chars, curr_chars
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.config import CONFIG


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_font(font_path: Path, size: int = 80) -> ImageFont.FreeTypeFont:
    """Load the font file for rendering."""
    if not font_path.exists():
        raise FileNotFoundError(
            f"Font not found at {font_path}. "
            "Please download OpenSans-Regular.ttf to the fonts/ directory."
        )
    return ImageFont.truetype(str(font_path), size)


def render_character(
    char: str,
    font: ImageFont.FreeTypeFont,
    image_size: int = 128,
) -> Image.Image:
    """
    Render a single character centered in a grayscale image.

    Returns a PIL Image with white background (1.0) and black text (0.0).
    """
    # Create white background
    img = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    # Get character bounding box for centering
    bbox = draw.textbbox((0, 0), char, font=font)
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    # Center the character
    x = (image_size - char_width) // 2 - bbox[0]
    y = (image_size - char_height) // 2 - bbox[1]

    # Draw black text on white background
    draw.text((x, y), char, font=font, fill=0)

    return img


def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    """Apply rotation augmentation, filling background with white."""
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)


def apply_scale(img: Image.Image, scale: float) -> Image.Image:
    """Apply scale augmentation, maintaining image dimensions."""
    size = img.size[0]
    new_size = int(size * scale)

    # Scale the image
    scaled = img.resize((new_size, new_size), resample=Image.BILINEAR)

    # Create output with white background
    result = Image.new("L", (size, size), color=255)

    # Paste centered
    offset = (size - new_size) // 2
    if scale > 1.0:
        # Crop center for upscaled
        crop_offset = (new_size - size) // 2
        cropped = scaled.crop((crop_offset, crop_offset, crop_offset + size, crop_offset + size))
        result = cropped
    else:
        # Paste for downscaled
        result.paste(scaled, (offset, offset))

    return result


def apply_translation(img: Image.Image, tx: int, ty: int) -> Image.Image:
    """Apply translation augmentation."""
    size = img.size[0]
    result = Image.new("L", (size, size), color=255)
    result.paste(img, (tx, ty))
    return result


def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur augmentation."""
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_noise(img_array: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian noise to normalized image array."""
    if std <= 0:
        return img_array
    noise = rng.normal(0, std, img_array.shape).astype(np.float32)
    noisy = img_array + noise
    return np.clip(noisy, 0.0, 1.0)


def generate_augmentation_params(
    aug_idx: int,
    num_augmentations: int,
    rng: np.random.Generator,
) -> dict:
    """
    Generate deterministic augmentation parameters.

    Each augmentation index gets a unique combination of parameters
    that varies smoothly across the augmentation space.
    """
    # Rotation: varies from min to max
    rot_min, rot_max = CONFIG.rotation_range
    rotation = rot_min + (rot_max - rot_min) * rng.random()

    # Scale: varies from min to max
    scale_min, scale_max = CONFIG.scale_range
    scale = scale_min + (scale_max - scale_min) * rng.random()

    # Translation: random within range
    trans_min, trans_max = CONFIG.translate_range
    tx = rng.integers(trans_min, trans_max + 1)
    ty = rng.integers(trans_min, trans_max + 1)

    # Blur: varies from 0 to max
    blur_min, blur_max = CONFIG.blur_range
    blur = blur_min + (blur_max - blur_min) * rng.random()

    # Noise: varies from 0 to max std
    noise_std = CONFIG.noise_std * rng.random()

    return {
        "rotation": rotation,
        "scale": scale,
        "translation": (int(tx), int(ty)),
        "blur": blur,
        "noise_std": noise_std,
    }


def apply_augmentations(
    img: Image.Image,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply all augmentations to an image.

    Returns normalized float32 array in [0, 1] range.
    """
    # Apply geometric transformations
    aug_img = apply_rotation(img, params["rotation"])
    aug_img = apply_scale(aug_img, params["scale"])
    aug_img = apply_translation(aug_img, *params["translation"])

    # Apply blur
    aug_img = apply_blur(aug_img, params["blur"])

    # Convert to normalized array (0 = black text, 1 = white background)
    img_array = np.array(aug_img, dtype=np.float32) / 255.0

    # Apply noise
    img_array = apply_noise(img_array, params["noise_std"], rng)

    return img_array


def validate_image(img_array: np.ndarray, char: str) -> bool:
    """Validate that the image contains actual character pixels."""
    # Check that there are some dark pixels (text)
    dark_pixels = np.sum(img_array < 0.5)
    if dark_pixels < 10:
        print(f"Warning: Character '{char}' has very few dark pixels ({dark_pixels})")
        return False
    return True


def generate_dataset(
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Generate the complete training dataset.

    Args:
        seed: Random seed for reproducibility
        output_path: Path to save the .pt file (default: data/train_dataset.pt)

    Returns:
        Dictionary with keys: images, prev_chars, curr_chars
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    if output_path is None:
        output_path = CONFIG.data_dir / "train_dataset.pt"

    # Load font
    print(f"Loading font from {CONFIG.font_path}...")
    font = load_font(CONFIG.font_path)

    # Character set: a-z (26 characters)
    characters = "abcdefghijklmnopqrstuvwxyz"
    num_chars = len(characters)  # 26
    num_contexts = num_chars + 1  # 27 (start token + 26 previous chars)
    num_augs = CONFIG.num_augmentations  # 20

    total_samples = num_chars * num_contexts * num_augs  # 14,040
    print(f"Generating {total_samples} samples...")
    print(f"  {num_chars} characters x {num_contexts} contexts x {num_augs} augmentations")

    # Pre-render base characters
    print("Pre-rendering base characters...")
    base_images = {}
    for char in characters:
        base_images[char] = render_character(char, font, CONFIG.image_size)

    # Allocate output tensors
    images = np.zeros((total_samples, 1, CONFIG.image_size, CONFIG.image_size), dtype=np.float32)
    prev_chars = np.zeros(total_samples, dtype=np.int64)
    curr_chars = np.zeros(total_samples, dtype=np.int64)

    # Generate samples
    sample_idx = 0
    for curr_idx, curr_char in enumerate(characters):
        curr_char_id = curr_idx + 1  # 1-26 for a-z
        base_img = base_images[curr_char]

        for prev_idx in range(num_contexts):
            # prev_idx: 0 = start token, 1-26 = a-z
            prev_char_id = prev_idx

            for aug_idx in range(num_augs):
                # Generate deterministic augmentation params
                params = generate_augmentation_params(aug_idx, num_augs, rng)

                # Apply augmentations
                aug_array = apply_augmentations(base_img, params, rng)

                # Validate
                if sample_idx < 26:  # Only validate first batch
                    validate_image(aug_array, curr_char)

                # Store
                images[sample_idx, 0] = aug_array
                prev_chars[sample_idx] = prev_char_id
                curr_chars[sample_idx] = curr_char_id
                sample_idx += 1

        # Progress update
        progress = (curr_idx + 1) / num_chars * 100
        print(f"  Progress: {progress:.0f}% ({curr_idx + 1}/{num_chars} characters)")

    print(f"Generated {sample_idx} samples")

    # Convert to tensors
    dataset = {
        "images": torch.from_numpy(images),
        "prev_chars": torch.from_numpy(prev_chars),
        "curr_chars": torch.from_numpy(curr_chars),
    }

    # Validate shapes
    assert dataset["images"].shape == (total_samples, 1, CONFIG.image_size, CONFIG.image_size), \
        f"Invalid images shape: {dataset['images'].shape}"
    assert dataset["prev_chars"].shape == (total_samples,), \
        f"Invalid prev_chars shape: {dataset['prev_chars'].shape}"
    assert dataset["curr_chars"].shape == (total_samples,), \
        f"Invalid curr_chars shape: {dataset['curr_chars'].shape}"

    # Validate value ranges
    assert dataset["images"].min() >= 0.0, f"Images min below 0: {dataset['images'].min()}"
    assert dataset["images"].max() <= 1.0, f"Images max above 1: {dataset['images'].max()}"
    assert dataset["prev_chars"].min() >= 0, "prev_chars min below 0"
    assert dataset["prev_chars"].max() <= 26, "prev_chars max above 26"
    assert dataset["curr_chars"].min() >= 1, "curr_chars min below 1"
    assert dataset["curr_chars"].max() <= 26, "curr_chars max above 26"

    # Save
    print(f"Saving dataset to {output_path}...")
    torch.save(dataset, output_path)

    # Verify saved file
    loaded = torch.load(output_path, weights_only=True)
    assert loaded["images"].shape == dataset["images"].shape, "Saved file verification failed"
    print(f"Dataset saved successfully ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return dataset


def main():
    """Generate the training dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    generate_dataset(seed=args.seed, output_path=output_path)


if __name__ == "__main__":
    main()
