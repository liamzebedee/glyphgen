"""
Demo application for the generative font renderer.

Generates "The Revolution Will Not Be Televised" with selected personality
styles and displays/saves the glyph sequence.

Usage:
    # Interactive personality selection
    python -m src.demo

    # Generate with specific personality
    python -m src.demo --personality fruity

    # Generate all personalities
    python -m src.demo --all

    # Custom text
    python -m src.demo --text "hello world" --personality aggressive-sans

    # Save to specific directory
    python -m src.demo -o outputs/demo/
"""

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image

from src.config import CONFIG
from src.generate import create_composite, create_engine, generate_glyphs, load_style


DEFAULT_TEXT = "The Revolution Will Not Be Televised"


def select_personality_interactive() -> str:
    """
    Interactive personality selection via CLI.

    Returns:
        Selected personality name
    """
    print("\nAvailable personalities:")
    for i, personality in enumerate(CONFIG.personalities, 1):
        print(f"  {i}. {personality}")

    while True:
        try:
            choice = input("\nSelect personality (number or name): ").strip()

            # Try as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(CONFIG.personalities):
                    return CONFIG.personalities[idx]
                print(f"Invalid number. Choose 1-{len(CONFIG.personalities)}.")
                continue

            # Try as name
            choice_lower = choice.lower()
            if choice_lower in CONFIG.personalities:
                return choice_lower
            # Try partial match
            matches = [p for p in CONFIG.personalities if p.startswith(choice_lower)]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                print(f"Ambiguous: {', '.join(matches)}")
                continue

            print(f"Unknown personality: {choice}")
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default personality: fruity")
            return "fruity"


def demo_single_personality(
    text: str,
    personality: str,
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    show: bool = True,
    compile_model: bool = False,
) -> tuple[list[Image.Image], Image.Image]:
    """
    Generate demo output for a single personality.

    Args:
        text: Text to render
        personality: Personality name
        checkpoint_path: Optional model checkpoint
        output_dir: Directory to save outputs (None = don't save)
        show: Whether to display the composite image
        compile_model: Whether to use torch.compile()

    Returns:
        Tuple of (list of glyph images, composite image)
    """
    print(f"\nGenerating with personality: {personality}")

    # Create engine and style
    engine = create_engine(checkpoint_path, compile_model=compile_model)
    style = load_style(personality=personality)

    # Generate glyphs
    images = generate_glyphs(text, engine, style)
    composite = create_composite(images)

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save composite
        composite_path = output_dir / f"{personality}_composite.png"
        composite.save(composite_path, "PNG")
        print(f"  Saved: {composite_path}")

        # Save individual glyphs
        chars = [c.lower() for c in text if c.isalpha()]
        for i, (img, char) in enumerate(zip(images, chars)):
            glyph_path = output_dir / f"{personality}_{i:02d}_{char}.png"
            img.save(glyph_path, "PNG")

    # Display if requested
    if show:
        composite.show(title=f"Demo: {personality}")

    alphabetic_count = sum(1 for c in text if c.isalpha())
    print(f"  Generated {alphabetic_count} glyphs")

    return images, composite


def demo_all_personalities(
    text: str,
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    show: bool = True,
    compile_model: bool = False,
) -> dict[str, tuple[list[Image.Image], Image.Image]]:
    """
    Generate demo output for all configured personalities.

    Args:
        text: Text to render
        checkpoint_path: Optional model checkpoint
        output_dir: Directory to save outputs (None = don't save)
        show: Whether to display composite images
        compile_model: Whether to use torch.compile()

    Returns:
        Dict mapping personality name to (images, composite) tuple
    """
    results: dict[str, tuple[list[Image.Image], Image.Image]] = {}

    print(f"Generating for all {len(CONFIG.personalities)} personalities...")

    for personality in CONFIG.personalities:
        images, composite = demo_single_personality(
            text=text,
            personality=personality,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            show=show,
            compile_model=compile_model,
        )
        results[personality] = (images, composite)

    # Create comparison composite (all personalities stacked)
    if output_dir and results:
        comparison = create_comparison_image(results)
        comparison_path = Path(output_dir) / "comparison.png"
        comparison.save(comparison_path, "PNG")
        print(f"\nSaved comparison: {comparison_path}")

    return results


def create_comparison_image(
    results: dict[str, tuple[list[Image.Image], Image.Image]],
    label_width: int = 150,
    row_spacing: int = 10,
    background: int = 255,
) -> Image.Image:
    """
    Create a comparison image with all personality outputs stacked.

    Args:
        results: Dict mapping personality to (images, composite)
        label_width: Width reserved for personality label
        row_spacing: Vertical spacing between rows
        background: Background grayscale value

    Returns:
        Combined comparison image
    """
    if not results:
        raise ValueError("Cannot create comparison from empty results")

    # Get dimensions from first result
    first_composite = list(results.values())[0][1]
    composite_width = first_composite.width
    composite_height = first_composite.height

    # Calculate total dimensions
    num_personalities = len(results)
    total_width = label_width + composite_width
    total_height = num_personalities * composite_height + (num_personalities - 1) * row_spacing

    # Create comparison image (RGB for text labels)
    comparison = Image.new("RGB", (total_width, total_height), (background, background, background))

    # Add each personality row
    y_offset = 0
    for personality, (_, composite) in results.items():
        # Convert composite to RGB for pasting
        composite_rgb = composite.convert("RGB")
        comparison.paste(composite_rgb, (label_width, y_offset))

        # Add text label (simple pixel-based)
        _draw_label(comparison, personality, 5, y_offset + composite_height // 2 - 6)

        y_offset += composite_height + row_spacing

    return comparison


def _draw_label(img: Image.Image, text: str, x: int, y: int) -> None:
    """
    Draw a simple text label on an image.

    Uses PIL's ImageDraw for text rendering.
    """
    try:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)

        # Try to use a better font, fall back to default
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except OSError:
                font = ImageFont.load_default()

        draw.text((x, y), text, fill=(0, 0, 0), font=font)
    except ImportError:
        # PIL without ImageDraw - skip labels
        pass


def main() -> None:
    """CLI entry point for demo application."""
    parser = argparse.ArgumentParser(
        description="Demo application for the generative font renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.demo                           # Interactive selection
  python -m src.demo --personality fruity      # Specific personality
  python -m src.demo --all                     # All personalities
  python -m src.demo --text "hello" -o outputs/
        """,
    )

    parser.add_argument(
        "--text", "-t",
        type=str,
        default=DEFAULT_TEXT,
        help=f"Text to render (default: {DEFAULT_TEXT!r})",
    )
    parser.add_argument(
        "--personality", "-p",
        type=str,
        default=None,
        choices=CONFIG.personalities,
        help=f"Personality to use: {', '.join(CONFIG.personalities)}",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate for all personalities",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=CONFIG.outputs_dir / "demo",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output images",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display images",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile() for faster inference",
    )

    args = parser.parse_args()

    # Validate text has alphabetic characters
    if not any(c.isalpha() for c in args.text):
        parser.error("Text must contain at least one alphabetic character (a-z)")

    output_dir = None if args.no_save else args.output
    show = not args.no_show

    print("=" * 60)
    print("GENERATIVE FONT RENDERER DEMO")
    print("=" * 60)
    print(f"\nText: {args.text!r}")
    print(f"Output: {output_dir or 'not saving'}")

    if args.all:
        # Generate all personalities
        demo_all_personalities(
            text=args.text,
            checkpoint_path=args.checkpoint,
            output_dir=output_dir,
            show=show,
            compile_model=args.compile,
        )
    else:
        # Single personality
        personality = args.personality
        if personality is None:
            personality = select_personality_interactive()

        demo_single_personality(
            text=args.text,
            personality=personality,
            checkpoint_path=args.checkpoint,
            output_dir=output_dir,
            show=show,
            compile_model=args.compile,
        )

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
