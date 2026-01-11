"""
Inference engine for GlyphNetwork with GPU optimization.

Features:
- CUDA device placement with optimal settings
- torch.compile() for graph optimization (PyTorch 2.0+)
- Pre-allocated buffer pool for reduced memory allocation overhead
- Warmup inference on model load for JIT compilation
- Single and batch inference APIs

Usage:
    engine = InferenceEngine.from_checkpoint("outputs/final_checkpoint.pt")
    glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_vector)
    batch = engine.generate_batch(prev_chars, curr_chars, style_vectors)
"""

import torch
from typing import Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from src.glyphnet import GlyphNetwork, create_model
from src.utils import get_device


@dataclass
class BufferPool:
    """Pre-allocated tensor buffers for inference to reduce allocation overhead."""

    device: torch.device
    max_batch_size: int = 32

    def __post_init__(self):
        """Pre-allocate buffers on device."""
        # Input buffers
        self.prev_char_buffer = torch.zeros(
            self.max_batch_size, dtype=torch.long, device=self.device
        )
        self.curr_char_buffer = torch.zeros(
            self.max_batch_size, dtype=torch.long, device=self.device
        )
        self.style_buffer = torch.zeros(
            self.max_batch_size, 64, dtype=torch.float32, device=self.device
        )
        # Output buffer (128x128 grayscale)
        self.output_buffer = torch.zeros(
            self.max_batch_size, 1, 128, 128, dtype=torch.float32, device=self.device
        )

    def get_input_views(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get views into input buffers for given batch size."""
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max {self.max_batch_size}"
            )
        return (
            self.prev_char_buffer[:batch_size],
            self.curr_char_buffer[:batch_size],
            self.style_buffer[:batch_size],
        )

    def get_output_view(self, batch_size: int) -> torch.Tensor:
        """Get view into output buffer for given batch size."""
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max {self.max_batch_size}"
            )
        return self.output_buffer[:batch_size]


class InferenceEngine:
    """
    Optimized inference engine for GlyphNetwork.

    Features:
    - Automatic device selection (CUDA > MPS > CPU)
    - Optional torch.compile() for graph optimization
    - Pre-allocated buffer pool for reduced allocation overhead
    - Warmup passes for JIT compilation
    """

    def __init__(
        self,
        model: GlyphNetwork,
        device: Optional[torch.device] = None,
        compile_model: bool = True,
        warmup_iterations: int = 5,
        max_batch_size: int = 32,
    ):
        """
        Initialize inference engine.

        Args:
            model: GlyphNetwork instance
            device: Target device (auto-detected if None)
            compile_model: Whether to use torch.compile()
            warmup_iterations: Number of warmup forward passes
            max_batch_size: Maximum batch size for buffer pool
        """
        self.device = device or get_device()
        self.compile_model = compile_model
        self.warmup_iterations = warmup_iterations
        self.max_batch_size = max_batch_size

        # Move model to device
        self.model = model.to(self.device)
        self.model.eval()

        # Apply torch.compile() if available and requested
        self._compiled = False
        if compile_model and hasattr(torch, "compile"):
            try:
                # Use reduce-overhead mode for inference
                self.model = torch.compile(  # type: ignore[assignment]
                    self.model, mode="reduce-overhead", fullgraph=False
                )
                self._compiled = True
            except Exception:
                # Fall back to uncompiled if compilation fails
                # (e.g., unsupported ops, older GPU)
                pass

        # Create buffer pool
        self.buffer_pool = BufferPool(device=self.device, max_batch_size=max_batch_size)

        # Run warmup
        self._warmup()

    def _warmup(self) -> None:
        """Run warmup inference passes to trigger JIT compilation."""
        # Use representative inputs for warmup
        with torch.no_grad():
            for batch_size in [1, 8, self.max_batch_size]:
                prev_chars = torch.zeros(
                    batch_size, dtype=torch.long, device=self.device
                )
                curr_chars = torch.ones(
                    batch_size, dtype=torch.long, device=self.device
                )
                style_z = torch.zeros(batch_size, 64, device=self.device)

                for _ in range(self.warmup_iterations):
                    self.model(prev_chars, curr_chars, style_z)

        # Synchronize CUDA to ensure warmup is complete
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: Optional[torch.device] = None,
        compile_model: bool = True,
        warmup_iterations: int = 5,
        max_batch_size: int = 32,
    ) -> "InferenceEngine":
        """
        Create InferenceEngine from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Target device (auto-detected if None)
            compile_model: Whether to use torch.compile()
            warmup_iterations: Number of warmup forward passes
            max_batch_size: Maximum batch size for buffer pool

        Returns:
            Initialized InferenceEngine
        """
        device = device or get_device()
        model = create_model(device=device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(
            model=model,
            device=device,
            compile_model=compile_model,
            warmup_iterations=warmup_iterations,
            max_batch_size=max_batch_size,
        )

    @classmethod
    def from_model(
        cls,
        model: GlyphNetwork,
        device: Optional[torch.device] = None,
        compile_model: bool = True,
        warmup_iterations: int = 5,
        max_batch_size: int = 32,
    ) -> "InferenceEngine":
        """
        Create InferenceEngine from an existing model.

        Args:
            model: GlyphNetwork instance
            device: Target device (auto-detected if None)
            compile_model: Whether to use torch.compile()
            warmup_iterations: Number of warmup forward passes
            max_batch_size: Maximum batch size for buffer pool

        Returns:
            Initialized InferenceEngine
        """
        return cls(
            model=model,
            device=device,
            compile_model=compile_model,
            warmup_iterations=warmup_iterations,
            max_batch_size=max_batch_size,
        )

    @torch.no_grad()
    def generate_glyph(
        self,
        prev_char: int,
        curr_char: int,
        style_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a single glyph.

        Args:
            prev_char: Previous character index [0-26]
            curr_char: Current character index [1-26]
            style_z: Style vector, shape (64,) or (1, 64)

        Returns:
            Glyph bitmap, shape (1, 128, 128), values in [0, 1]
        """
        # Prepare inputs using buffer pool
        prev_view, curr_view, style_view = self.buffer_pool.get_input_views(1)
        prev_view[0] = prev_char
        curr_view[0] = curr_char

        # Handle style vector shape
        if style_z.dim() == 1:
            style_view[0] = style_z.to(self.device)
        else:
            style_view[0] = style_z[0].to(self.device)

        # Run inference
        output = self.model(prev_view, curr_view, style_view)

        # Return copy to avoid buffer reuse issues
        return output[0].clone()

    @torch.no_grad()
    def generate_batch(
        self,
        prev_chars: torch.Tensor,
        curr_chars: torch.Tensor,
        style_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a batch of glyphs.

        Args:
            prev_chars: Previous character indices, shape (B,)
            curr_chars: Current character indices, shape (B,)
            style_z: Style vectors, shape (B, 64) or (64,) for shared style

        Returns:
            Glyph bitmaps, shape (B, 1, 128, 128), values in [0, 1]
        """
        batch_size = prev_chars.size(0)

        # Validate batch size
        if batch_size > self.max_batch_size:
            # Process in chunks
            return self._generate_chunked(prev_chars, curr_chars, style_z)

        # Prepare inputs
        prev_chars = prev_chars.to(self.device)
        curr_chars = curr_chars.to(self.device)

        # Handle shared style vector
        if style_z.dim() == 1:
            style_z = style_z.unsqueeze(0).expand(batch_size, -1)
        style_z = style_z.to(self.device)

        # Run inference
        output = self.model(prev_chars, curr_chars, style_z)

        return output.clone()

    def _generate_chunked(
        self,
        prev_chars: torch.Tensor,
        curr_chars: torch.Tensor,
        style_z: torch.Tensor,
    ) -> torch.Tensor:
        """Generate glyphs in chunks for large batches."""
        batch_size = prev_chars.size(0)
        results = []

        for i in range(0, batch_size, self.max_batch_size):
            end_idx = min(i + self.max_batch_size, batch_size)
            chunk_prev = prev_chars[i:end_idx]
            chunk_curr = curr_chars[i:end_idx]

            # Handle style vector slicing
            if style_z.dim() == 1:
                chunk_style = style_z
            else:
                chunk_style = style_z[i:end_idx]

            chunk_output = self.generate_batch(chunk_prev, chunk_curr, chunk_style)
            results.append(chunk_output)

        return torch.cat(results, dim=0)

    @torch.no_grad()
    def generate_sequence(
        self,
        text: str,
        style_z: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Generate glyphs for a text sequence.

        Args:
            text: Text to render (a-z only, lowercase)
            style_z: Style vector, shape (64,)

        Returns:
            List of glyph bitmaps, each shape (1, 128, 128)
        """
        glyphs = []
        prev_char = 0  # Start token

        for char in text.lower():
            if char.isalpha():
                curr_char = ord(char) - ord("a") + 1  # a=1, b=2, ..., z=26
                glyph = self.generate_glyph(prev_char, curr_char, style_z)
                glyphs.append(glyph)
                prev_char = curr_char

        return glyphs

    @property
    def is_compiled(self) -> bool:
        """Whether the model was successfully compiled."""
        return self._compiled

    def benchmark(self, num_iterations: int = 100) -> dict:
        """
        Benchmark inference performance.

        Args:
            num_iterations: Number of iterations for timing

        Returns:
            Dict with timing statistics
        """
        import time

        # Prepare inputs
        prev_char = torch.tensor([0], dtype=torch.long, device=self.device)
        curr_char = torch.tensor([1], dtype=torch.long, device=self.device)
        style_z = torch.zeros(1, 64, device=self.device)

        # Warmup
        for _ in range(10):
            self.model(prev_char, curr_char, style_z)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark single inference
        single_times = []
        for _ in range(num_iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            self.model(prev_char, curr_char, style_z)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            single_times.append((time.perf_counter() - start) * 1000)

        # Benchmark batch inference
        batch_size = 8
        prev_batch = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        curr_batch = torch.ones(batch_size, dtype=torch.long, device=self.device)
        style_batch = torch.zeros(batch_size, 64, device=self.device)

        batch_times = []
        for _ in range(num_iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            self.model(prev_batch, curr_batch, style_batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - start) * 1000)

        return {
            "device": str(self.device),
            "compiled": self._compiled,
            "single_mean_ms": sum(single_times) / len(single_times),
            "single_min_ms": min(single_times),
            "single_max_ms": max(single_times),
            "batch8_mean_ms": sum(batch_times) / len(batch_times),
            "batch8_min_ms": min(batch_times),
            "batch8_max_ms": max(batch_times),
            "batch8_per_glyph_ms": sum(batch_times) / len(batch_times) / batch_size,
            "throughput_glyphs_per_sec": 1000.0 / (sum(single_times) / len(single_times)),
        }


if __name__ == "__main__":
    # Quick demonstration
    print("Creating inference engine...")
    model = create_model()
    engine = InferenceEngine(model, compile_model=False)  # Skip compile for quick test

    print(f"Device: {engine.device}")
    print(f"Compiled: {engine.is_compiled}")

    # Generate single glyph
    style_z = torch.randn(64)
    glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
    print(f"Single glyph shape: {glyph.shape}")
    print(f"Single glyph range: [{glyph.min():.4f}, {glyph.max():.4f}]")

    # Generate batch
    prev_chars = torch.tensor([0, 1, 2])
    curr_chars = torch.tensor([1, 2, 3])
    batch = engine.generate_batch(prev_chars, curr_chars, style_z)
    print(f"Batch shape: {batch.shape}")

    # Generate sequence
    glyphs = engine.generate_sequence("hello", style_z)
    print(f"Sequence 'hello' produced {len(glyphs)} glyphs")

    # Benchmark
    print("\nBenchmarking...")
    stats = engine.benchmark(num_iterations=50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
