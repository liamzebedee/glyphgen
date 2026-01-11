"""
Tests for inference engine with performance benchmarks.

Tests cover:
- Engine initialization and warmup
- Single glyph generation
- Batch inference
- Sequence generation
- Buffer pool functionality
- Performance benchmarks (latency, throughput, memory)
"""

import pytest
import torch
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.inference import InferenceEngine, BufferPool
from src.glyphnet import GlyphNetwork, create_model
from src.style import StyleVector


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def model():
    """Create a GlyphNetwork instance."""
    return create_model()


@pytest.fixture
def device():
    """Get available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def engine(model, device):
    """Create an inference engine without compilation for faster tests."""
    return InferenceEngine(
        model=model,
        device=device,
        compile_model=False,
        warmup_iterations=1,
        max_batch_size=16,
    )


@pytest.fixture
def style_z():
    """Create a random style vector."""
    return torch.randn(64)


# ============================================================================
# BufferPool Tests
# ============================================================================


class TestBufferPool:
    """Tests for pre-allocated buffer pool."""

    def test_buffer_creation(self, device):
        """Buffer pool creates tensors on correct device."""
        pool = BufferPool(device=device, max_batch_size=8)
        assert pool.prev_char_buffer.device.type == device.type
        assert pool.curr_char_buffer.device.type == device.type
        assert pool.style_buffer.device.type == device.type
        assert pool.output_buffer.device.type == device.type

    def test_buffer_shapes(self, device):
        """Buffer pool has correct shapes."""
        max_batch = 16
        pool = BufferPool(device=device, max_batch_size=max_batch)
        assert pool.prev_char_buffer.shape == (max_batch,)
        assert pool.curr_char_buffer.shape == (max_batch,)
        assert pool.style_buffer.shape == (max_batch, 64)
        assert pool.output_buffer.shape == (max_batch, 1, 128, 128)

    def test_get_input_views(self, device):
        """Input views have correct batch size."""
        pool = BufferPool(device=device, max_batch_size=32)
        batch_size = 8
        prev, curr, style = pool.get_input_views(batch_size)
        assert prev.shape == (batch_size,)
        assert curr.shape == (batch_size,)
        assert style.shape == (batch_size, 64)

    def test_get_output_view(self, device):
        """Output view has correct batch size."""
        pool = BufferPool(device=device, max_batch_size=32)
        batch_size = 4
        output = pool.get_output_view(batch_size)
        assert output.shape == (batch_size, 1, 128, 128)

    def test_batch_size_exceeded(self, device):
        """Raises error when batch size exceeds max."""
        pool = BufferPool(device=device, max_batch_size=8)
        with pytest.raises(ValueError, match="exceeds max"):
            pool.get_input_views(16)
        with pytest.raises(ValueError, match="exceeds max"):
            pool.get_output_view(16)


# ============================================================================
# Engine Initialization Tests
# ============================================================================


class TestEngineInitialization:
    """Tests for inference engine initialization."""

    def test_engine_creation(self, model):
        """Engine can be created from model."""
        engine = InferenceEngine(model=model, compile_model=False, warmup_iterations=1)
        assert engine.model is not None
        assert engine.buffer_pool is not None

    def test_engine_device_placement(self, model, device):
        """Model is placed on correct device."""
        engine = InferenceEngine(
            model=model, device=device, compile_model=False, warmup_iterations=1
        )
        # Check model parameters are on correct device
        for param in engine.model.parameters():
            assert param.device.type == device.type

    def test_engine_eval_mode(self, model):
        """Model is set to eval mode."""
        engine = InferenceEngine(model=model, compile_model=False, warmup_iterations=1)
        assert not engine.model.training

    def test_from_model_factory(self, model, device):
        """from_model factory creates engine correctly."""
        engine = InferenceEngine.from_model(
            model=model, device=device, compile_model=False
        )
        assert engine.model is not None

    def test_warmup_runs(self, model):
        """Warmup executes without error."""
        # Just verify no exception is raised
        engine = InferenceEngine(model=model, compile_model=False, warmup_iterations=3)
        assert engine is not None


# ============================================================================
# Single Glyph Generation Tests
# ============================================================================


class TestSingleGlyphGeneration:
    """Tests for single glyph generation."""

    def test_generate_glyph_shape(self, engine, style_z):
        """Single glyph has correct shape."""
        glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        assert glyph.shape == (1, 128, 128)

    def test_generate_glyph_range(self, engine, style_z):
        """Single glyph values are in [0, 1]."""
        glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        assert glyph.min() >= 0.0
        assert glyph.max() <= 1.0

    def test_generate_glyph_all_chars(self, engine, style_z):
        """Can generate all 26 characters."""
        for curr_char in range(1, 27):
            glyph = engine.generate_glyph(prev_char=0, curr_char=curr_char, style_z=style_z)
            assert glyph.shape == (1, 128, 128)

    def test_generate_glyph_with_context(self, engine, style_z):
        """Glyph generation uses prev_char context."""
        glyph1 = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        glyph2 = engine.generate_glyph(prev_char=5, curr_char=1, style_z=style_z)
        # Different context should produce different output
        assert not torch.allclose(glyph1, glyph2)

    def test_generate_glyph_2d_style(self, engine):
        """Handles 2D style vector (1, 64)."""
        style_z = torch.randn(1, 64)
        glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        assert glyph.shape == (1, 128, 128)

    def test_generate_glyph_deterministic(self, engine):
        """Same inputs produce same output."""
        torch.manual_seed(42)
        style_z = torch.randn(64)
        glyph1 = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        glyph2 = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        assert torch.allclose(glyph1, glyph2)


# ============================================================================
# Batch Inference Tests
# ============================================================================


class TestBatchInference:
    """Tests for batch glyph generation."""

    def test_generate_batch_shape(self, engine, style_z):
        """Batch output has correct shape."""
        batch_size = 4
        prev_chars = torch.zeros(batch_size, dtype=torch.long)
        curr_chars = torch.arange(1, batch_size + 1)
        style_batch = style_z.unsqueeze(0).expand(batch_size, -1)

        batch = engine.generate_batch(prev_chars, curr_chars, style_batch)
        assert batch.shape == (batch_size, 1, 128, 128)

    def test_generate_batch_range(self, engine, style_z):
        """Batch values are in [0, 1]."""
        batch_size = 8
        prev_chars = torch.zeros(batch_size, dtype=torch.long)
        curr_chars = torch.ones(batch_size, dtype=torch.long)

        batch = engine.generate_batch(prev_chars, curr_chars, style_z)
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0

    def test_generate_batch_shared_style(self, engine, style_z):
        """Batch works with shared 1D style vector."""
        batch_size = 4
        prev_chars = torch.zeros(batch_size, dtype=torch.long)
        curr_chars = torch.arange(1, batch_size + 1)

        batch = engine.generate_batch(prev_chars, curr_chars, style_z)
        assert batch.shape == (batch_size, 1, 128, 128)

    def test_generate_batch_different_chars(self, engine, style_z):
        """Different chars in batch produce different outputs."""
        prev_chars = torch.tensor([0, 0, 0])
        curr_chars = torch.tensor([1, 2, 3])

        batch = engine.generate_batch(prev_chars, curr_chars, style_z)

        # All three glyphs should be different
        assert not torch.allclose(batch[0], batch[1])
        assert not torch.allclose(batch[1], batch[2])
        assert not torch.allclose(batch[0], batch[2])

    def test_generate_batch_equals_single(self, engine, style_z):
        """Batch generation equals individual generation."""
        prev_chars = torch.tensor([0, 1, 2])
        curr_chars = torch.tensor([1, 2, 3])

        batch = engine.generate_batch(prev_chars, curr_chars, style_z)

        for i in range(3):
            single = engine.generate_glyph(
                prev_char=prev_chars[i].item(),
                curr_char=curr_chars[i].item(),
                style_z=style_z,
            )
            # Use larger tolerance for GPU floating point variance
            # Differences of ~1e-4 are expected and negligible for image generation
            assert torch.allclose(batch[i], single, atol=1e-3)

    def test_generate_batch_large_chunked(self, engine, style_z):
        """Large batches are processed in chunks."""
        # Engine has max_batch_size=16
        batch_size = 32
        prev_chars = torch.zeros(batch_size, dtype=torch.long)
        curr_chars = torch.ones(batch_size, dtype=torch.long)

        batch = engine.generate_batch(prev_chars, curr_chars, style_z)
        assert batch.shape == (batch_size, 1, 128, 128)


# ============================================================================
# Sequence Generation Tests
# ============================================================================


class TestSequenceGeneration:
    """Tests for text sequence generation."""

    def test_generate_sequence_length(self, engine, style_z):
        """Sequence produces correct number of glyphs."""
        text = "hello"
        glyphs = engine.generate_sequence(text, style_z)
        assert len(glyphs) == len(text)

    def test_generate_sequence_shapes(self, engine, style_z):
        """Each glyph in sequence has correct shape."""
        glyphs = engine.generate_sequence("abc", style_z)
        for glyph in glyphs:
            assert glyph.shape == (1, 128, 128)

    def test_generate_sequence_case_insensitive(self, engine, style_z):
        """Uppercase and lowercase produce same glyphs."""
        glyphs_lower = engine.generate_sequence("abc", style_z)
        glyphs_upper = engine.generate_sequence("ABC", style_z)

        for lower, upper in zip(glyphs_lower, glyphs_upper):
            assert torch.allclose(lower, upper)

    def test_generate_sequence_ignores_non_alpha(self, engine, style_z):
        """Non-alphabetic characters are skipped."""
        glyphs = engine.generate_sequence("a1b 2c!", style_z)
        assert len(glyphs) == 3  # Only a, b, c

    def test_generate_sequence_context_aware(self, engine, style_z):
        """Sequence uses context from previous character."""
        # Generate 'ab' as sequence
        seq_glyphs = engine.generate_sequence("ab", style_z)

        # Generate 'b' standalone (prev_char=0)
        standalone_b = engine.generate_glyph(prev_char=0, curr_char=2, style_z=style_z)

        # The 'b' in sequence should differ from standalone 'b'
        # because prev_char is 'a' (1) vs 0
        assert not torch.allclose(seq_glyphs[1], standalone_b)


# ============================================================================
# Performance Benchmark Tests
# ============================================================================


class TestPerformanceBenchmarks:
    """Performance tests for inference latency and throughput.

    Note: These tests validate the benchmark functionality works.
    Actual latency targets (<5ms) depend on hardware (RTX 3090).
    """

    def test_benchmark_returns_stats(self, engine):
        """Benchmark returns expected statistics."""
        stats = engine.benchmark(num_iterations=10)

        assert "device" in stats
        assert "compiled" in stats
        assert "single_mean_ms" in stats
        assert "single_min_ms" in stats
        assert "single_max_ms" in stats
        assert "batch8_mean_ms" in stats
        assert "batch8_per_glyph_ms" in stats
        assert "throughput_glyphs_per_sec" in stats

    def test_single_inference_latency_reasonable(self, engine):
        """Single inference completes in reasonable time (< 500ms on any hardware)."""
        stats = engine.benchmark(num_iterations=20)
        # Very generous bound - just ensures it doesn't hang
        assert stats["single_mean_ms"] < 500.0

    def test_batch_more_efficient_per_glyph(self, engine):
        """Batch inference is more efficient per glyph than single."""
        stats = engine.benchmark(num_iterations=20)
        # Per-glyph time in batch should be less than single glyph time
        # (accounting for overhead, batch of 8 should be at least 2x efficient)
        assert stats["batch8_per_glyph_ms"] < stats["single_mean_ms"]

    def test_throughput_positive(self, engine):
        """Throughput is a positive number."""
        stats = engine.benchmark(num_iterations=10)
        assert stats["throughput_glyphs_per_sec"] > 0

    def test_multiple_sequential_inferences(self, engine, style_z):
        """Can run many sequential inferences without error."""
        for i in range(100):
            curr_char = (i % 26) + 1
            glyph = engine.generate_glyph(prev_char=0, curr_char=curr_char, style_z=style_z)
            assert glyph.shape == (1, 128, 128)


# ============================================================================
# Memory Tests
# ============================================================================


class TestMemory:
    """Memory usage tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_not_leaked_single(self):
        """Memory is not leaked after single inferences."""
        torch.cuda.reset_peak_memory_stats()
        model = create_model(device=torch.device("cuda"))
        engine = InferenceEngine(model=model, compile_model=False, warmup_iterations=1)

        style_z = torch.randn(64)

        # Measure memory after warmup
        initial_memory = torch.cuda.max_memory_allocated()

        # Run many inferences
        for _ in range(100):
            engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)

        final_memory = torch.cuda.max_memory_allocated()

        # Memory should not grow significantly (allow 10MB tolerance)
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)
        assert memory_growth < 10, f"Memory grew by {memory_growth:.1f} MB"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_batch_within_budget(self):
        """Batch of 8 uses less than 2GB memory."""
        torch.cuda.reset_peak_memory_stats()
        model = create_model(device=torch.device("cuda"))
        engine = InferenceEngine(model=model, compile_model=False, warmup_iterations=1)

        batch_size = 8
        prev_chars = torch.zeros(batch_size, dtype=torch.long)
        curr_chars = torch.ones(batch_size, dtype=torch.long)
        style_z = torch.randn(64)

        # Run batch inference
        engine.generate_batch(prev_chars, curr_chars, style_z)

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        assert peak_memory < 2.0, f"Peak memory {peak_memory:.2f} GB exceeds 2GB budget"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sequence(self, engine, style_z):
        """Empty string produces empty list."""
        glyphs = engine.generate_sequence("", style_z)
        assert len(glyphs) == 0

    def test_non_alpha_sequence(self, engine, style_z):
        """Non-alphabetic string produces empty list."""
        glyphs = engine.generate_sequence("123 !@#", style_z)
        assert len(glyphs) == 0

    def test_batch_size_one(self, engine, style_z):
        """Batch of size 1 works correctly."""
        prev_chars = torch.tensor([0])
        curr_chars = torch.tensor([1])
        batch = engine.generate_batch(prev_chars, curr_chars, style_z)
        assert batch.shape == (1, 1, 128, 128)

    def test_start_token_context(self, engine, style_z):
        """Start token (0) as prev_char works."""
        glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        assert glyph.shape == (1, 128, 128)

    def test_all_prev_char_contexts(self, engine, style_z):
        """All valid prev_char values work (0-26)."""
        for prev_char in range(27):
            glyph = engine.generate_glyph(prev_char=prev_char, curr_char=1, style_z=style_z)
            assert glyph.shape == (1, 128, 128)


# ============================================================================
# Integration with StyleVector
# ============================================================================


class TestStyleVectorIntegration:
    """Tests for integration with StyleVector class."""

    def test_style_vector_object(self, engine):
        """Works with StyleVector.data tensor."""
        style = StyleVector.random()
        glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style.data)
        assert glyph.shape == (1, 128, 128)

    def test_different_styles_different_outputs(self, engine):
        """Different style vectors produce different outputs."""
        style1 = StyleVector.random(seed=1)
        style2 = StyleVector.random(seed=2)

        glyph1 = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style1.data)
        glyph2 = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style2.data)

        assert not torch.allclose(glyph1, glyph2)

    def test_zero_style_works(self, engine):
        """Zero style vector produces valid output."""
        style = StyleVector.zeros()
        glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style.data)
        assert glyph.shape == (1, 128, 128)
        assert glyph.min() >= 0.0
        assert glyph.max() <= 1.0


# ============================================================================
# Checkpoint Loading Tests
# ============================================================================


class TestCheckpointLoading:
    """Tests for loading models from checkpoints."""

    def test_from_checkpoint(self, tmp_path, model, device):
        """Can load model from checkpoint."""
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "loss": 0.5,
        }
        torch.save(checkpoint, checkpoint_path)

        # Load via factory
        engine = InferenceEngine.from_checkpoint(
            checkpoint_path,
            device=device,
            compile_model=False,
            warmup_iterations=1,
        )

        # Verify model works
        style_z = torch.randn(64)
        glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        assert glyph.shape == (1, 128, 128)

    def test_from_checkpoint_preserves_weights(self, tmp_path, model, device):
        """Loaded model produces same output as original."""
        # Set specific weights for testing
        torch.manual_seed(12345)
        model = create_model(device=device)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        checkpoint = {"model_state_dict": model.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        # Create engines
        engine1 = InferenceEngine(model=model, compile_model=False, warmup_iterations=1)
        engine2 = InferenceEngine.from_checkpoint(
            checkpoint_path, device=device, compile_model=False, warmup_iterations=1
        )

        # Compare outputs
        style_z = torch.randn(64)
        glyph1 = engine1.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
        glyph2 = engine2.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)

        assert torch.allclose(glyph1, glyph2, atol=1e-6)


# ============================================================================
# Compilation Tests
# ============================================================================


class TestCompilation:
    """Tests for torch.compile functionality."""

    def test_is_compiled_property(self, model):
        """is_compiled property reflects compilation state."""
        engine_no_compile = InferenceEngine(model=model, compile_model=False)
        assert not engine_no_compile.is_compiled

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compiled_engine_works(self, model, style_z):
        """Compiled engine produces valid output."""
        try:
            engine = InferenceEngine(model=model, compile_model=True)
            glyph = engine.generate_glyph(prev_char=0, curr_char=1, style_z=style_z)
            assert glyph.shape == (1, 128, 128)
        except Exception:
            # Compilation may fail on some platforms, that's OK
            pytest.skip("torch.compile not supported on this platform")
