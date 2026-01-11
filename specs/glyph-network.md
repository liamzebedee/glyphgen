# Glyph Generation Architecture Specification

## Purpose

The Glyph Generation Network is a neural network component responsible for synthesizing 128x128 grayscale bitmap glyphs from character encodings and style embeddings. This module bridges symbolic character representations with continuous visual outputs, enabling dynamic glyph rendering for variable font styles and character sequences.

The network must operate within strict computational constraints (max 2.9M parameters, 20MB weight budget) to enable real-time inference and deployment on resource-constrained environments.

## Architecture Overview

### Input Specification

| Input | Type | Dimension | Range | Description |
|-------|------|-----------|-------|-------------|
| `prev_char` | Integer | Scalar | [0-26] | Previous character in sequence (0 = start token) |
| `curr_char` | Integer | Scalar | [1-26] | Current character to render (1-26 for A-Z) |
| `style_z` | Float Tensor | 64 | [-∞, +∞] | Style latent code (64-dimensional continuous vector) |

### Output Specification

| Output | Type | Dimension | Range | Description |
|--------|------|-----------|-------|-------------|
| `glyph_bitmap` | Float Tensor | (1, 1, 128, 128) | [0.0, 1.0] | Normalized grayscale bitmap (0 = black, 1 = white) |

### Network Components

#### 1. Embedding Layer
- **Previous character embedding**: Discrete embedding (27 tokens × 32 dims)
- **Current character embedding**: Discrete embedding (27 tokens × 32 dims)
- **Style latent**: Direct input, no embedding (64 dims)
- **Total embedded dimension**: 128 (32 + 32 + 64)

#### 2. MLP Module
- **Architecture**: Feed-forward network with residual connections (optional)
- **Hidden dimensions**: [256, 512, 256]
- **Activation**: ReLU
- **Output**: 512-dimensional feature vector
- **Purpose**: Fuse character context with style information

#### 3. Transposed Convolutional Decoder
- **Architecture**: Progressive upsampling from 8x8 to 128x128
- **Stages**:
  - Linear projection: 512 → 2048 (reshape to 8×8×32)
  - TransposeConv2d: (32, 64, kernel=4, stride=2, padding=1) → 16×16
  - TransposeConv2d: (64, 32, kernel=4, stride=2, padding=1) → 32×32
  - TransposeConv2d: (32, 16, kernel=4, stride=2, padding=1) → 64×64
  - TransposeConv2d: (16, 8, kernel=4, stride=2, padding=1) → 128×128
  - Conv2d: (8, 1, kernel=1) → output bitmap
- **Activations**: ReLU after each stage except final
- **Final activation**: Sigmoid (output range [0, 1])

## Parameter Budget Analysis

| Component | Estimated Parameters |
|-----------|----------------------|
| Prev char embedding (27 × 32) | 864 |
| Curr char embedding (27 × 32) | 864 |
| MLP: 128 → 256 | 32,896 |
| MLP: 256 → 512 | 131,584 |
| MLP: 512 → 256 | 131,328 |
| Decoder linear (256 → 2048) | 527,104 |
| TransposeConv stage 1 (32 → 64) | 32,832 |
| TransposeConv stage 2 (64 → 32) | 32,800 |
| TransposeConv stage 3 (32 → 16) | 8,208 |
| TransposeConv stage 4 (16 → 8) | 2,056 |
| Conv2d final (8 → 1) | 73 |
| **Total** | **~900K parameters** |

**Status**: Well within 2.9M parameter budget (31% utilization). Allows headroom for batch normalization, additional layers, or training-time regularization.

## Acceptance Criteria

### Behavioral Requirements

1. **Input Validation**
   - [ ] Network accepts `prev_char` values in range [0-26] without error
   - [ ] Network accepts `curr_char` values in range [1-26] without error
   - [ ] Network accepts `style_z` as 64-dimensional tensor
   - [ ] Rejects invalid inputs with descriptive error messages

2. **Output Quality**
   - [ ] Output shape is exactly (1, 1, 128, 128) for single sample input
   - [ ] Output values are bounded to [0.0, 1.0] range
   - [ ] Output bitmap is non-zero (contains readable glyph, not empty)
   - [ ] Output exhibits smooth anti-aliasing (no harsh pixel artifacts)

3. **Style Conditioning**
   - [ ] Different `style_z` values produce visually distinct glyphs
   - [ ] Same character with different styles shows clear variation
   - [ ] Style changes preserve character legibility

4. **Character Context**
   - [ ] Changing `prev_char` may produce subtle stylistic variations (for ligature/kerning support)
   - [ ] Changing `curr_char` produces distinctly different glyphs
   - [ ] All 26 characters (curr_char ∈ [1-26]) produce recognizable outputs

5. **Computational Performance**
   - [ ] Single forward pass completes in <100ms on CPU
   - [ ] Single forward pass completes in <10ms on GPU (NVIDIA V100 baseline)
   - [ ] Memory footprint ≤20MB for weights
   - [ ] Total parameters ≤2.9M

6. **Consistency**
   - [ ] Deterministic output given frozen weights and deterministic inputs
   - [ ] Batch processing produces identical per-sample outputs as individual processing
   - [ ] No internal randomness in inference mode

### Testable Criteria

```python
# Test harness pseudocode
def test_glyph_network():
    model = GlyphNetwork()

    # Test 1: Input validation
    assert model(prev_char=0, curr_char=1, style_z=torch.randn(64)).shape == (1, 1, 128, 128)
    assert model(prev_char=26, curr_char=26, style_z=torch.randn(64)).shape == (1, 1, 128, 128)

    # Test 2: Output bounds
    output = model(prev_char=0, curr_char=1, style_z=torch.randn(64))
    assert output.min() >= 0.0 and output.max() <= 1.0

    # Test 3: Style variation
    z1 = torch.randn(64)
    z2 = torch.randn(64)
    out1 = model(prev_char=0, curr_char=1, style_z=z1)
    out2 = model(prev_char=0, curr_char=1, style_z=z2)
    assert not torch.allclose(out1, out2)  # Different styles should differ

    # Test 4: Character variation
    out_a = model(prev_char=0, curr_char=1, style_z=torch.randn(64))
    out_b = model(prev_char=0, curr_char=2, style_z=torch.randn(64))
    assert not torch.allclose(out_a, out_b)  # Different chars should differ

    # Test 5: Performance
    import time
    start = time.time()
    for _ in range(100):
        model(prev_char=0, curr_char=1, style_z=torch.randn(64))
    elapsed = (time.time() - start) / 100
    assert elapsed < 0.1  # <100ms per forward pass on CPU
```

## Edge Cases

### Input Edge Cases

1. **Start Token (prev_char = 0)**
   - Network must handle sequence start gracefully
   - No prior character context available
   - Should produce glyphs identical to isolated character rendering
   - **Mitigation**: Embedding layer includes dedicated start token

2. **Style Vector Extremes**
   - Very large L2 norm in `style_z` (e.g., ∥z∥ >> 1)
   - All-zeros style vector
   - Style vector with extreme individual dimensions
   - **Mitigation**: Normalize style input during training; add gradient clipping

3. **Same Character Repetition (prev_char = curr_char)**
   - Network may optimize for common case (consecutive identical characters)
   - Risk of producing identical outputs for sequence vs. isolated character
   - **Mitigation**: Test that isolated 'A' and 'AA' middle produce similar but not identical outputs

4. **Character Boundary Conditions**
   - curr_char = 1 (first character, 'A')
   - curr_char = 26 (last character, 'Z')
   - prev_char = 0 (sequence start)
   - prev_char = 26 (last character as predecessor)
   - **Mitigation**: Ensure embedding lookup handles all valid indices

### Output Edge Cases

1. **Blank Glyphs**
   - Network may produce near-zero bitmaps (empty glyphs)
   - Can occur with untrained style vectors or pathological inputs
   - **Mitigation**: Validation loss prevents collapse; warn if max pixel value < 0.1

2. **Saturation**
   - Output fully saturated (all pixels ≈ 1.0, completely white)
   - Loss of glyph definition
   - **Mitigation**: Regularization on output variance; sigmoid prevents unbounded values

3. **Spatial Artifacts**
   - Checker-board patterns from transposed convolution
   - Severe banding or aliasing
   - **Mitigation**: Use appropriate kernel sizes and strides; apply smoothing loss

4. **Out-of-Bounds Memory Access**
   - prev_char or curr_char > 26 or < valid range
   - **Mitigation**: Assert/clamp inputs before embedding lookup

### Inference Edge Cases

1. **Batch Processing Mismatch**
   - Batch size > 1 may produce inconsistent results due to batch norm
   - **Mitigation**: Use LayerNorm instead of BatchNorm, or ensure eval mode behavior

2. **Hardware Precision**
   - Float16 inference may cause numerical instability
   - **Mitigation**: Provide float32 baseline; document precision requirements

3. **Sequential Dependency Misuse**
   - Caller chains outputs without using prev_char correctly
   - **Mitigation**: Document API clearly; add type hints

## Dependencies

### Required Dependencies

- **PyTorch** (≥1.9.0): Neural network framework
- **NumPy** (≥1.19.0): Numerical computations
- **Pillow** (≥8.0.0): Glyph bitmap I/O and validation

### Optional Dependencies

- **CUDA Toolkit** (≥11.0): GPU acceleration (NVIDIA)
- **TorchScript**: Model serialization and deployment
- **ONNX Runtime**: Cross-platform inference optimization

### Module Dependencies (Internal)

- Character encoding system: Must support 26-letter alphabet (A-Z)
- Style VAE/latent space: Produces 64-dimensional style vectors
- Training data pipeline: Character image dataset with style labels
- Visualization utilities: For glyph rendering and debugging

### Hardware Requirements

**Minimum**:
- CPU: 2+ cores, 2GB RAM (inference only)
- 20MB storage for weights

**Recommended**:
- GPU: NVIDIA GPU with 2GB+ VRAM (inference)
- CPU: 4+ cores for training
- 8GB+ RAM for batch training

### File Dependencies

- `glyph_network.py`: Main architecture implementation
- `config.yaml`: Hyperparameter configuration
- `checkpoint.pt`: Trained weights (20MB max)
- `test_glyph_network.py`: Unit tests
- `character_embedding.py`: Character encoding logic

## Implementation Notes

### Memory Layout
- All tensors use row-major (C-style) memory layout
- Grayscale bitmap stored as single-channel float32
- Embeddings stored as float32

### Numerical Stability
- Use LayerNorm to avoid batch statistics dependence
- Clamp style vector L2 norm during inference if needed
- Sigmoid output guarantees bounded range

### Deployment Checklist
- [ ] Model exported to TorchScript
- [ ] Quantization tested (int8 if required)
- [ ] Inference latency profiled on target hardware
- [ ] Edge cases validated
- [ ] Documentation generated

## Version History

- **v1.0** (2026-01-11): Initial specification with embedding→MLP→decoder architecture

