# Training Data Generation Specification

## Purpose

Generate a synthetic MNIST-style character dataset from the Open Sans font for training character recognition models. The module produces augmented character images with contextual metadata about previous characters, enabling the model to learn character prediction in sequence context.

The generated dataset will contain:
- 14,040 training samples (26 characters × 27 previous character contexts × 20 augmentations each)
- 128×128 grayscale images normalized to [0,1]
- Metadata linking images to current and previous characters
- PyTorch-compatible tensor format for easy model integration

## Acceptance Criteria

### Dataset Composition
- [ ] Generate exactly 26 character classes (a-z)
- [ ] Create 27 context variations for each character (26 previous characters + 1 start-of-sequence token)
- [ ] Apply 20 distinct augmentation variations to each base character-context pair
- [ ] Total samples generated: 14,040 (26 × 27 × 20)
- [ ] All samples must be traceable to source character and previous context

### Image Generation
- [ ] All generated images are 128×128 pixels
- [ ] All images are single-channel grayscale
- [ ] Font source is Open Sans (must be available or bundled)
- [ ] Character rendering is readable and distinguishable at 128×128 resolution
- [ ] Images are normalized to float32 values in range [0.0, 1.0]

### Augmentation Implementation
- [ ] **Rotation**: Apply rotation angles in range [-15°, 15°], minimum 20 variations
- [ ] **Scale**: Apply scale factors in range [0.8, 1.2], minimum 20 variations
- [ ] **Translation**: Apply pixel offsets in range [-5, 5] in both axes, minimum 20 variations
- [ ] **Blur**: Apply Gaussian blur with kernel sizes [1, 3, 5], minimum 20 variations
- [ ] **Noise**: Apply Gaussian noise with std in range [0.01, 0.1], minimum 20 variations
- [ ] All augmentations preserve image dimensions (128×128)
- [ ] Augmentation parameters are deterministically reproducible from seed
- [ ] Each augmentation type must appear in at least one variation of each character-context pair

### Data Format & Storage
- [ ] Output saved as single PyTorch .pt file (binary format)
- [ ] File contains three keys: `images`, `prev_chars`, `curr_chars`
  - `images`: torch.Tensor of shape (14040, 1, 128, 128) with dtype float32
  - `prev_chars`: torch.Tensor of shape (14040,) containing character indices 0-26 (0=start-token)
  - `curr_chars`: torch.Tensor of shape (14040,) containing character indices 0-25
- [ ] File is loadable via `torch.load()` without custom unpickling
- [ ] File size is reasonable (estimated 1-2 GB for 14,040 128×128 images)

### Reproducibility
- [ ] Generation is fully reproducible with fixed random seed
- [ ] Same seed produces byte-identical output across runs
- [ ] Random seed must be configurable and documented
- [ ] Augmentation order and parameters are logged or deterministic

### Error Handling
- [ ] Graceful failure if Open Sans font is not available
- [ ] Validation that generated images contain actual character pixels (not blank)
- [ ] Verification that output .pt file is readable and contains expected shapes
- [ ] Error messages clearly indicate what went wrong and how to resolve

### Performance
- [ ] Dataset generation completes in under 30 minutes on standard hardware
- [ ] Memory usage remains under 16 GB during generation
- [ ] No memory leaks during batch processing

## Edge Cases

### Font & Rendering
- [ ] Handle case where Open Sans font file is not installed on system
- [ ] Handle case where font rendering produces blank or extremely small characters
- [ ] Handle potential font rendering inconsistencies across platforms (Linux/macOS/Windows)
- [ ] Verify that all 26 characters (a-z) render distinctly and legibly

### Image Processing
- [ ] Rotation may produce images with empty corners - ensure they're filled with background (white/0.0)
- [ ] Extreme scale values (0.8 or 1.2) may cause character to be unrecognizable - verify readability
- [ ] Blur with kernel size 1 should be no-op (verify it produces minimal change)
- [ ] Noise on near-white backgrounds should not corrupt the image irreparably
- [ ] Chained augmentations may produce degenerate cases - verify output quality

### Data Integrity
- [ ] Handle incomplete augmentation cycles (e.g., if generation crashes mid-way)
- [ ] Verify no duplicate samples are created (all 14,040 should be unique)
- [ ] Ensure prev_chars and curr_chars tensors align with images (same batch size)
- [ ] Validate that character indices are within valid range (0-26 for prev_chars, 0-25 for curr_chars)

### Edge Inputs
- [ ] Start-of-sequence token (prev_char = 0) should produce valid images
- [ ] First character pair (prev_char=0, curr_char=0) must be handled correctly
- [ ] Last character pair (prev_char=25, curr_char=25) must be handled correctly

## Dependencies

### Required Libraries
- **PyTorch**: For tensor creation and .pt file format (torch.save/torch.load)
- **Pillow (PIL)**: For image creation, rendering, and manipulation
- **NumPy**: For numerical operations and random number generation
- **Fonts**: Open Sans font (TrueType .ttf or OpenType .otf file)
  - Should be bundled or auto-downloaded
  - Fallback mechanism if system installation unavailable

### Optional Libraries
- **OpenCV** (cv2): Alternative for faster image processing if performance is critical
- **torchvision**: If using torchvision.transforms for standardized augmentations

### System Requirements
- **Disk Space**: Minimum 5 GB free space (for output file + working space)
- **RAM**: Minimum 8 GB recommended, 16 GB for comfortable operation
- **Python Version**: 3.8+
- **OS**: Linux, macOS, or Windows

### Configuration Files
- Font path or download URL
- Random seed (for reproducibility)
- Output directory path
- Image dimensions (default 128×128)
- Number of augmentations per character-context pair

## Implementation Notes

### Suggested Approach
1. Load or download Open Sans font
2. Initialize character set (a-z) and context set (start-token + a-z)
3. For each (curr_char, prev_char) pair:
   - Render character to base image (128×128 grayscale)
   - Apply 20 augmentation variations with deterministic parameters
   - Collect result images and metadata
4. Stack all tensors and save to .pt file with metadata keys
5. Validate output file integrity and sample quality

### Testing Strategy
- Unit tests for individual augmentation functions
- Integration test for full dataset generation
- Validation test to verify output shapes, dtypes, and value ranges
- Spot-check visualization of generated samples
- Reproducibility test (generate twice, verify identical outputs)

### Documentation
- Code should include docstrings for augmentation functions
- README should explain dataset format and usage
- Example notebook showing how to load and visualize samples
