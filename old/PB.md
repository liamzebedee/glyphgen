# Project Breakdown: Generative Font Renderer

## High-Level Work Chunks

### Chunk 1: Project Setup & Infrastructure
**Delivers:** Project skeleton with dependencies, configuration, and utility modules

- Initialize Python project structure with proper packaging
- Set up dependencies: PyTorch, PIL/Pillow, NumPy, matplotlib
- Create configuration module (hyperparameters, paths, constants)
- Create utility modules (device detection, logging, file I/O helpers)
- Set up directory structure: `data/`, `networks/`, `outputs/`, `models/`

**Dependencies:** None (starting point)

---

### Chunk 2: Data Generation Pipeline
**Delivers:** Complete training dataset of 14,040 glyph images with labels

- Font loading and character rendering using PIL/FreeType
- Character-to-index mapping (A-Z → 1-26, null context → 0)
- Augmentation pipeline:
  - Rotation (random small angles)
  - Scale variations
  - Translation jitter
  - Stroke width modification
  - Gaussian blur
  - Additive noise
- Generate all combinations: 26 chars × 27 prev contexts × 20 augmentations
- Save as indexed dataset with metadata (prev_char, curr_char, image path)
- Create PyTorch Dataset/DataLoader classes

**Dependencies:** Chunk 1

---

### Chunk 3: GlyphNet Architecture
**Delivers:** Complete neural network model definition (~2.9M parameters)

- Character embedding layers (prev_char: 27-dim, curr_char: 26-dim)
- Style vector input layer (64-dim latent space)
- MLP encoder combining embeddings + style
- Transposed convolutional decoder:
  - Progressive upsampling to 128×128
  - Appropriate kernel sizes, strides, padding
- Output layer with sigmoid activation (grayscale 0-1)
- Parameter count verification (~2.9M)
- Model summary/introspection utilities

**Dependencies:** Chunk 1

---

### Chunk 4: Loss Functions & Training Utilities
**Delivers:** Custom loss functions and training infrastructure

- MSE loss component
- Edge-aware loss component:
  - Sobel/Laplacian edge detection
  - Edge-weighted pixel loss
- Combined weighted loss function
- AdamW optimizer configuration
- Cosine annealing learning rate scheduler
- Checkpoint save/load utilities
- Training metrics logging

**Dependencies:** Chunk 3

---

### Chunk 5: Training Loop
**Delivers:** Complete training pipeline producing trained weights (<20MB)

- Training loop with epoch/batch iteration
- Validation split and evaluation
- Random style_z sampling during training
- Progress logging and visualization
- Checkpoint saving (best model, periodic saves)
- Training for 100 epochs, batch size 64
- Performance validation (<5ms inference on target GPU)
- Final model export

**Dependencies:** Chunks 2, 3, 4

---

### Chunk 6: Glyph Rendering Engine
**Delivers:** Inference module for single glyph generation

- Model loading from checkpoint
- Single glyph inference function
- Batch glyph inference for efficiency
- Style vector management (fixed, random, interpolated)
- Output post-processing (tensor → image)
- Performance benchmarking utilities

**Dependencies:** Chunks 3, 5

---

### Chunk 7: Text Rendering Demo
**Delivers:** Complete text-to-image rendering with proper spacing

- Character width estimation from rendered glyphs
- Character-width-aware horizontal spacing calculation
- Sequential glyph rendering with context (prev_char awareness)
- Glyph concatenation into text strip
- Demo script rendering "The revolution will not be televised"
- Output image saving

**Dependencies:** Chunk 6

---

### Chunk 8: Personality Font System (Online Learning - Secondary)
**Delivers:** Multi-personality font framework with persistent learning

**Sub-chunk 8a: Personality Infrastructure**
- Directory structure: `networks/[personality_id]/`
- Three personalities: "fruity", "dumb", "aggressive_sans"
- Per-personality weight storage
- Sample rendering and archival per run

**Sub-chunk 8b: Evaluation Pipeline**
- Random glyph selection (3 per evaluation)
- Glyph rendering for evaluation
- Interface for Claude rating input (1-10 scale)
- Rating storage and history

**Sub-chunk 8c: REINFORCE-Style Learning**
- Style vector (style_z) gradient computation
- Reward-weighted gradient updates
- Decoder fine-tuning for style responsiveness
- Weight persistence after updates

**Dependencies:** Chunks 6, 7

---

## Implementation Order

```
Phase 1: Foundation
  [1] Project Setup ──────────────────────────────┐
                                                  │
Phase 2: Data & Model                             ▼
  [2] Data Generation ◄─────────────────────── Chunk 1
  [3] GlyphNet Architecture ◄─────────────────────┘

Phase 3: Training
  [4] Loss Functions ◄──────────────────────── Chunk 3
  [5] Training Loop ◄───────────────── Chunks 2, 3, 4

Phase 4: Inference & Demo
  [6] Rendering Engine ◄────────────────── Chunks 3, 5
  [7] Text Demo ◄──────────────────────────── Chunk 6

Phase 5: Online Learning (Secondary)
  [8] Personality System ◄─────────────── Chunks 6, 7
```

---

## Critical Path

**1 → 2 → 5 → 6 → 7** is the critical path to the working demo.

Chunk 3 (architecture) and Chunk 4 (loss functions) can be developed in parallel with Chunk 2 (data generation), then converge at Chunk 5 (training).

Chunk 8 (personality/online learning) is explicitly secondary and can be deferred until the core system is functional.

---

## Deliverables Summary

| Chunk | Primary Deliverable | Key Metrics |
|-------|---------------------|-------------|
| 1 | Project skeleton | All directories, requirements.txt |
| 2 | Training dataset | 14,040 samples, .pt file |
| 3 | GlyphNet model | ~2.9M params, <20MB |
| 4 | Loss + training utils | Edge-aware loss, schedulers |
| 5 | Trained weights | <5ms inference, <20MB |
| 6 | Rendering engine | Single/batch inference |
| 7 | Demo script | "The revolution..." image |
| 8 | Online learning | 3 personality fonts |
