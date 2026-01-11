# Technical Specification: Generative Font Renderer

## 1. System Overview and Architecture

### 1.1 Purpose
A neural network-based font rendering system that generates 128x128 black-and-white glyph bitmaps from character inputs, supporting context-aware rendering through a 2-character sliding window (previous + current character). The system includes an online learning component that evolves font "personalities" through Claude-based aesthetic evaluation.

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GENERATIVE FONT RENDERER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐    ┌──────────────────┐    ┌────────────────────────┐   │
│  │   Training    │    │  Neural Network  │    │    Inference Engine    │   │
│  │   Data Gen    │───▶│   (GlyphNet)     │───▶│    (< 5ms/glyph)       │   │
│  │   (MNIST-like)│    │                  │    │                        │   │
│  └───────────────┘    └────────┬─────────┘    └────────────────────────┘   │
│                                │                                            │
│                                ▼                                            │
│                    ┌──────────────────────┐                                │
│                    │   Online Learning    │                                │
│                    │      System          │                                │
│                    │  ┌────────────────┐  │                                │
│                    │  │ Claude Eval    │  │                                │
│                    │  │ (Personality)  │  │                                │
│                    │  └────────────────┘  │                                │
│                    └──────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Data Generator | `generate_dataset.py` | Creates MNIST-style training data from Open Sans |
| Neural Network | `glyphnet.py` | Single-file PyTorch model definition + training |
| Demo Renderer | `demo.py` | Renders sample text to concatenated image |
| Online Learning | `online_learn.py` | Claude-integrated personality evolution system |

---

## 2. Neural Network Architecture (GlyphNet)

### 2.1 Design Constraints Analysis

**Weight Budget:** 20MB = 20,971,520 bytes = ~5,242,880 float32 parameters

**Latency Budget:** 5ms per glyph on RTX 3090
- RTX 3090: ~35.6 TFLOPS FP32
- 5ms budget allows ~178 billion FLOPs (theoretical max)
- Practical target: <10M FLOPs per inference for safety margin

### 2.2 Architecture: Conditional Decoder with Context Embedding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GLYPHNET ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS:                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐              │
│  │ prev_char    │  │ curr_char    │  │ style_vector (z)     │              │
│  │ [0-26]       │  │ [0-26]       │  │ [latent_dim=64]      │              │
│  │ (0=padding)  │  │ (1-26=a-z)   │  │ (learnable per font) │              │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘              │
│         │                 │                      │                          │
│         ▼                 ▼                      │                          │
│  ┌──────────────────────────────┐                │                          │
│  │   Character Embedding        │                │                          │
│  │   nn.Embedding(27, 64)       │                │                          │
│  │   = 1,728 params             │                │                          │
│  └──────────────┬───────────────┘                │                          │
│                 │                                │                          │
│                 ▼                                │                          │
│  ┌──────────────────────────────┐                │                          │
│  │   Concatenate                │◀───────────────┘                          │
│  │   [prev_emb, curr_emb, z]    │                                           │
│  │   = 64 + 64 + 64 = 192       │                                           │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────┐                                           │
│  │   MLP Projection             │                                           │
│  │   Linear(192, 512)           │                                           │
│  │   + LayerNorm + GELU         │                                           │
│  │   Linear(512, 4*4*256=4096)  │                                           │
│  │   = 192*512 + 512*4096       │                                           │
│  │   = 98,304 + 2,097,152       │                                           │
│  │   = ~2.2M params             │                                           │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────┐                                           │
│  │   Reshape to 4x4x256         │                                           │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │              DECODER (Transposed Convolutions)                    │      │
│  │                                                                   │      │
│  │   Block 1: 4x4x256 → 8x8x128                                     │      │
│  │   ConvT2d(256,128,4,2,1) + BN + GELU                             │      │
│  │   = 256*128*4*4 = 524,288 params                                 │      │
│  │                                                                   │      │
│  │   Block 2: 8x8x128 → 16x16x64                                    │      │
│  │   ConvT2d(128,64,4,2,1) + BN + GELU                              │      │
│  │   = 128*64*4*4 = 131,072 params                                  │      │
│  │                                                                   │      │
│  │   Block 3: 16x16x64 → 32x32x32                                   │      │
│  │   ConvT2d(64,32,4,2,1) + BN + GELU                               │      │
│  │   = 64*32*4*4 = 32,768 params                                    │      │
│  │                                                                   │      │
│  │   Block 4: 32x32x32 → 64x64x16                                   │      │
│  │   ConvT2d(32,16,4,2,1) + BN + GELU                               │      │
│  │   = 32*16*4*4 = 8,192 params                                     │      │
│  │                                                                   │      │
│  │   Block 5: 64x64x16 → 128x128x1                                  │      │
│  │   ConvT2d(16,1,4,2,1) + Sigmoid                                  │      │
│  │   = 16*1*4*4 = 256 params                                        │      │
│  │                                                                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  OUTPUT: 128x128x1 grayscale bitmap [0,1]                                  │
│                                                                             │
│  TOTAL PARAMETERS: ~2.9M (well under 5.2M budget = ~11.6MB)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Detailed Layer Specification

```python
class GlyphNet(nn.Module):
    """
    Generative font renderer network.

    Input shapes:
        prev_char: (batch,) - integer [0-26], 0=start token
        curr_char: (batch,) - integer [1-26], 1-26 = a-z
        style_z:   (batch, 64) - style/personality vector

    Output shape:
        bitmap: (batch, 1, 128, 128) - grayscale [0,1]
    """

    # Architecture constants
    VOCAB_SIZE = 27          # 0=padding/start, 1-26=a-z
    CHAR_EMBED_DIM = 64
    STYLE_DIM = 64
    HIDDEN_DIM = 512
    INIT_SPATIAL = 4
    INIT_CHANNELS = 256
    OUTPUT_SIZE = 128
```

### 2.4 Parameter Count Verification

| Layer | Parameters | Cumulative |
|-------|------------|------------|
| Embedding(27, 64) | 1,728 | 1,728 |
| Linear(192, 512) | 98,816 | 100,544 |
| LayerNorm(512) | 1,024 | 101,568 |
| Linear(512, 4096) | 2,101,248 | 2,202,816 |
| ConvT2d(256, 128) + BN | 524,544 | 2,727,360 |
| ConvT2d(128, 64) + BN | 131,200 | 2,858,560 |
| ConvT2d(64, 32) + BN | 32,832 | 2,891,392 |
| ConvT2d(32, 16) + BN | 8,240 | 2,899,632 |
| ConvT2d(16, 1) | 257 | 2,899,889 |

**Total: ~2.9M parameters = ~11.6MB** (within 20MB budget)

### 2.5 Style Vector Design

The `style_z` vector is the key to personality expression:

```python
# During base training: style_z is sampled from N(0, 1)
# This teaches the network to respond to style variations

# During online learning: style_z becomes a learnable parameter
# Each "font personality" has its own optimized style_z

class FontPersonality:
    def __init__(self, latent_dim=64):
        self.style_z = nn.Parameter(torch.randn(1, latent_dim))
```

---

## 3. Training Data Generation

### 3.1 Dataset Specification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING DATASET STRUCTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Format: PyTorch .pt file containing dict                                   │
│                                                                             │
│  {                                                                          │
│      'images': Tensor[N, 1, 128, 128],  # Rendered glyphs, float32 [0,1]   │
│      'prev_chars': Tensor[N],            # Previous char indices [0-26]     │
│      'curr_chars': Tensor[N],            # Current char indices [1-26]      │
│      'metadata': {                                                          │
│          'font': 'OpenSans-Regular',                                        │
│          'size': 96,                     # Font size in points              │
│          'variations': [...],            # Applied augmentations            │
│      }                                                                      │
│  }                                                                          │
│                                                                             │
│  Dataset size: 26 * 27 * 20 = 14,040 samples                               │
│  (26 current chars × 27 prev chars × 20 augmentation variants)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Generation Pipeline

```python
# generate_dataset.py

"""
Training data generation from Open Sans font.

Pipeline:
1. Load Open Sans TTF via PIL/Pillow
2. For each (prev_char, curr_char) pair:
   a. Render curr_char at 96pt on 128x128 canvas
   b. Apply positioning based on prev_char (kerning simulation)
   c. Apply augmentations for variety
   d. Normalize to [0, 1] grayscale
3. Save as PyTorch tensor file
"""

AUGMENTATIONS = [
    # Geometric
    {'rotation': (-5, 5)},        # Slight rotation
    {'scale': (0.9, 1.1)},        # Scale variation
    {'translate': (-4, 4)},       # Position jitter

    # Stylistic (for expressiveness training)
    {'stroke_width': (0, 2)},     # Outline thickness
    {'blur': (0, 1)},             # Slight blur
    {'noise': (0, 0.05)},         # Salt noise
]
```

### 3.3 Rendering Configuration

```python
RENDER_CONFIG = {
    'canvas_size': (128, 128),
    'font_path': 'fonts/OpenSans-Regular.ttf',
    'font_size': 96,              # Large enough to fill canvas
    'background': 0,              # Black background
    'foreground': 255,            # White text
    'antialias': True,
    'center_glyph': True,
    'horizontal_offset_range': (-8, 8),  # Context-based positioning
}
```

### 3.4 Kerning Simulation

```python
def compute_context_offset(prev_char: int, curr_char: int) -> Tuple[int, int]:
    """
    Simulate kerning/ligature behavior based on character pair.

    Returns (x_offset, y_offset) for curr_char placement.

    Example pairs with special handling:
    - 'f' + 'i' → overlap for fi ligature appearance
    - 'T' + 'o' → tuck 'o' under 'T' overhang
    - 'A' + 'V' → negative kerning
    """
    KERNING_TABLE = {
        (ord('f') - ord('a') + 1, ord('i') - ord('a') + 1): (-4, 0),
        (ord('f') - ord('a') + 1, ord('l') - ord('a') + 1): (-3, 0),
        # ... more pairs
    }
    return KERNING_TABLE.get((prev_char, curr_char), (0, 0))
```

---

## 4. Inference Pipeline

### 4.1 Performance-Optimized Inference

```python
class GlyphRenderer:
    """
    High-performance glyph renderer targeting <5ms per glyph.

    Optimizations:
    1. Model kept on GPU permanently
    2. torch.inference_mode() context
    3. Batched rendering when possible
    4. Pre-allocated output tensors
    5. CUDA graph capture for repeated calls
    """

    def __init__(self, weights_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model = GlyphNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

        # Pre-allocate for single glyph (most common case)
        self._prev_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self._curr_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self._style_buf = torch.zeros(1, 64, device=self.device)

        # Warm up CUDA
        self._warmup()

    @torch.inference_mode()
    def render_glyph(
        self,
        prev_char: str,
        curr_char: str,
        style_z: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Render single glyph.

        Args:
            prev_char: Previous character or '' for start
            curr_char: Current character to render
            style_z: Optional style vector (uses default if None)

        Returns:
            128x128 uint8 numpy array
        """
        # Encode characters
        self._prev_buf[0] = self._encode_char(prev_char)
        self._curr_buf[0] = self._encode_char(curr_char)

        if style_z is not None:
            self._style_buf.copy_(style_z)

        # Forward pass
        output = self.model(self._prev_buf, self._curr_buf, self._style_buf)

        # Convert to numpy uint8
        return (output[0, 0].cpu().numpy() * 255).astype(np.uint8)

    @torch.inference_mode()
    def render_text(self, text: str, style_z: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Render full text string, concatenating glyphs horizontally.

        Uses batched inference for efficiency.
        """
        # Prepare batch
        n = len(text)
        prev_chars = torch.zeros(n, dtype=torch.long, device=self.device)
        curr_chars = torch.zeros(n, dtype=torch.long, device=self.device)

        for i, char in enumerate(text):
            prev_chars[i] = self._encode_char(text[i-1] if i > 0 else '')
            curr_chars[i] = self._encode_char(char)

        style = style_z.expand(n, -1) if style_z else torch.zeros(n, 64, device=self.device)

        # Batched forward pass
        glyphs = self.model(prev_chars, curr_chars, style)

        # Concatenate horizontally with overlap
        return self._concatenate_glyphs(glyphs, text)
```

### 4.2 Latency Breakdown Target

| Operation | Target | Notes |
|-----------|--------|-------|
| Char encoding | 0.01ms | Lookup table |
| GPU transfer | 0.1ms | Already on GPU |
| Forward pass | 3.5ms | Main computation |
| Tensor→NumPy | 0.5ms | Device sync + copy |
| **Total** | **<5ms** | With margin |

---

## 5. Training System

### 5.1 Base Training Configuration

```python
TRAINING_CONFIG = {
    # Data
    'dataset_path': 'data/opensans_glyphs.pt',
    'batch_size': 64,
    'num_workers': 4,

    # Optimization
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'T_0': 10,  # Restart period

    # Training
    'epochs': 100,
    'style_z_std': 1.0,  # Sample style vectors from N(0, 1)

    # Loss
    'reconstruction_loss': 'MSE',  # Or BCE for sharper edges
    'perceptual_weight': 0.1,      # Optional VGG perceptual loss

    # Regularization
    'dropout': 0.1,
    'label_smoothing': 0.0,

    # Hardware
    'device': 'cuda',
    'mixed_precision': True,  # FP16 for speed
    'compile': True,          # torch.compile for 2.0+
}
```

### 5.2 Loss Function

```python
class GlyphLoss(nn.Module):
    """
    Combined loss for glyph reconstruction.

    Components:
    1. MSE reconstruction loss (primary)
    2. Edge-aware loss (encourages sharp boundaries)
    3. Style consistency loss (optional, for coherent fonts)
    """

    def __init__(self, edge_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.edge_weight = edge_weight

        # Sobel edge detection kernels
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = self.mse(pred, target)

        # Edge loss - match edge maps
        pred_edges = self._detect_edges(pred)
        target_edges = self._detect_edges(target)
        edge_loss = self.mse(pred_edges, target_edges)

        return recon_loss + self.edge_weight * edge_loss
```

---

## 6. Online Learning System

### 6.1 Directory Structure

```
networks/
├── base_model.pt                 # Base trained weights
├── 1-fruity/
│   ├── config.json
│   ├── style_z.pt
│   └── runs/
│       ├── 0_weights.pt
│       ├── 0_sample.png
│       ├── 0_eval_0_glyph-a.png
│       ├── 0_eval_0_glyph-a_feedback.txt
│       └── ...
├── 2-dumb/
└── 3-aggressive-sans/
```

### 6.2 Claude Evaluation Protocol

```python
CLAUDE_EVAL_PROMPT = """
You are evaluating a generated font character for its visual personality.

The font is meant to express: "{personality}"

Look at this rendered character and rate how well it expresses the "{personality}" aesthetic.

Scoring criteria:
- 1-3: Does not express the personality at all, or actively contradicts it
- 4-5: Neutral, standard font appearance
- 6-7: Shows some hints of the personality
- 8-9: Clearly expresses the personality
- 10: Perfectly embodies the personality in an interesting way

Respond with ONLY a JSON object in this exact format:
{{
    "score": <integer 1-10>,
    "reasoning": "<brief explanation, max 50 words>"
}}
"""
```

### 6.3 Reward-Based Training

The style_z vector is treated as a policy parameter. We:
1. Sample perturbations around current style_z
2. Render glyphs with perturbed styles
3. Get Claude ratings as rewards
4. Estimate gradient: high-reward perturbations should be reinforced
5. Update style_z in direction of higher rewards

---

## 7. File Structure

```
aoe/
├── fonts/
│   └── OpenSans-Regular.ttf
├── data/
│   ├── opensans_glyphs.pt
│   └── validation_glyphs.pt
├── src/
│   ├── glyphnet.py
│   ├── generate_dataset.py
│   ├── demo.py
│   ├── online_learn.py
│   └── utils.py
├── networks/
│   ├── base_model.pt
│   └── [personality-fonts]/
├── outputs/
│   └── demo_output.png
└── requirements.txt
```

---

## 8. Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.23.0
fonttools>=4.38.0
anthropic>=0.18.0
tqdm>=4.64.0
python-slugify>=8.0.0
```

---

## 9. Implementation Order

1. **Phase 1: Foundation** - Network architecture
2. **Phase 2: Data** - Dataset generation from Open Sans
3. **Phase 3: Training** - Base model training
4. **Phase 4: Demo** - Text rendering demo
5. **Phase 5: Online Learning** - Claude-based personality evolution
