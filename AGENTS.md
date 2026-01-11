# AGENTS.md - Operational Guide

Keep this file brief. Status updates belong in IMPLEMENTATION_PLAN.md.

## Project: Generative Font Renderer

Neural network that renders 128x128 glyph bitmaps with personality and context-awareness.

## Build & Run

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt

# Generate training data
python src/generate_dataset.py

# Train base model
python src/train.py --epochs 100

# Run demo
python src/demo.py

# Online learning (requires ANTHROPIC_API_KEY)
python src/online_learn.py --personality fruity --runs 10
```

## Validation (Backpressure)

```bash
# Tests
pytest tests/

# Typecheck (if using type hints)
mypy src/

# Lint
ruff check src/
```

## Codebase Patterns

- Single-file modules preferred (per spec: `glyphnet.py` is one file)
- Shared utilities in `src/utils.py`
- Config constants in `src/config.py`
- All paths relative to project root
- PyTorch 2.0+ required for torch.compile()

## Directory Structure

```
aoe/
├── src/
│   ├── glyphnet.py       # Neural network
│   ├── generate_dataset.py
│   ├── train.py
│   ├── demo.py
│   ├── online_learn.py
│   ├── config.py
│   └── utils.py
├── data/
│   └── opensans_glyphs.pt
├── networks/
│   └── [id]/runs/
├── fonts/
│   └── OpenSans-Regular.ttf
├── outputs/
├── specs/
└── tests/
```

## Hardware

- Training: RTX 3090 (24GB VRAM)
- Inference target: <5ms per glyph
- Model size: <20MB weights (~2.9M params)

## Operational Notes

<!-- Add learnings here as you build -->
