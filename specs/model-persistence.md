# Model Persistence Specification

## Purpose

Enable trained models to persist their state across user sessions, including trained weights, personality vectors, and complete run history. This ensures users can resume training, evaluate past models, and maintain a continuous record of personality font evolution.

## Overview

Model persistence provides the foundational data layer for maintaining personality font state across application sessions. The system manages file storage, retrieval, and versioning of trained models organized by font ID within a structured directory hierarchy.

## Acceptance Criteria

### Storage Structure
- [ ] Models are stored in `networks/[id]/` directory structure, where `[id]` follows the format: `{number}-{personality-name}` (e.g., "1-fruity", "2-dumb", "3-aggressive-sans")
- [ ] Each font ID directory contains exactly one `runs/` subdirectory
- [ ] The `runs/` directory contains one subdirectory per training run with unique identifiers

### Weights Persistence
- [ ] Trained model weights are saved to disk after each completed training run
- [ ] Weights file includes complete model architecture parameters and layer states
- [ ] Weights can be loaded successfully to resume from previous training state
- [ ] Loading weights restores the exact same model state (bit-for-bit compatibility not required, but numerical equivalence required to 6 decimal places)
- [ ] Weights file size does not exceed 500MB per model

### Run History
- [ ] Each run creates a new directory in `networks/[id]/runs/` with a unique identifier
- [ ] Run identifiers are sortable and chronologically ordered (e.g., ISO 8601 timestamps or sequential numbers)
- [ ] Run metadata includes: start time, end time, completion status, and training parameters
- [ ] Run history is queryable to list all runs for a given font ID in chronological order

### Files Per Run
- [ ] Each run directory contains:
  - `weights`: Model weights file (format: .pt or .pth for PyTorch)
  - `sample.png`: Generated sample output image from the model
  - `eval-glyphs/`: Directory containing evaluation glyph samples
  - `feedback.json`: Structured feedback data from training session
- [ ] All files in a run directory are atomic (either all present or all absent; no partial runs)
- [ ] Missing expected files for a completed run raises recoverable error with clear messaging

### Personality Vector Persistence
- [ ] Personality vectors are embedded within or alongside weights file
- [ ] Personality vector state is identical after load-and-resume cycles
- [ ] Personality vector dimensionality is preserved (no lossy compression)
- [ ] Personality vectors can be exported separately for analysis

### Session Recovery
- [ ] Application startup successfully discovers all existing fonts and runs
- [ ] Listing fonts returns all font IDs in `networks/` directory
- [ ] Accessing a font loads its most recent run state
- [ ] Run selection by ID retrieves exact run state without ambiguity
- [ ] Startup time to discover 100+ runs does not exceed 2 seconds

### Data Integrity
- [ ] Corrupted weight files are detected and reported with actionable error messages
- [ ] Partial/incomplete runs (missing expected files) are flagged as incomplete
- [ ] Duplicate font IDs are impossible (enforced at creation time)
- [ ] File permissions allow read/write access to all model files
- [ ] Concurrent access to different font IDs does not cause corruption

### Backward Compatibility
- [ ] System handles missing optional fields in legacy feedback.json files
- [ ] System handles weight files from previous model versions (with appropriate warnings)
- [ ] Migration path exists if weight file format changes (versioning strategy defined)

## Edge Cases

### File System Issues
- **Corrupted weights file**: System detects and reports with line-of-sight to manual recovery
- **Missing run files**: Run is marked as incomplete; individual files (e.g., sample.png) may be regenerated
- **Disk full during save**: Save operation fails cleanly without partial state; no corrupted files left behind
- **File locked by other process**: System retries with exponential backoff (up to 3 retries) or raises clear error
- **Special characters in font ID**: Font IDs must only contain alphanumeric characters and hyphens

### State Consistency
- **Weights exist but evaluation files missing**: Run is usable (weights can be loaded); missing assets can be regenerated on demand
- **Multiple runs in same timestamp**: Run IDs include microsecond precision and sequence numbers to ensure uniqueness
- **Parent directory deleted externally**: System detects missing directory on startup and raises recoverable error
- **Font ID folder exists but runs/ subdirectory missing**: System creates runs/ directory automatically or raises error if auto-creation is disabled

### Naming and ID Conflicts
- **Duplicate font IDs across case variations**: File system case-sensitivity considered in design; enforced lowercase font IDs
- **ID format violations**: IDs not matching pattern {number}-{name} are rejected at creation time
- **Non-existent font access**: Attempting to load non-existent font ID raises KeyError-like exception with helpful message

### Large-Scale Operations
- **Listing 10,000+ runs**: Directory listing completes within 5 seconds; pagination supported if needed
- **Loading 500MB+ weight file**: Memory usage stays within reasonable bounds; streaming/lazy loading considered for future iterations

## Dependencies

### Internal Dependencies
- **Training pipeline**: Must call persistence layer to save weights and feedback after each run
- **Evaluation system**: Must read weights to perform model evaluation
- **UI/CLI layer**: Must display available fonts and run history via persistence API

### External Dependencies
- **PyTorch serialization**: Uses `torch.save()` / `torch.load()` for weights (or equivalent framework)
- **File system**: Requires read/write access to `/home/liam/Documents/projects/aoe/networks/` directory
- **Python standard library**: `os`, `pathlib`, `json`, `datetime` modules

### Framework/Library Requirements
- **Deep Learning Framework**: PyTorch 1.9+ (or TensorFlow 2.5+, depending on implementation)
- **Image handling**: PIL/Pillow for sample.png I/O
- **JSON serialization**: Built-in `json` module for feedback storage

## Implementation Notes

### Directory Structure Example
```
networks/
├── 1-fruity/
│   └── runs/
│       ├── 2026-01-11T08:30:00.000000/
│       │   ├── weights
│       │   ├── sample.png
│       │   ├── eval-glyphs/
│       │   │   ├── A.png
│       │   │   ├── B.png
│       │   │   └── ...
│       │   └── feedback.json
│       └── 2026-01-11T10:15:30.500000/
│           ├── weights
│           ├── sample.png
│           ├── eval-glyphs/
│           └── feedback.json
├── 2-dumb/
│   └── runs/
│       └── 2026-01-10T14:22:10.300000/
│           └── ...
└── 3-aggressive-sans/
    └── runs/
        └── ...
```

### API Surface (High-Level)
- `save_model(font_id, weights, personality_vector, sample_image, eval_glyphs, feedback)` → run_id
- `load_model(font_id, run_id=None)` → (weights, personality_vector, metadata)
- `list_fonts()` → List[str]
- `list_runs(font_id)` → List[RunMetadata]
- `get_run_info(font_id, run_id)` → RunMetadata

### Version Strategy
- Weight file versioning via metadata field: `"weights_version": "v1"`
- Major version changes trigger migration logic or deprecation warnings
- Feedback schema versioned independently: `"feedback_schema": "1.0"`

## Success Metrics

- All acceptance criteria passing
- No data loss across session boundaries
- Load time for 1,000+ runs under 2 seconds
- 100% file integrity verification on load
- Clear error messages for all failure scenarios
- Unit test coverage: >90% for persistence layer
