# Online Learning Mechanism Specification

## Purpose

The Online Learning Mechanism enables real-time weight updates based on Claude feedback without requiring full model retraining. This system implements a REINFORCE-style policy gradient approach on the style latent variable (style_z) to continuously improve glyph generation and style responsiveness based on personality match ratings from Claude evaluations.

### Goals

1. **Incremental Learning**: Update model weights in response to immediate feedback signals
2. **Efficiency**: Avoid computationally expensive full retraining cycles
3. **Responsiveness**: Fine-tune decoder layers to improve style coherence with personality characteristics
4. **Persistence**: Maintain learned updates across generation runs in organized network directories

---

## Acceptance Criteria

### Behavioral Requirements

1. **Feedback Integration**
   - System accepts Claude ratings (1-10 scale) for personality-style matching
   - Ratings are processed into policy gradient signals
   - Gradient signals are applied to style_z and decoder parameters

2. **Weight Update Mechanism**
   - REINFORCE-style updates are computed based on feedback scores
   - Decoder layers (at least final 2-3 layers) are fine-tuned with calculated gradients
   - Updates occur within a single generation run without model checkpointing

3. **Run Persistence**
   - Updated weights are stored in `networks/[id]/runs/[run_id]/` directory structure
   - Each run maintains its own weight snapshots if multiple feedback cycles occur
   - Metadata captures feedback history (scores, gradients applied, timestamp)

4. **Style Responsiveness**
   - After learning updates, subsequent glyph generations show measurable style variation aligned with feedback
   - Low-rated glyphs (1-4) trigger gradient updates that reduce similar style_z values in future generations
   - High-rated glyphs (7-10) trigger updates that increase likelihood of similar style_z in future generations

### Testable Metrics

1. **Gradient Convergence**
   - Gradient norms are computed and logged for each feedback cycle
   - Gradient norms remain stable (no NaN or Inf values) over 10+ consecutive updates
   - Loss computed from feedback signal decreases monotonically over at least 3 consecutive update steps

2. **Style Coherence**
   - Average personality match score on a test set improves by at least 10% after 5 feedback cycles
   - Style_z embeddings for positively-rated glyphs cluster closer together post-update
   - Generated glyphs from same style_z show reduced variance in evaluation scores after learning

3. **Computational Efficiency**
   - Single feedback cycle with gradient update completes in < 500ms (on reference hardware)
   - Memory overhead of maintaining update state < 50MB per active network

4. **Run Organization**
   - All weight updates for a run are contained in `networks/[network_id]/runs/[run_id]/` subdirectory
   - Metadata file `run_metadata.json` logs: timestamps, feedback scores, applied gradients, style_z values
   - Checkpoints stored as `weights_v[N].pt` where N increments with each update cycle

---

## Edge Cases

### Feedback Extremes

1. **Contradictory Feedback**
   - Same style_z receives ratings 2 and 9 in different feedback cycles
   - **Handling**: Weight updates proportionally to feedback recency; implement decay for older signals
   - **Test**: Verify system doesn't oscillate indefinitely; loss converges despite contradictions

2. **Sparse Feedback**
   - Only 1-2 glyphs rated in a run before new run begins
   - **Handling**: Ensure updates don't overfit to minimal data; apply gradient clipping or adaptive learning rate
   - **Test**: Verify generalization on held-out test set despite minimal training signal

3. **Extreme Ratings**
   - All feedback is extreme (all 1s or all 10s) for an extended period
   - **Handling**: Scale gradient signals or implement confidence weighting to avoid one-directional drift
   - **Test**: Verify style_z distribution remains bounded and interpretable

### State Management Issues

4. **Stale Weights in Multi-Run Sessions**
   - New run starts but loads weights from previous run that had poor feedback
   - **Handling**: Implement version control; track feedback quality metrics alongside weights
   - **Test**: Verify weights are correctly loaded and applied to new generation batches

5. **Concurrent Feedback Updates**
   - Multiple feedback signals arrive for same run simultaneously or during gradient computation
   - **Handling**: Queue feedback updates or implement locking mechanism for weight updates
   - **Test**: Verify no race conditions; weight updates are applied atomically

### Model-Specific Edge Cases

6. **Gradient Explosion**
   - REINFORCE gradient on style_z produces extremely large updates destabilizing style parameters
   - **Handling**: Implement gradient clipping (L2 norm threshold) or learning rate scheduling
   - **Test**: Monitor gradient norms; verify max gradient magnitude < 10x baseline after clipping

7. **Decoder Layer Saturation**
   - Fine-tuned decoder layers reach weight extremes, causing numerical instability
   - **Handling**: Apply L2 regularization; monitor weight statistics during updates
   - **Test**: Verify weight distributions remain Gaussian-like post-update; no weights exceed [-5, 5] range

8. **Cold Start**
   - First generation in a network has no feedback history for online learning
   - **Handling**: Use initial weights from base model; start learning after minimum feedback threshold (e.g., 3 rated glyphs)
   - **Test**: Verify system generates valid glyphs without learning data

---

## Dependencies

### Internal Dependencies

1. **Policy Gradient Engine**
   - `src/training/reinforce.py`: Computes policy gradient from feedback scores
   - Must expose: `compute_policy_gradient(feedback_scores, style_z, device) -> Tensor`

2. **Feedback Interface**
   - `src/evaluation/claude_feedback.py`: Receives and normalizes Claude ratings
   - Must expose: `process_feedback(rating: int) -> float` (converts 1-10 scale to [-1, 1] or [0, 1])

3. **Model Components**
   - Decoder module with identifiable fine-tunable layers
   - Style_z latent variable interface accessible for gradient computation
   - Must support in-place weight updates without full forward pass

4. **Persistence Layer**
   - `src/utils/checkpoint.py`: Saves/loads weights and metadata
   - Directory structure: `networks/[network_id]/runs/[run_id]/`
   - Must handle: weight snapshots, metadata serialization, version tracking

5. **Logging & Telemetry**
   - `src/utils/logging.py`: Records gradient norms, loss curves, update timestamps
   - Must track: feedback signals, applied gradients, weight statistics, memory usage

### External Dependencies

1. **PyTorch** (>= 2.0)
   - `torch.autograd`: For gradient computation and application
   - `torch.optim`: Potentially for adaptive learning rate scheduling (AdamW, etc.)

2. **Claude API**
   - Feedback submission endpoint (rating persistence)
   - Assumed to be handled by evaluation pipeline, not this component

### Data Dependencies

1. **Network Configuration**
   - `networks/[network_id]/config.yaml`: Contains model architecture, style_z dimension, learning rate
   - Must specify: which decoder layers are fine-tunable, gradient clipping thresholds, learning rate schedule

2. **Run History**
   - Previous feedback scores stored in `networks/[network_id]/runs/*/run_metadata.json`
   - Optional: Load historical patterns for meta-learning or adaptive thresholds

### Hardware/Environment Dependencies

1. **GPU/CUDA**
   - Gradient computation benefits from GPU acceleration
   - Fallback to CPU supported but slower (< 1 glyphs/second acceptable threshold)

2. **Storage**
   - Minimum 100MB per network for checkpoint history (configurable retention)
   - Sequential write performance critical for high-frequency feedback cycles

3. **System Resources**
   - Multi-threaded or async I/O for checkpoint operations to avoid blocking generation
   - Memory pool for temporary gradient buffers (typically < 50MB)

---

## Implementation Notes

### Key Interactions

- **Generation Pipeline**: After glyph generation, style_z values and generated tensors are saved with metadata
- **Feedback Loop**: Claude ratings trigger `apply_online_update()` → gradient computation → weight update → checkpoint
- **Run Lifecycle**: Weights accumulate updates throughout run; final snapshot saved at run completion

### Configuration Parameters (to be defined in network config)

```yaml
online_learning:
  enabled: true
  learning_rate: 0.001
  gradient_clip_norm: 1.0
  finetune_layers: ["decoder.3", "decoder.4"]
  feedback_scale: "linear"  # or "nonlinear"
  update_frequency: "per_rating"  # or "batched"
```

### Expected Integration Points

1. Generation pipeline loads run directory before generating glyphs
2. Feedback processor triggers weight updates asynchronously or synchronously
3. Checkpointing system persists weights without interrupting generation
4. Monitoring system logs all metrics for post-run analysis
