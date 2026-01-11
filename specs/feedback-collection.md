# Feedback Signal Collection Spec

## Purpose

Provide Claude-based personality evaluation feedback for evolved glyphs. The system captures qualitative personality ratings (1-10 scale) across multiple dimensions (fruity, aggressive, elegant, etc.) for randomly sampled glyphs during each evaluation run. These feedback signals serve as training data for the evolution algorithm, allowing the system to optimize glyph generation toward desired personality characteristics.

## Overview

- **Trigger**: Each evaluation run samples 3 random glyphs from the current population
- **Process**: Claude analyzes base64-encoded glyph images via vision API
- **Output**: Per-glyph personality ratings stored in structured feedback files
- **Storage**: `networks/[id]/runs/N_eval_X_glyph-[char]_feedback.txt`
- **Purpose**: Quantify aesthetic/personality traits to guide evolutionary selection

## Acceptance Criteria

### Functional Requirements

1. **Sampling**
   - Exactly 3 random glyphs are selected per evaluation run
   - Sampling is uniformly random from available glyphs in current population
   - Glyphs without prior feedback can be sampled (no feedback bias)
   - Same glyph is not sampled twice in a single evaluation run

2. **Image Encoding**
   - Glyph images are converted to base64 format before API transmission
   - Image format is consistently PNG or specified bitmap format
   - Encoded images maintain fidelity sufficient for visual analysis
   - Encoding process does not corrupt or distort glyph appearance

3. **Claude Evaluation**
   - Claude API receives base64-encoded image with personality rating prompt
   - Claude provides 1-10 numeric rating for each personality dimension
   - At least 3 personality dimensions are evaluated (e.g., fruity, aggressive, elegant)
   - Response format is consistent and machine-parseable (JSON or structured text)
   - API calls include appropriate error handling and retry logic

4. **Feedback Storage**
   - Feedback file created at: `networks/[id]/runs/N_eval_X_glyph-[char]_feedback.txt`
   - Where `N` is evaluation run number, `X` is sample index (0-2), `[char]` is glyph character
   - File format is consistent and parseable (JSON, YAML, or delimited text)
   - Each feedback record includes: glyph ID, timestamp, all personality ratings, model metadata
   - Feedback persists across application restarts
   - Multiple feedback entries for same glyph accumulate without overwriting

5. **Personality Dimensions**
   - At minimum: fruity, aggressive (as specified)
   - May include: elegant, playful, minimal, bold, organic, geometric, etc.
   - Dimension list is centralized/configurable
   - Ratings scale consistently from 1 (minimal trait) to 10 (extreme trait)

6. **Integration**
   - Feedback collection runs after each evaluation cycle
   - Does not block main evaluation process (async acceptable)
   - Feedback collection status is logged/reported
   - Failed feedback collection does not prevent evaluation completion

### Behavioral Requirements

1. **Success Path**
   - User runs evaluation that triggers feedback collection
   - 3 random glyphs selected without error
   - Claude vision API successfully processes each image
   - All 3 feedback files created with complete ratings
   - System logs indicate successful feedback collection

2. **Partial Failure**
   - If 1-2 glyphs fail evaluation, system continues with remaining samples
   - Logs clearly indicate which glyphs succeeded/failed
   - Evaluation run still completes
   - User is notified of any partial failures

3. **Full Failure**
   - If Claude API is unreachable, graceful degradation occurs
   - Error is logged with timestamp and context
   - Evaluation run completes without feedback
   - System remains in consistent state

## Edge Cases

1. **Population Size**
   - If population has fewer than 3 glyphs, feedback collection adapts
   - Possible behaviors: evaluate all available, skip feedback, raise error (define behavior)
   - Documented clearly in code

2. **Image Encoding Failures**
   - Glyph file missing or corrupted: skip glyph, log error, continue with next
   - Encoding fails (e.g., invalid format): log error, skip glyph
   - File system permissions prevent read: handled gracefully

3. **Claude API Issues**
   - Timeout (>30s): retry up to 2 times with exponential backoff
   - Rate limit (429): wait and retry (respect Retry-After header)
   - Auth failure (401): log error, skip feedback, don't retry
   - Invalid response format: log error, skip glyph, continue

4. **Concurrent Evaluation Runs**
   - Multiple evaluation runs may sample overlapping glyphs
   - Feedback files for same glyph can coexist (numbered or timestamped)
   - No race conditions in file writing (atomic writes or locking)
   - Each run has independent feedback collection

5. **Storage Constraints**
   - Feedback file directory may not exist: auto-create with proper permissions
   - Disk full during write: graceful error, partial file cleanup
   - Path length exceeds system limit: handled or documented
   - Special characters in glyph IDs: properly escaped in filenames

6. **Stale or Invalid Data**
   - Glyph file deleted between sampling and feedback collection: skip gracefully
   - Feedback file already exists for glyph: append or overwrite (define behavior)
   - Timestamp/metadata inconsistencies: handled or ignored per design

7. **Personality Rating Ambiguity**
   - Claude unable to rate (e.g., abstract/ambiguous glyph): default behavior
   - Options: use neutral midpoint (5), skip dimension, request clarification, log issue
   - Documented in implementation

## Dependencies

### External Services
- **Claude Vision API**: Required for glyph image analysis
  - Model: Claude 3.5 Sonnet or compatible vision-enabled model
  - Authentication: API key (via environment variable or config)
  - Rate limits: Plan for rate limit handling
  - Cost: Budget for image analysis tokens

### Internal Components
- **Glyph Storage**: Ability to locate and read glyph files by ID
- **Network State**: Access to current population data
- **Run Management**: Integration with evaluation run numbering/tracking
- **Filesystem**: Write access to `networks/[id]/runs/` directory

### Configuration
- **Personality Dimensions**: Centralized definition (config file or enum)
- **Rating Scale**: Defined endpoints and interpretation (1-10 standard)
- **Feedback Prompt**: Template for Claude with clear personality descriptions
- **Retry Policy**: Max retries, backoff strategy, timeout values
- **Sample Size**: Configurable (currently 3, may vary per design)

### Environment
- **Base64 Encoding Library**: Available in language/runtime
- **File I/O**: Standard filesystem access
- **API Client**: Anthropic SDK or HTTP client for Claude API
- **Error Logging**: System-wide logging infrastructure

### Data Format
- **Image Format**: Standardized (PNG recommended for glyphs)
- **Feedback Format**: JSON or structured text (specify schema)
- **Personality Taxonomy**: Defined vocabulary of trait names and descriptions

## Implementation Notes

1. **Prompt Engineering**: Craft Claude prompt to be precise about personality interpretation
   - Example: "Rate how 'fruity' this glyph is (1=minimal, 10=extremely fruity). Consider curves, roundness, and playful characteristics."
   - Should include visual reference or examples if helpful

2. **Performance**: Base64 encoding and API calls may be slow
   - Consider batch processing if multiple images sent simultaneously
   - Async/parallel processing to avoid blocking main loop
   - Cache encoded images if same glyph evaluated multiple times

3. **Testing Strategy**
   - Unit tests: Feedback file creation, format validation
   - Integration tests: End-to-end with mock Claude API
   - Mock Claude responses for deterministic testing
   - Edge case testing with missing files, invalid data

4. **Monitoring**
   - Track feedback collection success rate per run
   - Log latency of Claude API calls
   - Monitor storage usage of feedback files
   - Alert on repeated API failures

5. **Future Enhancements**
   - Machine learning model to predict ratings (reduce API calls)
   - User feedback override mechanism
   - Feedback aggregation and trend analysis
   - Integration with evolutionary fitness function
