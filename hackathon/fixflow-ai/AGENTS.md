# FixFlow AI Engineering Contract

## Mission
Build a reliable, offline-first AI complaint prioritization and resource optimization platform.

## Non-negotiable Rules

### 1. Reproducibility
- Default seed = 42
- Record all experiment configuration
- Record model version and benchmark version

### 2. Reliability
- Critical path must work without internet
- Never require an external LLM for core scoring
- Always provide a saved fallback model
- Validate model artifacts before loading

### 3. ML
- Use robust Huber regression loss
- Use regularization (dropout + weight decay)
- Use gradient clipping
- Measure calibration (ECE)
- Measure OOD performance
- Measure group-wise fairness

### 4. Optimization
- Use explicit hard constraints
- Reject infeasible candidates
- Record every generation
- Save convergence history
- Preserve best feasible candidate

### 5. UI
- Every AI result must display an explanation
- Never display unsupported certainty
- Display model health indicators
- Provide Demo Mode (offline)

### 6. Testing
- Every core module must have tests
- Add regression tests for fixed bugs
- Run complete benchmark before declaring system complete

### 7. Code Quality
- Small modular functions
- Type hints on all functions
- No giant monolithic files
- No hard-coded constants (use config.yaml)
- Docstrings on all public functions

### 8. Data
- Do not require PII
- Synthetic benchmark data must be reproducible
