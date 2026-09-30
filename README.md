# fix-flowAI
AI-powered complaint prioritization and resource optimization system
# 🔧 FixFlow AI — From Complaint to Action

> **AI-powered complaint prioritization and resource optimization system** that converts unstructured complaints into optimized action plans using machine learning, multi-objective evolutionary optimization, and constrained resource allocation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Problem Statement

**Track:** Machine Learning & Artificial Intelligence — Multi-Objective Heuristics + Deep Evolutionary Networks

Cities, campuses, and municipalities receive thousands of complaints daily across diverse domains — road damage, electrical hazards, water leakage, waste management, and more. Current systems only **collect and display** complaints, forcing administrators to manually triage, prioritize, and allocate resources.

**FixFlow AI** solves this by building an **AI pipeline** that:
1. **Understands** natural-language complaints
2. **Predicts** severity, urgency, and resolution requirements
3. **Detects** duplicate reports automatically
4. **Calculates** dynamic priority scores
5. **Optimizes** resource allocation under real-world constraints
6. **Adapts** when conditions change (weather, surges, distribution shifts)

## 🧠 Approach & Algorithmic Logic

### Architecture Overview

```
COMPLAINT → NLP Processing → Multi-Task MLP → Priority Engine → Resource Optimizer → ACTION PLAN
                                    ↑                                    ↑
                          Evolutionary Optimization              Constraint Satisfaction
                           (NSGA-II Multi-Objective)            (Workers/Vehicles/Time)
```

### Five AI Engines

| Engine | Function | Method |
|--------|----------|--------|
| **1. Complaint Understanding** | NLP classification + regression | TF-IDF + Multi-task MLP (shared backbone, 3 heads) |
| **2. Duplicate Detection** | Group similar complaints | TF-IDF cosine similarity clustering |
| **3. Impact Prediction** | Estimate affected scope | Multi-factor regression (people, duration, safety) |
| **4. Dynamic Priority** | Rank issues optimally | Weighted priority: P = w_s·S + w_u·U + w_i·I + w_r·R + w_c·C − w_t·T |
| **5. Resource Optimizer** | Select feasible repair plan | Greedy selection + local improvement under constraints |

### Multi-Task Neural Network

```
Input (TF-IDF 300-dim + 15 structured features)
    ↓
Shared Backbone: Linear(315→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout
    ↓                    ↓                    ↓
Category Head       Severity Head       Resolution Head
(8-class softmax)   (sigmoid regression) (sigmoid regression)
```

**Training Objective:**
```
L = L_category(CE + label_smoothing) + λ₁·L_severity(Huber) + λ₂·L_resolution(Huber) + λ₃·L_reg
```

### Evolutionary Hyperparameter Optimization (NSGA-II)

**Chromosome:** `[hidden_1, hidden_2, dropout, lr, weight_decay, huber_delta, loss_w1, loss_w2]`

**Multi-Objective Optimization:**
- Maximize: ID Macro F1, OOD Macro F1, Seed Stability
- Minimize: ECE (calibration error), Group Gap (fairness), Latency, Parameter Count

**Hard Constraints (candidates violating these are INFEASIBLE):**
- Parameters ≤ 200,000
- Latency ≤ 50ms
- ECE ≤ 0.08
- Group Gap ≤ 0.05

### Resource Optimization

```
max Σ Pᵢxᵢ
subject to:
  Σ Timeᵢxᵢ ≤ T (available time)
  Σ Workersᵢxᵢ ≤ W (available workers)
  Σ Vehicleᵢxᵢ ≤ V (available vehicles)
  xᵢ ∈ {0, 1}
```

Solved via **priority-weighted greedy selection + local swap improvement**.

### Distribution Shift & Robustness

Six controlled perturbation scenarios test model resilience:
1. **Normal** — training-like data
2. **Missing Fields** — randomly remove location, weather, affected_people
3. **Text Noise** — typos, punctuation, case changes
4. **Numeric Shift** — shifted crowd/resolution distributions
5. **Heavy Rain** — weather shock with cascading severity changes
6. **Complaint Surge** — 10× volume with increased duplicate density

### Robust Loss & Regularization
- **Classification:** Cross-entropy with label smoothing (ε=0.1)
- **Regression:** Huber loss (δ configurable) instead of MSE
- **Regularization:** Dropout + Weight Decay + Gradient Clipping + Early Stopping

## 🏗 How the Solution Works End-to-End

1. **User submits complaint** in natural language with optional metadata
2. **Feature extraction:** TF-IDF vectorization + structured feature engineering
3. **Multi-task prediction:** Category classification + severity/resolution regression
4. **Duplicate detection:** Cosine similarity groups similar complaints
5. **Priority calculation:** Weighted multi-factor score (0-100)
6. **Resource optimization:** Greedy + local improvement selects feasible repair plan
7. **Explainability:** Feature contributions explain every decision
8. **Drift simulation:** Environmental changes trigger priority re-computation
9. **Benchmark validation:** Automated 9-step evaluation ensures quality

## 📊 Evaluation Metrics

| Metric | Target | Method |
|--------|--------|--------|
| Classification Accuracy | > 90% | Macro F1 on ID validation |
| OOD Robustness | > 85% | Macro F1 on OOD scenarios |
| Calibration | ECE < 0.08 | Expected Calibration Error |
| Fairness | Gap < 0.05 | Max group F1 difference |
| Latency | < 50ms | Single-sample inference |
| Parameters | < 200K | Model parameter count |
| Determinism | Variance = 0 | 3 runs with seed=42 |

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/fixflow-ai.git
cd fixflow-ai
pip install -r requirements.txt

# Run the benchmark
python evaluator/run_all.py

# Launch the dashboard
streamlit run app.py
```

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📁 Project Structure

```
fixflow-ai/
├── app.py                    # Streamlit entry point
├── config.yaml               # All configuration (no hard-coded constants)
├── AGENTS.md                 # Engineering contract
│
├── data/                     # Data generation & perturbation
│   ├── synthetic_generator.py
│   └── perturbations.py
│
├── models/                   # ML pipeline
│   ├── features.py           # Feature engineering (TF-IDF + structured)
│   ├── multitask_model.py    # PyTorch multi-task MLP
│   ├── train.py              # Training with robust loss
│   └── predict.py            # Inference + explanation
│
├── evolution/                # Evolutionary optimization
│   ├── chromosome.py         # Genetic representation
│   ├── population.py         # Population management
│   ├── nsga2.py              # NSGA-II Pareto optimizer
│   ├── crossover.py          # SBX crossover
│   ├── mutation.py           # Polynomial mutation
│   └── optimize.py           # Main optimizer loop
│
├── prioritization/           # Priority & resource allocation
│   ├── priority.py           # Dynamic priority engine
│   ├── resource_optimizer.py # Constrained optimization
│   └── constraints.py        # Hard constraint definitions
│
├── robustness/               # Distribution shift testing
│   ├── drift.py              # PSI drift detection
│   ├── scenarios.py          # Perturbation scenarios
│   └── stress_test.py        # Stress test runner
│
├── evaluation/               # Metrics & benchmarking
│   ├── metrics.py            # Classification/regression metrics
│   ├── calibration.py        # ECE & reliability diagrams
│   ├── fairness.py           # Group-wise fairness
│   ├── confidence.py         # Bootstrap CI & Hoeffding bounds
│   ├── reproducibility.py    # Determinism verification
│   └── benchmark.py          # Complete benchmark runner
│
├── explainability/           # AI transparency
│   ├── feature_contribution.py
│   └── explanation.py
│
├── database/                 # Persistence
│   ├── schema.py             # SQLite schema
│   └── repository.py         # Data access layer
│
├── ui/                       # Streamlit pages
│   ├── components.py         # Shared UI components
│   ├── dashboard.py          # Command Center
│   ├── report.py             # Report Problem
│   ├── triage.py             # AI Triage
│   ├── issues.py             # Issue Explorer
│   ├── optimize_ui.py        # Resource Optimizer
│   ├── drift_sim.py          # Drift Simulator
│   ├── ai_lab.py             # AI Lab (evolution, Pareto, etc.)
│   └── benchmark_ui.py       # Benchmark Dashboard
│
├── evaluator/                # Automated evaluation
│   ├── run_all.py            # One-click benchmark
│   ├── scoring.py            # Score computation
│   └── ast_report.py         # AST code quality
│
├── tests/                    # pytest test suite
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_optimizer.py
│   ├── test_constraints.py
│   ├── test_drift.py
│   └── test_benchmark.py
│
└── results/                  # Benchmark outputs
```

## ⚙️ Assumptions & Constraints

- **Offline-first:** Core ML pipeline works without internet or external APIs
- **Deterministic:** All operations seeded (seed=42) for reproducibility
- **No PII:** Uses synthetic benchmark data only
- **Resource budgets:** Model must fit within parameter/latency constraints
- **Demo Mode:** Built-in mode with bundled data for reliable presentations

## 🏆 Innovation

> "Most complaint systems only collect and display complaints. FixFlow decides **what should be fixed first** and dynamically changes that decision when the situation changes."

1. **Multi-task learning** with robust loss for simultaneous category/severity/resolution prediction
2. **NSGA-II evolutionary optimization** with hard deployment constraints
3. **Dynamic re-prioritization** under weather shocks and complaint surges
4. **Constrained resource allocation** (not just ranking — actual feasible plans)
5. **Self-evaluating system** with automated 9-step benchmark
6. **Distribution shift resilience** with 6 perturbation scenarios

## 📄 License

MIT License — Built for OptiForge 2026 Hackathon
