# RetailSim Project Context

This document captures the full context of the RetailSim project for session continuity. It provides everything needed to understand the codebase, architecture decisions, and current state.

---

## Project Overview

**RetailSim** is a world model for retail transaction simulation built on the Dunnhumby "Let's Get Sort of Real" (LGSR) dataset. The goal is to predict customer shopping behavior - specifically, what products a customer will buy given their history, the time of shopping, the store they visit, and their shopping mission.

### Core Concept

The model learns to answer: *"Given everything we know about a customer, what products will they buy in their next shopping trip?"*

This is framed as a **Masked Event Modeling** task (similar to BERT):
- Take a shopping basket: [Milk, Bread, Eggs, Cheese, Apples]
- Mask 15% of products: [Milk, [MASK], Eggs, Cheese, [MASK]]
- Train the model to predict the masked products

### Dataset

The Dunnhumby LGSR dataset contains:
- ~2.5 million transactions
- ~2,500 customers
- ~5,000 products
- ~400 stores
- ~2 years of data (117 weeks)

Key files:
- `raw_data/transactions.csv` - Main transaction data
- `raw_data/time.csv` - Temporal metadata

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RetailSim Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Section 2          Section 3          Section 4          Section 5    │
│  ──────────         ──────────         ──────────         ──────────   │
│  Data Pipeline  →   Feature Eng.   →   Tensor Prep   →   Training     │
│                                                                         │
│  • Prices           • Pseudo-Brands    • T1-T6 Tensors   • Dataset     │
│  • Product Graph    • Price 64d        • Batching        • Model       │
│  • Affinity         • Graph 256d       • Masking         • Losses      │
│  • Missions         • Customer 160d                      • Trainer     │
│                     • Store 96d                          • Evaluate    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Six Tensors (T1-T6)

The model receives six input tensors per shopping trip:

| Tensor | Name | Dimension | Description |
|--------|------|-----------|-------------|
| T1 | Customer Context | 192 | Customer history, preferences, demographics |
| T2 | Product Embeddings | S × 256 | Product characteristics (per item in basket) |
| T3 | Temporal Context | 64 | When: day of week, hour, season, holidays |
| T4 | Price Features | S × 64 | Price signals (per item in basket) |
| T5 | Store Context | 96 | Store format, location, assortment |
| T6 | Trip Context | 48 | Shopping mission type, basket size, price sensitivity |

**Dense context**: T1 + T3 + T5 + T6 = 400 dimensions
**Sequence features**: T2 + T4 = 320 dimensions per product

---

## World Model Architecture

```
                    INPUTS
    ┌─────────┐
    │ T1:Cust │──┐
    │  [192]  │  │
    └─────────┘  │    ┌──────────────────┐
    ┌─────────┐  ├───▶│  CONTEXT FUSION  │
    │ T3:Time │──┤    │  [400] → [512]   │
    │  [64]   │  │    └────────┬─────────┘
    └─────────┘  │             │
    ┌─────────┐  │             ▼
    │ T5:Store│──┤    ┌─────────────────────────┐
    │  [96]   │  │    │     MAMBA ENCODER       │
    └─────────┘  │    │     (4 layers)          │
    ┌─────────┐  │    │     O(n) complexity     │
    │ T6:Trip │──┘    │                         │
    │  [48]   │       │  OUTPUT: [(S+1) × 512]  │
    └─────────┘       └───────────┬─────────────┘
                                  │
    ┌─────────┐                   ▼
    │T2:Prods │──┐    ┌─────────────────────────┐
    │[S×256]  │  │    │   TRANSFORMER DECODER   │
    └─────────┘  ├───▶│   (2 layers)            │
    ┌─────────┐  │    │   Cross-attention       │
    │T4:Price │──┘    │                         │
    │[S×64]   │       │  OUTPUT: Product logits │
    └─────────┘       │          [B × 5003]     │
                      └─────────────────────────┘
```

**Model size**: ~23M parameters

### Why Mamba + Transformer?

- **Mamba Encoder**: O(n) State Space Model - efficient for long sequences (customer histories can span 100+ weeks)
- **Transformer Decoder**: Cross-attention allows "asking questions" to the encoder about the customer state

---

## Training Strategy

### Three-Phase Training (20 epochs total)

| Phase | Epochs | Learning Rate | Mask % | Focus |
|-------|--------|---------------|--------|-------|
| Warmup | 1-3 | 1e-5 | 15% | Basic patterns, focal loss only |
| Main | 4-15 | 5e-5 | 15% | Full multi-task learning |
| Finetune | 16-20 | 1e-5 | 20% | Harder examples, polish |

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Focal Loss | 60% | Product prediction (handles class imbalance) |
| Contrastive Loss | 20% | Learn product relationships (co-purchase similarity) |
| Auxiliary Losses | 20% | Basket size, price sensitivity, mission type |

### Temporal Split

```
Week 1 ────────────────────> Week 80 ───> Week 95 ───> Week 117
│                                  │           │            │
│◄────── TRAINING DATA ───────────►│◄─ VALID ─►│◄── TEST ──►│
```

No data leakage: model only sees past data when predicting.

---

## Directory Structure

```
retail_sim/
├── raw_data/
│   ├── transactions.csv      # Dunnhumby LGSR dataset
│   └── time.csv              # Temporal metadata
│
├── data/
│   ├── processed/            # Section 2 outputs
│   │   ├── prices_derived.parquet
│   │   ├── product_graph.pkl
│   │   ├── customer_store_affinity.parquet
│   │   └── customer_mission_patterns.parquet
│   │
│   ├── features/             # Section 3 outputs
│   │   ├── pseudo_brands.parquet
│   │   ├── price_features.parquet
│   │   ├── product_embeddings.pkl
│   │   ├── customer_embeddings.parquet
│   │   └── store_features.parquet
│   │
│   └── training/             # Section 5 outputs
│       ├── samples_train.parquet
│       ├── samples_val.parquet
│       ├── samples_test.parquet
│       └── cache/
│           ├── customer_embeddings.npy
│           ├── store_embeddings.npy
│           └── product_embeddings.npy
│
├── src/
│   ├── data_pipeline/        # Section 2
│   │   ├── stage1_price_derivation.py
│   │   ├── stage2_product_graph.py
│   │   ├── stage3_customer_store_affinity.py
│   │   ├── stage4_mission_patterns.py
│   │   └── run_pipeline.py
│   │
│   ├── feature_engineering/  # Section 3
│   │   ├── layer1_pseudo_brand.py
│   │   ├── layer2_fourier_price.py
│   │   ├── layer3_graph_embeddings.py
│   │   ├── layer4_customer_history.py
│   │   ├── layer5_store_context.py
│   │   └── run_feature_engineering.py
│   │
│   ├── tensor_preparation/   # Section 4
│   │   ├── t1_customer_context.py
│   │   ├── t2_product_sequence.py
│   │   ├── t3_temporal_context.py
│   │   ├── t4_price_context.py
│   │   ├── t5_store_context.py
│   │   ├── t6_trip_context.py
│   │   ├── dataset.py
│   │   └── run_tensor_preparation.py
│   │
│   └── training/             # Section 5
│       ├── prepare_samples.py
│       ├── prepare_tensor_cache.py
│       ├── dataset.py        # WorldModelDataset, DataLoader
│       ├── model.py          # Mamba + Transformer architecture
│       ├── losses.py         # Focal, Contrastive, Multi-task
│       ├── train.py          # TrainingConfig, Trainer
│       ├── evaluate.py       # Evaluator, metrics
│       ├── __init__.py
│       └── README.md
│
├── tests/
│   ├── test_data_pipeline/
│   ├── test_feature_engineering/
│   ├── test_tensor_preparation/
│   └── test_training/
│       ├── test_losses.py    # 19 tests
│       ├── test_model.py     # 29 tests
│       └── test_train.py     # 15 tests
│
├── docs/
│   ├── index.md
│   ├── section2_data_pipeline.md
│   ├── section3_feature_engineering.md
│   ├── section4.0_tensor_preparation.md
│   ├── section4.1_tensor_prep_seq_dia.md
│   ├── section5.0_training_data_architecture.md
│   ├── section5.1_training.md
│   ├── section5.2_training_deep_dive.md
│   └── PROJECT_CONTEXT.md    # This file
│
├── checkpoints/              # Saved models
├── pyproject.toml
└── README.md
```

---

## Key Implementation Details

### 1. Cold-Start Handling (T1)

For customers with limited history:
- New customers (< 5 trips): 80% weight on demographic segments
- Established customers: Gradual shift to behavioral features

```python
# Adaptive blending
blend_weight = min(1.0, num_trips / 20)  # 0-1 over 20 trips
customer_embedding = (1 - blend_weight) * segment_embedding + blend_weight * history_embedding
```

### 2. Attribute-Heavy Store Encoding (T5)

Prevents overfitting with limited customers per store:
- 80d from attributes (format, region, operations)
- 16d from store identity

### 3. Fourier Price Encoding (T4)

Captures price patterns using sine/cosine waves:
- Base price level
- Promotion frequency
- Current price position
- Volatility signals

### 4. BERT-Style Masking

```python
# 15% of basket items masked
for each masked position:
    80% → replace with [MASK] token
    10% → replace with random product
    10% → keep original (but still predict)
```

### 5. Bucket Batching

Group similar-length sequences together to minimize padding:
```python
buckets = {
    'small': (1, 10),
    'medium': (11, 25),
    'large': (26, 50),
    'xlarge': (51, max_seq_len)
}
```

---

## Module Exports

### src/training/__init__.py

```python
from .prepare_samples import enhance_temporal_metadata
from .prepare_tensor_cache import prepare_tensor_cache
from .dataset import WorldModelDataset, WorldModelDataLoader, EvaluationDataLoader
from .model import WorldModel, WorldModelConfig, create_world_model
from .losses import FocalLoss, ContrastiveLoss, WorldModelLoss
from .train import TrainingConfig, Trainer
from .evaluate import Evaluator, EvaluationMetrics, run_evaluation
```

---

## Running the Full Pipeline

```bash
# 1. Data Pipeline (5 seconds)
python -m src.data_pipeline.run_pipeline --nrows 10000

# 2. Feature Engineering (60 seconds)
python -m src.feature_engineering.run_feature_engineering --nrows 10000

# 3. Tensor Preparation (2 seconds)
python -m src.tensor_preparation.run_tensor_preparation

# 4. Prepare Training Samples
python -m src.training.prepare_samples

# 5. Prepare Tensor Cache
python -m src.training.prepare_tensor_cache

# 6. Train
python -m src.training.train --epochs 20 --batch-size 256

# 7. Evaluate
python -m src.training.evaluate checkpoints/best_model.pt --detailed
```

---

## Test Coverage

All 63 tests passing:

| Module | Tests | Description |
|--------|-------|-------------|
| test_losses.py | 19 | FocalLoss, ContrastiveLoss, AuxiliaryLoss, WorldModelLoss |
| test_model.py | 29 | ContextFusion, ProductFusion, MambaBlock, MambaEncoder, TransformerDecoder, OutputHeads, WorldModel |
| test_train.py | 15 | TrainingConfig, Trainer, Checkpointing, Phase transitions |

Run tests:
```bash
pytest tests/test_training/ -v
```

---

## Design Decisions and Rationale

### Why Mamba over pure Transformer?

1. **Efficiency**: O(n) vs O(n²) - critical for long customer histories
2. **Memory**: Linear memory usage allows processing longer sequences
3. **Selective State Space**: Can "choose" what to remember (important events vs routine)

### Why Focal Loss?

With 5000+ products and only ~10 per basket:
- 99.8% of predictions are "don't buy"
- Standard cross-entropy would just learn "predict nothing"
- Focal loss down-weights easy negatives, focuses on hard positives

### Why Contrastive Loss?

Products bought together should have similar embeddings:
- Milk ↔ Bread (often together) → similar
- Pasta ↔ Pasta Sauce (frequently paired) → similar
- Wine ↔ Nappies (rarely together) → different

This creates a meaningful product embedding space.

### Why Auxiliary Tasks?

Forces the model to understand context holistically:
- "This is a large weekly shop" → different predictions than "quick top-up"
- Multi-task learning improves main task performance

---

## Current State

### Implemented

- [x] Section 2: Full data pipeline (4 stages)
- [x] Section 3: Feature engineering (5 layers)
- [x] Section 4: Tensor preparation (T1-T6)
- [x] Section 5: Training infrastructure
  - [x] WorldModelDataset and DataLoader
  - [x] WorldModel (Mamba + Transformer)
  - [x] Loss functions (Focal, Contrastive, Auxiliary)
  - [x] TrainingConfig and Trainer
  - [x] Evaluator and metrics
- [x] Tests: 63 tests passing
- [x] Documentation: Comprehensive docs in /docs

### Future Work (Not Yet Implemented)

- [ ] Reinforcement learning for sequential decisions
- [ ] Counterfactual simulation capabilities
- [ ] Price optimization experiments
- [ ] Real-time inference API
- [ ] Model serving infrastructure

---

## Key Files Quick Reference

| Purpose | File |
|---------|------|
| Main model | `src/training/model.py` |
| Training loop | `src/training/train.py` |
| Loss functions | `src/training/losses.py` |
| Dataset/DataLoader | `src/training/dataset.py` |
| Evaluation | `src/training/evaluate.py` |
| Model tests | `tests/test_training/test_model.py` |
| Training tests | `tests/test_training/test_train.py` |
| Deep dive docs | `docs/section5.2_training_deep_dive.md` |

---

## Dependencies

### Required
- Python >= 3.10
- torch >= 2.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyarrow >= 14.0.0
- scipy >= 1.11.0
- networkx >= 3.0

### Optional
- jupyter >= 1.0.0 (notebook support)

---

## Session Notes

This context document was created to preserve project knowledge across Claude Code sessions. When starting a new session:

1. Read this file first: `docs/PROJECT_CONTEXT.md`
2. Check `docs/index.md` for documentation structure
3. Run tests to verify everything works: `pytest tests/ -v`
4. Check `src/training/__init__.py` for available exports

---

*Last updated: November 2024*
*Design spec version: RetailSim_Data_Pipeline_and_World_Model_Design.md v7.6*
