# RetailSim Project Context

This document captures the full context of the RetailSim project for session continuity. It provides everything needed to understand the codebase, architecture decisions, and current state.

---

## Project Overview

**RetailSim** is a world model for retail transaction simulation built on the Dunnhumby "Let's Get Sort of Real" (LGSR) dataset. The goal is to predict customer shopping behavior - specifically, what products a customer will buy given their history, the time of shopping, the store they visit, and their shopping mission.

### Core Concept

The model learns to answer: *"Given everything we know about a customer, what products will they buy in their next shopping trip?"*

**Two Training Paradigms:**

1. **Masked Event Modeling** (BERT-style) - for learning embeddings:
   - Take a shopping basket: [Milk, Bread, Eggs, Cheese, Apples]
   - Mask 15% of products: [Milk, [MASK], Eggs, Cheese, [MASK]]
   - Predict the masked products in the SAME basket

2. **Next-Basket Prediction** (Recommended for RL) - for simulation:
   - Take full basket at time t: [Milk, Cheese, Bread]
   - Predict ENTIRE basket at time t+1: [Yogurt, Eggs, Fruit, ...]
   - This is what's needed for RL/simulation environments

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
│       ├── dataset.py           # WorldModelDataset (masked prediction)
│       ├── dataset_next_basket.py  # NextBasketDataset (for RL)
│       ├── model.py             # Mamba + Transformer (masked)
│       ├── model_next_basket.py # Next-basket model (for RL)
│       ├── losses.py            # Focal, Contrastive, Multi-task
│       ├── losses_next_basket.py   # Multi-label BCE + metrics
│       ├── train.py             # TrainingConfig, Trainer (masked)
│       ├── train_next_basket.py # Trainer for next-basket
│       ├── evaluate.py          # Evaluator, metrics
│       └── __init__.py
│
├── tests/
│   ├── test_data_pipeline/
│   ├── test_feature_engineering/
│   ├── test_tensor_preparation/
│   └── test_training/
│       ├── test_losses.py       # 19 tests (masked prediction)
│       ├── test_model.py        # 29 tests (masked model)
│       ├── test_train.py        # 15 tests
│       └── test_next_basket.py  # 27 tests (next-basket pipeline)
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
# Masked prediction (original)
from .dataset import WorldModelDataset, WorldModelDataLoader
from .model import WorldModel, WorldModelConfig
from .losses import FocalLoss, ContrastiveLoss, WorldModelLoss
from .train import TrainingConfig, Trainer

# Next-basket prediction (for RL/simulation)
from .dataset_next_basket import NextBasketDataset, NextBasketDataLoader
from .model_next_basket import NextBasketWorldModel, NextBasketModelConfig
from .losses_next_basket import NextBasketLoss, NextBasketMetrics
```

---

## Running the Full Pipeline

### Masked Prediction (for embeddings)

```bash
# 1-5. Data pipeline, features, tensor prep (same as below)

# 6. Train masked model
python -m src.training.train --epochs 20 --batch-size 128 --device mps

# 7. Evaluate
python -m src.training.evaluate checkpoints/best_model.pt --detailed
```

### Next-Basket Prediction (for RL/simulation) - RECOMMENDED

```bash
# 1. Data Pipeline
python -m src.data_pipeline.run_pipeline

# 2. Feature Engineering
python -m src.feature_engineering.run_feature_engineering

# 3. Data Preparation
python -m src.data_preparation.run_data_preparation

# 4. Generate Next-Basket Samples
python -m src.data_preparation.stage4_next_basket_samples

# 5. Train Next-Basket Model
python -m src.training.train_next_basket \
    --epochs 20 \
    --batch-size 64 \
    --device mps \
    --gradient-checkpointing

# Metrics: Precision@10, Recall@10, F1@10, NDCG@10
```

---

## Test Coverage

All 90 tests passing:

| Module | Tests | Description |
|--------|-------|-------------|
| test_losses.py | 19 | FocalLoss, ContrastiveLoss, AuxiliaryLoss, WorldModelLoss |
| test_model.py | 29 | ContextFusion, ProductFusion, MambaBlock, MambaEncoder, TransformerDecoder |
| test_train.py | 15 | TrainingConfig, Trainer, Checkpointing, Phase transitions |
| test_next_basket.py | 27 | NextBasketWorldModel, FocalBCE, Metrics (P@k, R@k, F1, NDCG) |

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
  - [x] WorldModelDataset and DataLoader (masked prediction)
  - [x] WorldModel (Mamba + Transformer, ~23M params)
  - [x] Loss functions (Focal, Contrastive, Auxiliary)
  - [x] **NextBasketDataset and DataLoader (for RL)**
  - [x] **NextBasketWorldModel (~15M params)**
  - [x] **Next-basket losses (Focal BCE) and metrics (P@k, R@k, F1, NDCG)**
  - [x] TrainingConfig and Trainer (both paradigms)
  - [x] Evaluator and metrics
- [x] Tests: 90 tests passing
- [x] Documentation: Comprehensive docs in /docs

### Future Work (Not Yet Implemented)

- [ ] Reinforcement learning agent using NextBasketWorldModel
- [ ] Counterfactual simulation capabilities
- [ ] Price optimization experiments
- [ ] Real-time inference API
- [ ] Model serving infrastructure

---

## Key Files Quick Reference

### Masked Prediction (for embeddings)
| Purpose | File |
|---------|------|
| Model | `src/training/model.py` |
| Training | `src/training/train.py` |
| Loss | `src/training/losses.py` |
| Dataset | `src/training/dataset.py` |

### Next-Basket Prediction (for RL) - RECOMMENDED
| Purpose | File |
|---------|------|
| Model | `src/training/model_next_basket.py` |
| Training | `src/training/train_next_basket.py` |
| Loss + Metrics | `src/training/losses_next_basket.py` |
| Dataset | `src/training/dataset_next_basket.py` |
| Sample generation | `src/data_preparation/stage4_next_basket_samples.py` |

### Testing
| Purpose | File |
|---------|------|
| Masked tests | `tests/test_training/test_model.py` |
| Next-basket tests | `tests/test_training/test_next_basket.py` |
| Documentation | `docs/section5.2_training_deep_dive.md` |

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

*Last updated: December 2024*
*Design spec version: RetailSim_Data_Pipeline_and_World_Model_Design.md v7.6*
*Major update: Added next-basket prediction pipeline for RL/simulation*
