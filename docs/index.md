# RetailSim Documentation

RetailSim is a data pipeline and world model framework for retail transaction simulation, built on the Dunnhumby Let's Get Sort of Real (LGSR) dataset.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RetailSim Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Section 2  │    │   Section 3  │    │   Section 4  │              │
│  │    Data      │───▶│   Feature    │───▶│    Tensor    │───▶ Model   │
│  │   Pipeline   │    │  Engineering │    │  Preparation │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ • Prices     │    │ • Pseudo-    │    │ • T1-T6      │              │
│  │ • Graph      │    │   Brands     │    │   Tensors    │              │
│  │ • Affinity   │    │ • Price 64d  │    │ • Dataset    │              │
│  │ • Missions   │    │ • Graph 256d │    │ • DataLoader │              │
│  │              │    │ • Cust 160d  │    │              │              │
│  │              │    │ • Store 96d  │    │              │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
cd retail_sim

# Install with Poetry
poetry install

# Or with pip
pip install -e .
```

### Running the Full Pipeline

```bash
# Step 1: Data Pipeline
python -m src.data_pipeline.run_pipeline --nrows 10000

# Step 2: Feature Engineering
python -m src.feature_engineering.run_feature_engineering --nrows 10000

# Step 3: Tensor Preparation
python -m src.tensor_preparation.run_tensor_preparation
```

### Using the DataLoader

```python
from pathlib import Path
from src.tensor_preparation import RetailSimDataset, RetailSimDataLoader

# Initialize dataset
dataset = RetailSimDataset(project_root=Path('.'))

# Create dataloader
dataloader = RetailSimDataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    apply_masking=True  # For BERT-style training
)

# Training loop
for batch in dataloader:
    dense_context = batch.get_dense_context()      # [B, 400]
    seq_features = batch.get_sequence_features()   # [B, S, 320]
    attention_mask = batch.attention_mask          # [B, S]

    # Your model training here...
```

## Documentation Sections

### [Section 2: Data Pipeline](section2_data_pipeline.md)

Processes raw transaction data through four stages:

| Stage | Output | Description |
|-------|--------|-------------|
| 1 | `prices_derived.parquet` | Unit prices with waterfall imputation |
| 2 | `product_graph.pkl` | Heterogeneous product graph |
| 3 | `customer_store_affinity.parquet` | Loyalty and switching metrics |
| 4 | `customer_mission_patterns.parquet` | Shopping mission distributions |

### [Section 3: Feature Engineering](section3_feature_engineering.md)

Transforms data into rich embeddings across five layers:

| Layer | Output | Dimension |
|-------|--------|-----------|
| 1 | Pseudo-Brands | categorical |
| 2 | Price Features | 64d |
| 3 | Product Embeddings | 256d |
| 4 | Customer History | 160d |
| 5 | Store Context | 96d |

### [Section 4: Tensor Preparation](section4.0_tensor_preparation.md)

Assembles features into model-ready tensors:

| Tensor | Dimension | Type |
|--------|-----------|------|
| T1 | 192d | Dense (Customer) |
| T2 | 256d/item | Sequence (Products) |
| T3 | 64d | Dense (Temporal) |
| T4 | 64d/item | Sequence (Prices) |
| T5 | 96d | Dense (Store) |
| T6 | 48d | Dense (Trip) |

**Combined**: 400d dense context + 320d/item sequence features

### [Section 5: Training](section5.0_training_data_architecture.md)

World Model training infrastructure:

| Component | Description |
|-----------|-------------|
| [Data Architecture](section5.0_training_data_architecture.md) | Temporal splits, bucket batching |
| [Training Pipeline](section5.1_training.md) | Model, losses, training loop |
| [Deep Dive: Masked Prediction](section5.2.1_training_deep_dive.md) | Detailed explanation of tensors and data flow |
| [Next-Basket Prediction](section5.2.2_next_basket_prediction.md) | Stage 2: What will they buy? |
| [Store Visit Prediction](section5.2.3_store_visit_prediction.md) | **Stage 1: Where will they shop?** |

### [Section 6: End-to-End Data Flow](section6_data_flow_end_to_end.md)

Complete data flow documentation from raw data to training:

| Topic | Description |
|-------|-------------|
| Pipeline Overview | Visual flow from transactions.csv to models |
| Data Pipeline Stages | Prices, graph, affinity, missions |
| Feature Engineering | 5 layers producing embeddings |
| Tensor Specifications | T1-T6 dimensions with examples |
| Training Data Preparation | Temporal splits, next-basket samples |
| Model Input/Output | Exact dimensions for both models |
| Two-Stage Inference | How models work together |

**Two-Stage World Model (Recommended for RL/Simulation):**
```
Stage 1: Store Visit Prediction (~459K params)
    Input:  Customer + Time + Previous Store
    Output: P(next_store) → Which store will they visit?

Stage 2: Next-Basket Prediction (~15M params)
    Input:  Customer + Time + Predicted Store + Previous Basket
    Output: P(products) → What will they buy at that store?
```

**Alternative Training Paradigm:**
- **Masked Prediction** (~23M params): BERT-style, good for embeddings

## Project Structure

```
retail_sim/
├── raw_data/
│   ├── transactions.csv      # Dunnhumby LGSR dataset
│   └── time.csv
├── data/
│   ├── processed/            # Section 2 outputs
│   │   ├── prices_derived.parquet
│   │   ├── product_graph.pkl
│   │   ├── customer_store_affinity.parquet
│   │   └── customer_mission_patterns.parquet
│   └── features/             # Section 3 outputs
│       ├── pseudo_brands.parquet
│       ├── price_features.parquet
│       ├── product_embeddings.pkl
│       ├── customer_embeddings.parquet
│       └── store_features.parquet
├── src/
│   ├── data_pipeline/        # Section 2
│   │   ├── stage1_price_derivation.py
│   │   ├── stage2_product_graph.py
│   │   ├── stage3_customer_store_affinity.py
│   │   ├── stage4_mission_patterns.py
│   │   └── run_pipeline.py
│   ├── feature_engineering/  # Section 3
│   │   ├── layer1_pseudo_brand.py
│   │   ├── layer2_fourier_price.py
│   │   ├── layer3_graph_embeddings.py
│   │   ├── layer4_customer_history.py
│   │   ├── layer5_store_context.py
│   │   └── run_feature_engineering.py
│   ├── tensor_preparation/   # Section 4
│   │   ├── t1_customer_context.py
│   │   ├── t2_product_sequence.py
│   │   ├── t3_temporal_context.py
│   │   ├── t4_price_context.py
│   │   ├── t5_store_context.py
│   │   ├── t6_trip_context.py
│   │   ├── dataset.py
│   │   └── run_tensor_preparation.py
│   └── training/             # Section 5
│       ├── model.py          # Masked prediction model
│       ├── model_store_visit.py    # Stage 1: Store visit prediction
│       ├── model_next_basket.py    # Stage 2: Next-basket prediction
│       ├── dataset_store_visit.py  # Store visit dataset
│       ├── dataset_next_basket.py  # Next-basket dataset
│       ├── losses_store_visit.py   # Store visit loss/metrics
│       ├── losses_next_basket.py   # Next-basket loss/metrics
│       ├── train_store_visit.py    # Train Stage 1
│       ├── train_next_basket.py    # Train Stage 2
│       ├── evaluate_store_visit.py # Evaluate Stage 1
│       └── evaluate.py       # Evaluation metrics
├── docs/                     # Documentation
│   ├── index.md
│   ├── section2_data_pipeline.md
│   ├── section3_feature_engineering.md
│   ├── section4.0_tensor_preparation.md
│   ├── section5.0_training_data_architecture.md
│   ├── section5.1_training.md
│   ├── section5.2.1_training_deep_dive.md
│   ├── section5.2.2_next_basket_prediction.md
│   └── section5.2.3_store_visit_prediction.md
├── pyproject.toml
└── README.md
```

## Key Features

### Cold-Start Handling
Adaptive blending for customers with limited history:
- New customers (< 5 trips): 80% weight on demographic segments
- Established customers: Gradual shift to behavioral features

### BERT-Style Training
Masked Event Modeling for self-supervised pre-training:
- 15% of basket items masked
- 80% replaced with [MASK], 10% random, 10% unchanged
- Enables learning product relationships

### Multi-Task Learning
T6 (Trip Context) serves dual purpose:
- Input: Conditions basket generation on mission type
- Output: Auxiliary prediction target for mission classification

### Attribute-Heavy Store Encoding
Prevents overfitting with limited customers per store:
- 80d from attributes (format, region, operations)
- 16d from store identity
- Captures generalizable patterns over memorization

## Dependencies

### Required
- Python >= 3.10
- pandas >= 2.0.0
- numpy >= 1.24.0
- networkx >= 3.0
- pyarrow >= 14.0.0
- scipy >= 1.11.0

### Optional
- torch >= 2.0.0 (GPU-accelerated GraphSAGE)
- jupyter >= 1.0.0 (notebook support)

## Performance

Processing 10,000 transactions:

| Stage | Time |
|-------|------|
| Data Pipeline | ~5s |
| Feature Engineering | ~60s |
| Tensor Preparation | ~2s |

Memory usage scales linearly with transaction count.

## Training the World Model

### Two-Stage Approach (Recommended)

```bash
# Step 1: Prepare training data
python -m src.data_preparation.stage4_next_basket_samples

# Step 2: Train Stage 1 (Store Visit Prediction)
python -m src.training.train_store_visit --epochs 10 --batch-size 128

# Step 3: Train Stage 2 (Next-Basket Prediction)
python -m src.training.train_next_basket --epochs 20 --batch-size 64

# Step 4: Evaluate both stages
python -m src.training.evaluate_store_visit --checkpoint models/store_visit/best_model.pt
python -m src.training.evaluate_next_basket --checkpoint models/next_basket/best_model.pt
```

### Quick Start (Single Model)

```bash
# Prepare training data
python -m src.training.prepare_samples
python -m src.training.prepare_tensor_cache

# Train masked prediction model
python -m src.training.train --epochs 20 --batch-size 256

# Evaluate
python -m src.training.evaluate checkpoints/best_model.pt --split test
```

See [Store Visit Prediction](section5.2.3_store_visit_prediction.md) and [Next-Basket Prediction](section5.2.2_next_basket_prediction.md) for detailed training guides.

## Next Steps

Future work will implement:
- Reinforcement learning for sequential decisions
- Counterfactual simulation capabilities
- Price optimization experiments

## License

MIT License
