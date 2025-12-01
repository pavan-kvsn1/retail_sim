# RetailSim

Data Pipeline & World Model for Retail Transaction Simulation based on the Dunnhumby "Let's Get Sort of Real" (LGSR) dataset.

## Overview

RetailSim implements a comprehensive data processing and feature engineering pipeline for retail transaction data, designed to support world model training for customer behavior simulation.

## Architecture

```
RetailSim Architecture
├── Section 2: Data Pipeline
│   ├── Stage 1: Price Derivation
│   ├── Stage 2: Product Graph Construction
│   ├── Stage 3: Customer-Store Affinity
│   └── Stage 4: Mission Pattern Extraction
│
├── Section 3: Feature Engineering
│   ├── Layer 1: Pseudo-Brand Inference
│   ├── Layer 2: Fourier Price Encoding (64d)
│   ├── Layer 3: Graph Embeddings - GraphSAGE (256d)
│   ├── Layer 4: Customer History Encoding (160d)
│   └── Layer 5: Store Context Features (96d)
│
├── Section 4: Tensor Preparation
│   ├── T1: Customer Context [192d]
│   ├── T2: Product Sequence [256d/item]
│   ├── T3: Temporal Context [64d]
│   ├── T4: Price Context [64d/item]
│   ├── T5: Store Context [96d]
│   └── T6: Trip Context [48d]
│
└── Section 5: World Model Training
    ├── Input Processing Layer
    ├── Mamba Encoder (4 layers, ~23M params)
    ├── Transformer Decoder (2 layers)
    ├── Multi-Task Output Heads
    └── BERT-style Masked Product Modeling
```

## Installation

### Using Poetry (Recommended)

```bash
# Install dependencies
poetry install

# With PyTorch support
poetry install --extras torch

# With Jupyter notebook support
poetry install --extras notebooks

# With all optional dependencies
poetry install --extras all
```

### Using pip

```bash
pip install pandas numpy networkx pyarrow scipy
# Optional: pip install torch jupyter
```

## Usage

### 1. Data Pipeline (Section 2)

```bash
# Run with sample data (10k rows)
poetry run retail-data-pipeline --nrows 10000

# Or directly
python src/data_pipeline/run_pipeline.py --nrows 10000
```

**Outputs** (in `data/processed/`):
- `prices_derived.parquet` - Derived prices with promotions
- `product_graph.pkl` - Heterogeneous product graph
- `customer_store_affinity.parquet` - Customer loyalty metrics
- `customer_mission_patterns.parquet` - Shopping behavior patterns

### 2. Feature Engineering (Section 3)

```bash
# Run feature engineering (requires data pipeline outputs)
poetry run retail-feature-engineering --nrows 10000

# Or directly
python src/feature_engineering/run_feature_engineering.py --nrows 10000
```

**Outputs** (in `data/features/`):
- `pseudo_brands.parquet` - Inferred brand clusters
- `price_features.parquet` - 64d Fourier price encoding
- `product_embeddings.pkl` - 256d GraphSAGE embeddings
- `customer_history_embeddings.pkl` - 160d customer embeddings
- `store_features.parquet` - 96d store context

### 3. Tensor Preparation (Section 4)

```bash
# Build training tensors from features
poetry run python src/tensor_preparation/build_tensors.py
```

**Outputs** (in `data/tensors/`):
- `train_tensors.pt` - Training set (~80%)
- `val_tensors.pt` - Validation set (~10%)
- `test_tensors.pt` - Test set (~10%)

### 4. World Model Training (Section 5)

```bash
# Basic training
poetry run python src/training/train.py --epochs 20 --batch-size 128

# With gradient checkpointing (reduces memory)
poetry run python src/training/train.py --epochs 20 --batch-size 128 --gradient-checkpointing

# On Apple Silicon GPU
poetry run python src/training/train.py --device mps --batch-size 128

# Resume from checkpoint
poetry run python src/training/train.py --resume checkpoints/latest.pt
```

**Key arguments:**
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 128, reduce if OOM)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: cpu/cuda/mps (default: auto-detect)
- `--gradient-checkpointing`: Enable memory-efficient training
- `--eval-every`: Validation frequency in steps (default: 500)

**Outputs** (in `checkpoints/`):
- `best.pt` - Best model by validation loss
- `latest.pt` - Most recent checkpoint
- `epoch_N.pt` - Per-epoch checkpoints

## Project Structure

```
retail_sim/
├── pyproject.toml          # Poetry configuration
├── README.md
├── raw_data/               # Raw LGSR dataset (not tracked)
│   ├── transactions.csv
│   └── time.csv
├── data/                   # Processed outputs (not tracked)
│   ├── processed/          # Data pipeline outputs
│   ├── features/           # Feature engineering outputs
│   └── tensors/            # Training tensors
├── checkpoints/            # Model checkpoints (not tracked)
├── logs/                   # Training logs (not tracked)
├── src/
│   ├── data_pipeline/      # Section 2: Data Pipeline
│   │   ├── stage1_price_derivation.py
│   │   ├── stage2_product_graph.py
│   │   ├── stage3_customer_store_affinity.py
│   │   ├── stage4_mission_patterns.py
│   │   └── run_pipeline.py
│   ├── feature_engineering/ # Section 3: Feature Engineering
│   │   ├── layer1_pseudo_brand.py
│   │   ├── layer2_fourier_price.py
│   │   ├── layer3_graph_embeddings.py
│   │   ├── layer4_customer_history.py
│   │   ├── layer5_store_context.py
│   │   └── run_feature_engineering.py
│   ├── tensor_preparation/  # Section 4: Tensor Building
│   │   └── build_tensors.py
│   ├── training/            # Section 5: World Model
│   │   ├── model.py         # WorldModel architecture
│   │   ├── dataset.py       # TripDataset with bucket batching
│   │   ├── losses.py        # Focal, Contrastive, Auxiliary losses
│   │   └── train.py         # Training loop
│   └── utils/
├── docs/                    # Design documentation
├── tests/
└── configs/
```

## Development

```bash
# Install with dev dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black src/
poetry run ruff check src/ --fix

# Type checking
poetry run mypy src/
```

## Data Requirements

This project uses the Dunnhumby "Let's Get Sort of Real" (LGSR) dataset. Place the following files in `raw_data/`:
- `transactions.csv` - Transaction records
- `time.csv` - Week calendar mapping

## License

MIT
