# World Model Training Module

This module provides the complete training pipeline for the RetailSim World Model, a hybrid Mamba-Transformer architecture for retail basket prediction.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          WORLD MODEL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐                                   │
│  │ T1: Customer │    │ T2: Products │                                   │
│  │    [192d]    │    │  [B, S, 256] │                                   │
│  └──────┬───────┘    └──────┬───────┘                                   │
│         │                   │                                            │
│  ┌──────┴───────┐    ┌──────┴───────┐                                   │
│  │ T3: Temporal │    │ T4: Price    │                                   │
│  │    [64d]     │    │  [B, S, 64]  │                                   │
│  └──────┬───────┘    └──────┬───────┘                                   │
│         │                   │                                            │
│  ┌──────┴───────┐           │                                           │
│  │ T5: Store    │           │                                           │
│  │    [96d]     │           │                                           │
│  └──────┬───────┘           │                                           │
│         │                   │                                            │
│  ┌──────┴───────┐           │                                           │
│  │ T6: Trip     │           │                                           │
│  │    [48d]     │           │                                           │
│  └──────┬───────┘           │                                           │
│         │                   │                                            │
│         v                   v                                            │
│  ┌──────────────┐    ┌──────────────┐                                   │
│  │Context Fusion│    │Product Fusion│                                   │
│  │  [400→512]   │    │ [320→512]    │                                   │
│  └──────┬───────┘    └──────┬───────┘                                   │
│         │                   │                                            │
│         └────────┬──────────┘                                           │
│                  │                                                       │
│                  v                                                       │
│         ┌───────────────────┐                                           │
│         │   MAMBA ENCODER   │                                           │
│         │   (4 layers)      │                                           │
│         │   O(n) complexity │                                           │
│         └─────────┬─────────┘                                           │
│                   │                                                      │
│                   v                                                      │
│         ┌───────────────────┐                                           │
│         │TRANSFORMER DECODER│                                           │
│         │   (2 layers)      │                                           │
│         │ Cross-attention   │                                           │
│         └─────────┬─────────┘                                           │
│                   │                                                      │
│                   v                                                      │
│         ┌───────────────────┐                                           │
│         │   OUTPUT HEADS    │                                           │
│         │ • Masked Products │                                           │
│         │ • Basket Size     │                                           │
│         │ • Price Sens.     │                                           │
│         │ • Mission Type    │                                           │
│         └───────────────────┘                                           │
│                                                                          │
│  Total Parameters: ~23M                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/training/
├── __init__.py           # Package exports
├── dataset.py            # WorldModelDataset, DataLoader
├── losses.py             # FocalLoss, ContrastiveLoss, WorldModelLoss
├── model.py              # WorldModel architecture
├── train.py              # Training loop
├── evaluate.py           # Evaluation metrics
├── prepare_samples.py    # Data preparation (Stage 1)
├── prepare_tensor_cache.py  # Tensor cache (Stage 2)
```

## Quick Start

### 1. Data Preparation

```bash
# Stage 1: Prepare temporal split samples
python -m src.training.prepare_samples

# Stage 2: Prepare tensor cache
python -m src.training.prepare_tensor_cache
```

### 2. Training

```bash
# Basic training
python -m src.training.train

# With custom parameters
python -m src.training.train \
    --batch-size 256 \
    --epochs 20 \
    --lr 5e-5

# Resume from checkpoint
python -m src.training.train \
    --resume checkpoints/checkpoint_epoch_10.pt
```

### 3. Evaluation

```bash
# Evaluate on test set
python -m src.training.evaluate checkpoints/best_model.pt \
    --split test \
    --output results/test_results.json

# Quick evaluation (limited batches)
python -m src.training.evaluate checkpoints/best_model.pt \
    --max-batches 100
```

## Training Configuration

### Three-Phase Training Schedule

| Phase | Epochs | Learning Rate | Mask Prob | Loss Components |
|-------|--------|---------------|-----------|-----------------|
| Warm-up | 1-3 | 1e-5 | 15% | Focal only |
| Main | 4-15 | 5e-5 | 15% | All tasks |
| Fine-tune | 16-20 | 1e-5 | 20% | All tasks |

### Loss Weights (Main/Fine-tune Phase)

| Component | Weight | Purpose |
|-----------|--------|---------|
| Focal Loss | 0.60 | Masked product prediction |
| Contrastive Loss | 0.20 | Product co-occurrence |
| Basket Size | 0.08 | Size prediction (S/M/L) |
| Price Sensitivity | 0.08 | Sensitivity prediction |
| Mission Type | 0.04 | Trip classification |

## API Reference

### WorldModel

```python
from src.training import WorldModel, WorldModelConfig

config = WorldModelConfig(
    d_model=512,
    n_products=5003,
    mamba_num_layers=4,
    decoder_num_layers=2
)

model = WorldModel(config)
```

### Training

```python
from src.training import TrainingConfig, Trainer

config = TrainingConfig(
    batch_size=256,
    num_epochs=20,
    learning_rate=5e-5,
    device='cuda'
)

trainer = Trainer(config)
trainer.train()
```

### Evaluation

```python
from src.training.evaluate import Evaluator, load_model_from_checkpoint

model = load_model_from_checkpoint('checkpoints/best_model.pt', device)
evaluator = Evaluator(model, device)

# Overall metrics
metrics = evaluator.evaluate_dataset(test_dataset)

# Per-bucket (cold-start analysis)
bucket_metrics = evaluator.evaluate_by_bucket(test_dataset)
```

## Metrics

### Masked Product Prediction

- **Accuracy**: Top-1 accuracy on masked positions
- **Precision@K**: Hit rate in top-K predictions (K=1,5,10)
- **Recall@K**: Coverage of true products in top-K
- **MRR**: Mean Reciprocal Rank

### Auxiliary Tasks

- **Basket Size Accuracy**: S/M/L classification
- **Price Sensitivity Accuracy**: LA/MM/UM classification
- **Mission Type Accuracy**: Top-up/Full Shop/etc.

## Hardware Requirements

### Single GPU (Recommended)

- GPU: NVIDIA A100 40GB or similar
- Memory: ~5 GB per batch (batch_size=256)
- Training time: ~40 hours for 20 epochs
- Storage: 27 GB (data + checkpoints)

### Apple Silicon (MPS)

- Device: M1/M2/M3 with 16GB+ RAM
- Training: Supported but slower
- Mixed precision: Not available

## Checkpoints

Checkpoints are saved to `checkpoints/` with the following:

```python
{
    'epoch': int,
    'global_step': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'best_val_loss': float,
    'config': dict
}
```

Best model is saved as `checkpoints/best_model.pt`.

## Testing

```bash
# Run all training tests
pytest tests/test_training/ -v

# Run specific test file
pytest tests/test_training/test_model.py -v

# Run with coverage
pytest tests/test_training/ --cov=src/training
```

## References

- Design Document: `RetailSim_Data_Pipeline_and_World_Model_Design.md`
- Mamba Paper: [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- Focal Loss Paper: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
