# Store Visit Prediction (Stage 1)

This document explains the **store visit prediction** model, which is **Stage 1** of the two-stage world model for retail simulation.

## Table of Contents

1. [Why Two-Stage Prediction?](#why-two-stage-prediction)
2. [Architecture Overview](#architecture-overview)
3. [Model Details](#model-details)
4. [Loss Function and Metrics](#loss-function-and-metrics)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation](#evaluation)
7. [Using the Model](#using-the-model)
8. [Integration with Next-Basket Prediction](#integration-with-next-basket-prediction)
9. [Troubleshooting](#troubleshooting)

---

## Why Two-Stage Prediction?

### The Simulation Problem

When simulating customer behavior for RL, we need to answer two sequential questions:

```
Question 1: WHERE will the customer shop next?
    → Store Visit Prediction (this document)

Question 2: WHAT will they buy at that store?
    → Next-Basket Prediction (section5.2.2)
```

### Why Not Joint Prediction?

A single model could theoretically predict both store and basket together, but separating them offers advantages:

| Approach | Pros | Cons |
|----------|------|------|
| **Joint Model** | End-to-end optimization | Complex, hard to debug, larger |
| **Two-Stage** | Modular, interpretable, debuggable | Sequential inference |

The two-stage approach allows:
- **Independent evaluation**: Measure store prediction accuracy separately
- **Modular updates**: Improve one stage without retraining the other
- **Interpretability**: Understand why the model chose a specific store
- **Flexibility**: Use different architectures for each stage

### Store Choice Matters

Store prediction is crucial because:
- **Product availability** varies by store (not all stores carry all products)
- **Prices differ** across stores (same product, different prices)
- **Customer behavior changes** by store (bulk store vs convenience store)
- **Store-specific promotions** affect basket composition

---

## Architecture Overview

```
                    TWO-STAGE WORLD MODEL
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   STAGE 1: Store Visit Prediction                               │
    │   ════════════════════════════════                              │
    │                                                                 │
    │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
    │   │ Customer  │  │ Temporal  │  │  Previous │  │  Basket   │   │
    │   │  Context  │  │  Context  │  │   Store   │  │  Summary  │   │
    │   │  [B,192]  │  │  [B, 64]  │  │  [B, 96]  │  │  [B, 64]  │   │
    │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
    │         │              │              │              │          │
    │         └──────────────┴──────────────┴──────────────┘          │
    │                              │                                   │
    │                              ▼                                   │
    │                   ┌─────────────────────┐                       │
    │                   │        MLP          │                       │
    │                   │   (2 layers, 256d)  │                       │
    │                   └──────────┬──────────┘                       │
    │                              │                                   │
    │                              ▼                                   │
    │                   ┌─────────────────────┐                       │
    │                   │   Store Classifier  │                       │
    │                   │     [B, 761]        │                       │
    │                   │   P(store|context)  │                       │
    │                   └──────────┬──────────┘                       │
    │                              │                                   │
    │                              ▼                                   │
    │                      Predicted Store                             │
    │                              │                                   │
    └──────────────────────────────┼───────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   STAGE 2: Next-Basket Prediction                               │
    │   ═══════════════════════════════                               │
    │                                                                 │
    │   Input: Customer + Time + Predicted Store + Previous Basket    │
    │   Output: P(product) for each product                           │
    │                                                                 │
    │   (See section5.2.2_next_basket_prediction.md)                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Model Details

### Input Features

| Feature | Dimension | Source | Description |
|---------|-----------|--------|-------------|
| Customer Context | 192 | T1 | Static (64d) + History (96d) + Affinity (32d) |
| Temporal Context | 64 | T3 | When is the next visit (week, day, hour, season) |
| Previous Store | 96 | Embedding | Learnable embedding for last visited store |
| Basket Summary | 64 | Optional | Pooled embedding of previous basket products |

**Total input dimension**: 416 (with basket) or 352 (without)

### Model Architecture

```python
@dataclass
class StoreVisitModelConfig:
    # Input dimensions
    customer_dim: int = 192      # T1: Customer context
    temporal_dim: int = 64       # T3: Temporal context
    store_dim: int = 96          # T5: Store embedding
    basket_summary_dim: int = 64 # Summary of previous basket

    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    # Output
    num_stores: int = 761  # Actual store count from data

    # Training
    use_basket_summary: bool = True
```

### BasketSummarizer Module

When `use_basket_summary=True`, the model includes a basket summarizer:

```python
class BasketSummarizer(nn.Module):
    """
    Summarizes previous basket into a fixed-size vector.
    Useful for capturing what was purchased (which may influence store choice).
    """

    def forward(
        self,
        product_embeddings: Tensor,  # [B, S, 256]
        attention_mask: Tensor,       # [B, S]
    ) -> Tensor:
        # Project and pool
        projected = self.proj(product_embeddings)  # [B, S, 64]

        # Masked mean pooling
        pooled = (projected * mask).sum(dim=1) / mask.sum(dim=1)

        return self.norm(pooled)  # [B, 64]
```

### Parameter Count

| Configuration | Parameters |
|---------------|------------|
| With basket summary | ~459K |
| Without basket summary | ~395K |

Much smaller than the next-basket model (~15M) because:
- Classification task (761 classes) vs multi-label (5000 products)
- No transformer encoder needed
- Simple MLP sufficient for store prediction

---

## Loss Function and Metrics

### Cross-Entropy Loss

Store visit prediction is a **multi-class classification** problem:

```python
class StoreVisitLoss(nn.Module):
    """
    Cross-entropy loss for store visit prediction.

    Supports:
    - Class weights for imbalanced stores
    - Label smoothing for regularization
    """

    def __init__(
        self,
        num_stores: int,
        class_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        ...

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
```

### Class Imbalance

Store visits are imbalanced (some stores more popular than others):

```
Store Visit Distribution (example):
═══════════════════════════════════════════════════════════════════════

Popular stores (top 10%):    60% of visits
Medium stores (middle 50%):  35% of visits
Rare stores (bottom 40%):    5% of visits
```

**Solution**: Class weights using inverse frequency with smoothing:

```python
def compute_store_class_weights(
    store_counts: Dict[int, int],
    num_stores: int,
    smoothing: float = 0.3,  # Lower = more aggressive weighting
) -> Tensor:
    """
    weight_i = (total / count_i) ^ smoothing

    smoothing=0: Equal weights
    smoothing=1: Full inverse frequency (aggressive)
    smoothing=0.3: Moderate upweighting of rare stores
    """
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Exact match (top-1) | Higher is better |
| **Top-3 Accuracy** | Target in top 3 predictions | Higher is better |
| **Top-5 Accuracy** | Target in top 5 predictions | Higher is better |
| **Top-10 Accuracy** | Target in top 10 predictions | Higher is better |
| **MRR** | Mean Reciprocal Rank | Higher is better |

### Expected Metrics

```
Store Visit Prediction (typical):
═══════════════════════════════════════════════════════════════════════

Random baseline (761 stores):
    Accuracy:    0.13%
    Top-5:       0.66%
    MRR:         0.003

After training:
    Accuracy:    60-80%   (customers often repeat stores)
    Top-3:       85-92%
    Top-5:       90-95%
    MRR:         0.70-0.85
```

**Why is accuracy so high?**
- Many customers are **loyal** to 1-2 stores
- Temporal patterns (weekday store vs weekend store)
- The model learns these preferences quickly

---

## Training Pipeline

### Step 1: Ensure Data is Prepared

The store visit model uses the same samples as next-basket prediction:

```bash
# Generate next-basket samples (if not already done)
python -m src.data_preparation.stage4_next_basket_samples
```

This creates:
- `data/prepared/train_next_basket.parquet`
- `data/prepared/validation_next_basket.parquet`
- `data/prepared/test_next_basket.parquet`

Each sample includes `input_store_id` (previous store) and `target_store_id` (next store).

### Step 2: Train the Model

```bash
# Basic training
python -m src.training.train_store_visit --epochs 10 --batch-size 128

# Full options
python -m src.training.train_store_visit \
    --epochs 10 \
    --batch-size 128 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --num-layers 2 \
    --label-smoothing 0.1 \
    --class-weight-smoothing 0.3 \
    --device auto
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--weight-decay` | 0.01 | L2 regularization |
| `--hidden-dim` | 256 | MLP hidden dimension |
| `--num-layers` | 2 | MLP layers |
| `--dropout` | 0.1 | Dropout rate |
| `--label-smoothing` | 0.1 | Label smoothing factor |
| `--no-class-weights` | False | Disable class weights |
| `--class-weight-smoothing` | 0.3 | Class weight smoothing |
| `--no-basket` | False | Disable basket summary |
| `--eval-every` | 1 | Validate every N epochs |
| `--save-every` | 1 | Save checkpoint every N epochs |
| `--device` | auto | Device (auto/cuda/mps/cpu) |

### Step 3: Monitor Training

Expected log output:

```
2024-01-15 10:30:00 - INFO - Using device: mps
2024-01-15 10:30:05 - INFO - Train: 12,034,799 samples, 94022 batches
2024-01-15 10:30:05 - INFO - Val: 2,193,028 samples, 17133 batches
2024-01-15 10:30:05 - INFO - Number of stores: 761
2024-01-15 10:30:05 - INFO - Model parameters: 458,873
2024-01-15 10:30:05 - INFO - Class weights: min=0.64, max=2.89
2024-01-15 10:30:05 - INFO - Starting training for 10 epochs

Epoch 1 | Step 100/94022 | Loss: 4.2341 | Acc: 0.1523 | Speed: 2100 samples/sec
Epoch 1 | Step 200/94022 | Loss: 3.1234 | Acc: 0.3421 | Speed: 2150 samples/sec
...
Epoch 1 | Train Loss: 2.1234 | Train Acc: 0.5234 | Val Loss: 1.8923 | Val Acc: 0.6234 | Top-3: 0.8521 | MRR: 0.7123
New best accuracy: 0.6234

Epoch 5 | Train Loss: 1.2345 | Train Acc: 0.7234 | Val Loss: 1.1234 | Val Acc: 0.7523 | Top-3: 0.9123 | MRR: 0.8234
...
```

### Checkpoints

Saved to `models/store_visit/`:
- `best_model.pt` - Best validation accuracy
- `checkpoint_epoch_N.pt` - Periodic checkpoints
- `final_model.pt` - Final model

---

## Evaluation

### Run Evaluation

```bash
# Evaluate on validation set
python -m src.training.evaluate_store_visit \
    --checkpoint models/store_visit/best_model.pt \
    --split validation

# Evaluate on test set
python -m src.training.evaluate_store_visit \
    --checkpoint models/store_visit/best_model.pt \
    --split test

# Show example predictions
python -m src.training.evaluate_store_visit \
    --checkpoint models/store_visit/best_model.pt \
    --examples 10
```

### Output

```
============================================================
STORE VISIT PREDICTION EVALUATION (validation)
============================================================

Samples evaluated: 2,193,028

--- Overall Metrics ---
  Loss:           1.1234
  Accuracy:       0.7523 (75.23%)
  Top-3 Accuracy: 0.9123 (91.23%)
  Top-5 Accuracy: 0.9456 (94.56%)
  Top-10 Accuracy:0.9712 (97.12%)
  MRR:            0.8234

--- Store Transition Analysis ---
  Same store visits:      65.2% of data
    Accuracy:             0.9512
  Different store visits: 34.8% of data
    Accuracy:             0.5823

--- Per-Store Summary ---
  Stores evaluated:       761
  Mean per-store acc:     0.6823
  Median per-store acc:   0.7123
  Min accuracy:           0.1234
  Max accuracy:           0.9876
  Stores with 0% acc:     12

--- Top Confusions ---
  Predicted Store_A instead of Store_B: 1,234 times
  Predicted Store_C instead of Store_D: 987 times
  ...
```

### Key Insights

**Same vs Different Store Analysis:**
- High accuracy on "same store" visits → Model learns customer loyalty
- Lower accuracy on "different store" visits → Harder to predict store switches
- This is expected: store switching is inherently less predictable

---

## Using the Model

### Loading a Trained Model

```python
import torch
from src.training.model_store_visit import StoreVisitPredictor, StoreVisitModelConfig

# Load checkpoint
checkpoint = torch.load('models/store_visit/best_model.pt')

# Reconstruct config
config = StoreVisitModelConfig(**checkpoint['config'])

# Create and load model
model = StoreVisitPredictor(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get vocabulary for index-to-store mapping
vocabulary = checkpoint['vocabulary']
idx_to_store = vocabulary['idx_to_store']
```

### Predicting Store Visits

```python
# Prepare inputs
customer_context = torch.randn(1, 192)  # From dataset
temporal_context = torch.randn(1, 64)    # When is next visit
previous_store_idx = torch.tensor([42])  # Last visited store index

# Get predictions
with torch.no_grad():
    outputs = model(
        customer_context=customer_context,
        temporal_context=temporal_context,
        previous_store_idx=previous_store_idx,
    )

# Get probabilities
store_probs = outputs['store_probs']  # [1, 761]

# Top-5 predictions
top5_probs, top5_indices = store_probs.topk(5, dim=-1)
for i in range(5):
    store_id = idx_to_store[str(top5_indices[0, i].item())]
    prob = top5_probs[0, i].item()
    print(f"  {store_id}: {prob:.4f}")
```

### Sampling a Store

```python
# Deterministic (argmax)
predicted_store = model.predict_store(
    customer_context, temporal_context, previous_store_idx, top_k=1
)[0]  # Returns top-1 store index

# Probabilistic (for simulation diversity)
sampled_store = model.sample_store(
    customer_context, temporal_context, previous_store_idx,
    temperature=1.0  # Higher = more random
)
```

---

## Integration with Next-Basket Prediction

### Sequential Inference

```python
from src.training.model_store_visit import StoreVisitPredictor
from src.training.model_next_basket import NextBasketWorldModel

# Load both models
store_model = load_store_model('models/store_visit/best_model.pt')
basket_model = load_basket_model('models/next_basket/best_model.pt')

# Stage 1: Predict store
store_probs = store_model(customer_context, temporal_context, prev_store_idx)
predicted_store_idx = store_probs['store_probs'].argmax(dim=-1)

# Get store embedding for Stage 2
store_embedding = store_embeddings[predicted_store_idx]  # [B, 96]

# Stage 2: Predict basket at that store
basket_probs = basket_model(
    input_embeddings=prev_basket_embeddings,
    input_price_features=price_features,
    input_attention_mask=attention_mask,
    customer_context=customer_context,
    temporal_context=temporal_context,
    store_context=store_embedding,  # <-- From Stage 1
    trip_context=trip_context,
)

# Get predicted products
predicted_products = (basket_probs > 0.5).float()
```

### Full Simulation Loop

```python
class RetailSimulator:
    def __init__(self, store_model, basket_model, store_embeddings):
        self.store_model = store_model
        self.basket_model = basket_model
        self.store_embeddings = store_embeddings

    def simulate_visit(self, customer_state, time_context):
        """Simulate a customer's next shopping trip."""

        # Stage 1: Where will they shop?
        store_idx = self.store_model.sample_store(
            customer_context=customer_state['context'],
            temporal_context=time_context,
            previous_store_idx=customer_state['last_store'],
            temperature=1.0,
        )

        # Stage 2: What will they buy?
        store_emb = self.store_embeddings[store_idx]
        basket_probs = self.basket_model.get_product_probabilities(
            input_embeddings=customer_state['last_basket_emb'],
            input_price_features=customer_state['price_features'],
            input_attention_mask=customer_state['attention_mask'],
            customer_context=customer_state['context'],
            temporal_context=time_context,
            store_context=store_emb,
            trip_context=customer_state['trip_context'],
        )

        # Sample basket
        basket = torch.bernoulli(basket_probs)

        return {
            'store': store_idx,
            'basket': basket,
            'store_probs': store_probs,
            'basket_probs': basket_probs,
        }
```

---

## Troubleshooting

### Issue: Low Accuracy (< 50% after epoch 3)

**Possible causes:**
1. Learning rate too high → Try `--lr 5e-5`
2. Not using class weights → Remove `--no-class-weights`
3. Label smoothing too high → Try `--label-smoothing 0.05`

### Issue: Accuracy Stuck at ~65%

**This is normal!** Around 65% of visits are to the same store as last time. The model quickly learns this pattern. To improve:
1. Focus on "different store" accuracy
2. Add more temporal features
3. Increase model capacity: `--hidden-dim 512 --num-layers 3`

### Issue: Poor Performance on Rare Stores

**Solutions:**
1. Increase class weight smoothing: `--class-weight-smoothing 0.5`
2. More aggressive upweighting helps rare stores
3. Consider grouping very rare stores

### Issue: Out of Memory

**Solutions:**
1. Reduce batch size: `--batch-size 64`
2. Disable basket summary: `--no-basket`
3. Model is small (~459K params), memory usually not an issue

### Issue: Training Too Slow

**Solutions:**
1. Increase batch size (if memory allows): `--batch-size 256`
2. Use GPU: `--device cuda` or `--device mps`
3. Disable basket summary for faster training: `--no-basket`

---

## Files Reference

| Purpose | File |
|---------|------|
| Model definition | `src/training/model_store_visit.py` |
| Dataset | `src/training/dataset_store_visit.py` |
| Loss + Metrics | `src/training/losses_store_visit.py` |
| Training script | `src/training/train_store_visit.py` |
| Evaluation script | `src/training/evaluate_store_visit.py` |
| Unit tests | `tests/test_training/test_store_visit.py` |

---

## Summary

| Aspect | Store Visit (Stage 1) | Next-Basket (Stage 2) |
|--------|----------------------|----------------------|
| **Question** | WHERE will they shop? | WHAT will they buy? |
| **Output** | 761-class softmax | 5000-product sigmoid |
| **Loss** | Cross-entropy | Focal BCE |
| **Parameters** | ~459K | ~15M |
| **Key Metric** | Accuracy, MRR | F1@k, NDCG |
| **Training Time** | ~1 hour (MPS) | ~10 hours (MPS) |

The two-stage approach provides a complete world model for retail simulation:
1. **Stage 1** predicts the store (fast, accurate)
2. **Stage 2** predicts the basket conditioned on that store (detailed, comprehensive)

---

*See also:*
- [Next-Basket Prediction (Stage 2)](section5.2.2_next_basket_prediction.md)
- [Training Deep Dive](section5.2.1_training_deep_dive.md)
- [Training Data Architecture](section5.0_training_data_architecture.md)
