# Next-Basket Prediction for RL/Simulation

This document explains the **next-basket prediction** training paradigm, which is the recommended approach for building a world model suitable for reinforcement learning and simulation.

## Table of Contents

1. [Why Next-Basket Prediction?](#why-next-basket-prediction)
2. [Masked vs Next-Basket: The Key Difference](#masked-vs-next-basket-the-key-difference)
3. [Architecture Overview](#architecture-overview)
4. [Loss Function: Focal BCE](#loss-function-focal-bce)
5. [Learning Rate Scheduling](#learning-rate-scheduling)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Using the Model for RL](#using-the-model-for-rl)
9. [Troubleshooting](#troubleshooting)

---

## Why Next-Basket Prediction?

### The Problem with Masked Prediction

The original masked prediction (BERT-style) approach learns:
- "What products co-occur in the same basket?"
- Reconstructs missing items in the **current** basket

This is useful for learning product embeddings, but it doesn't answer the key question for simulation:

> **"What will the customer buy on their NEXT visit?"**

### What RL Needs

In a reinforcement learning setting:

```
State:      Customer history + current basket + context
Action:     Which products to recommend/promote/stock
Transition: What the customer actually buys next ← THIS IS THE WORLD MODEL
Reward:     Revenue, margin, customer satisfaction
```

The world model must predict **future behavior**, not reconstruct current behavior.

---

## Masked vs Next-Basket: The Key Difference

```
MASKED PREDICTION (section5.2):
═══════════════════════════════════════════════════════════════════════

    Time t basket: [milk, cheese, bread, apples, eggs]
                         ↓
    Mask 15%:      [milk, [MASK], bread, [MASK], eggs]
                         ↓
    Predict:       [milk, cheese, bread, apples, eggs]  ← SAME basket

    Loss: Cross-entropy per masked position
    Learns: Product co-occurrence within a single trip


NEXT-BASKET PREDICTION (this document):
═══════════════════════════════════════════════════════════════════════

    Time t basket:   [milk, cheese, bread, apples, eggs]  ← FULL input
    Customer history: [...all previous baskets...]
    Context t+1:     Store, day, hour for next visit
                         ↓
    Predict:         [yogurt, bananas, bread, butter, ...]  ← NEXT basket

    Loss: Multi-label BCE (sigmoid per product)
    Learns: What customer will buy on their next trip
```

### Architecture Comparison

| Aspect | Masked Prediction | Next-Basket Prediction |
|--------|-------------------|------------------------|
| **Input** | Current basket (masked) | Full current basket |
| **Target** | Masked tokens only | Entire next basket |
| **Output Layer** | Softmax per position | Sigmoid per product |
| **Loss Function** | Cross-entropy | Focal BCE |
| **Primary Metrics** | Token accuracy | P@k, R@k, F1@k, NDCG |
| **Parameters** | ~23M | ~15M |
| **Use Case** | Embeddings, co-occurrence | RL, simulation |

---

## Architecture Overview

```
                        NEXT-BASKET WORLD MODEL
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   INPUTS (all from time t)                                      │
    │   ════════════════════════                                      │
    │                                                                 │
    │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
    │   │ Input Basket │  │  Customer   │  │  Context    │            │
    │   │ [B, S, 320]  │  │   [B, 192]  │  │ t+1 [B,208] │            │
    │   │ (products +  │  │  (history)  │  │(store,time) │            │
    │   │   prices)    │  │             │  │             │            │
    │   └──────┬───────┘  └──────┬──────┘  └──────┬──────┘            │
    │          │                 │                │                   │
    │          ▼                 ▼                ▼                   │
    │   ┌─────────────────────────────────────────────────────┐      │
    │   │              TRANSFORMER ENCODER                     │      │
    │   │              (4 layers, 8 heads)                     │      │
    │   │                                                      │      │
    │   │   • Processes input basket sequence                  │      │
    │   │   • Incorporates context via concatenation           │      │
    │   │   • Pools to single [B, 512] representation          │      │
    │   └──────────────────────┬──────────────────────────────┘      │
    │                          │                                      │
    │                          ▼                                      │
    │   ┌─────────────────────────────────────────────────────┐      │
    │   │              BASKET PREDICTOR                        │      │
    │   │              (MLP + Output Heads)                    │      │
    │   │                                                      │      │
    │   │   Main:  [B, 512] → [B, V] product logits           │      │
    │   │   Aux:   Basket size, mission type, etc.            │      │
    │   └──────────────────────┬──────────────────────────────┘      │
    │                          │                                      │
    │                          ▼                                      │
    │   ┌─────────────────────────────────────────────────────┐      │
    │   │              OUTPUT                                  │      │
    │   │                                                      │      │
    │   │   product_probs: [B, V] ← sigmoid(logits)           │      │
    │   │                                                      │      │
    │   │   P(product_i in next basket | current state)       │      │
    │   └─────────────────────────────────────────────────────┘      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### Model Size

- **Parameters**: ~15M (smaller than masked model)
- **Hidden dimension**: 512
- **Encoder layers**: 4
- **Vocabulary**: ~5000 products

---

## Loss Function: Focal BCE

### The Class Imbalance Problem

Next-basket prediction faces extreme class imbalance:

```
Vocabulary size:     ~5,000 products
Typical basket size: ~10 products
Positive rate:       10 / 5000 = 0.2%

For each prediction:
    ✓ Positives (bought):     ~10 products
    ✗ Negatives (not bought): ~4,990 products
```

Standard BCE would be dominated by easy negatives ("customer won't buy caviar" is trivially correct).

### Why Focal BCE?

| Challenge | How Focal BCE Solves It |
|-----------|-------------------------|
| **Multi-label output** | BCE with sigmoid per product (not softmax) |
| **99.8% negatives** | `pos_weight` upweights rare positive examples |
| **Easy negatives dominate** | Focal term `(1-p)^γ` down-weights confident predictions |

### The Math

```
Standard BCE:
────────────────────────────────────────────────────────────────────
    L = -[y · log(p) + (1-y) · log(1-p)]

    Where:
        y = target (0 or 1)
        p = predicted probability


Focal BCE:
────────────────────────────────────────────────────────────────────
    L = -[y · α · (1-p)^γ · log(p) + (1-y) · p^γ · log(1-p)]

    Where:
        γ = 2.0   → focusing parameter
        α = 10.0  → positive class weight

    Key insight:
        • When p is high and y=0: p^γ shrinks the loss (easy negative)
        • When p is low and y=1: (1-p)^γ ≈ 1 (hard positive, full loss)
```

### Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gamma` | 2.0 | Higher = more focus on hard examples. Range: [1.0, 5.0] |
| `pos_weight` | 10.0 | Upweights positive class. Approximate: vocab_size / avg_basket_size |

### Implementation

```python
class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross-Entropy Loss for multi-label classification.

    Handles extreme class imbalance where most products are NOT purchased.
    Down-weights easy negatives, focuses learning on hard examples.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: [B, V] raw scores (pre-sigmoid)
            targets: [B, V] multi-hot target (0 or 1 per product)

        Returns:
            Scalar loss value
        """
        probs = torch.sigmoid(logits)

        # Focal weights
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Class weights
        alpha = torch.where(targets == 1, self.pos_weight, 1.0)

        # BCE with focal modulation
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        loss = alpha * focal_weight * bce

        return loss.mean()
```

### Loss Comparison: Masked vs Next-Basket

| Aspect | Masked Prediction | Next-Basket Prediction |
|--------|-------------------|------------------------|
| **Loss Type** | Cross-Entropy | Focal BCE |
| **Output Activation** | Softmax (per position) | Sigmoid (per product) |
| **Target Format** | Single product ID per position | Multi-hot vector [B, V] |
| **Challenge Addressed** | Large vocabulary | Extreme class imbalance |
| **Typical Loss Range** | 6.0 → 4.0 | 1.5 → 0.8 |

### Expected Loss Progression

```
Training Loss (typical):
═══════════════════════════════════════════════════════════════════════

Epoch 1:   Loss: 1.50 → 1.20  (learning basic patterns)
Epoch 5:   Loss: 1.00 → 0.90  (refining predictions)
Epoch 10:  Loss: 0.85 → 0.80  (convergence begins)
Epoch 20:  Loss: 0.75 → 0.70  (fine-tuning)

Warning signs:
    • Loss stuck > 1.2 after epoch 5 → learning rate too low
    • Loss < 0.3 → possible overfitting, check val loss
    • Loss oscillating wildly → learning rate too high
```

---

## Learning Rate Scheduling

Proper learning rate scheduling is critical for stable training and optimal convergence. We support three scheduling strategies.

### Available Schedulers

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| `cosine` (default) | Linear warmup + cosine decay | Pretraining from scratch |
| `onecycle` | Super-convergence schedule | Fast training, limited compute |
| `cosine_restarts` | Cosine with periodic restarts | Finetuning, escaping local minima |

### 1. Cosine Schedule (Recommended for Pretraining)

```
LR
 │
 │     ╭────────╮
 │    ╱          ╲
 │   ╱            ╲
 │  ╱              ╲
 │ ╱                ╲____
 │╱                      ─────
 └─────────────────────────────► Steps
   warmup      cosine decay
   (10%)          (90%)
```

**How it works:**
1. **Warmup phase** (10% of steps): Linear increase from 0 to `max_lr`
2. **Decay phase** (90% of steps): Cosine decay from `max_lr` to `min_lr`

```bash
python -m src.training.train_next_basket \
    --scheduler cosine \
    --warmup-ratio 0.1 \
    --min-lr-ratio 0.01 \
    --lr 1e-4
```

**Why warmup?**
- Prevents early gradient explosion
- Allows optimizer to calibrate momentum estimates
- Critical for transformer models with attention

### 2. OneCycleLR (Fast Training)

```
LR
 │
 │         ╭───╮
 │        ╱     ╲
 │       ╱       ╲
 │      ╱         ╲
 │     ╱           ╲
 │    ╱             ╲___
 │___╱                   ─────
 └─────────────────────────────► Steps
   warmup    peak    annealing
```

**How it works:**
- Starts low, ramps up to peak, then decays
- Uses super-convergence principles (Smith & Topin, 2019)
- Can train in fewer epochs with higher LR

```bash
python -m src.training.train_next_basket \
    --scheduler onecycle \
    --lr 3e-4  # Can use higher peak LR
```

### 3. Cosine with Warm Restarts (Finetuning)

```
LR
 │
 │ ╭─╮     ╭───╮       ╭───────╮
 │ │  ╲   ╱     ╲     ╱         ╲
 │ │   ╲ ╱       ╲   ╱           ╲
 │ │    ╳         ╲ ╱             ╲
 │ │   ╱ ╲         ╳               ╲
 │ │  ╱   ╲       ╱ ╲               ╲___
 └─┴─────────────────────────────────────► Steps
    T_0   2*T_0      4*T_0
```

**How it works:**
- Cosine decay with periodic "restarts" back to max LR
- Each cycle is 2x longer than the previous (T_mult=2)
- Helps escape local minima during finetuning

```bash
python -m src.training.train_next_basket \
    --scheduler cosine_restarts \
    --lr 5e-5  # Lower LR for finetuning
```

**Best for:**
- Finetuning on new data
- When training gets stuck
- Continued training after initial convergence

### Recommended Settings by Task

| Task | Scheduler | Peak LR | Warmup | Min LR Ratio |
|------|-----------|---------|--------|--------------|
| **Pretraining (scratch)** | `cosine` | 1e-4 | 10% | 0.01 |
| **Fast pretraining** | `onecycle` | 3e-4 | 10% | 0.01 |
| **Finetuning (new data)** | `cosine_restarts` | 5e-5 | 5% | 0.01 |
| **Continued training** | `cosine` | 5e-5 | 5% | 0.001 |

### The Math

**Linear Warmup:**
```
lr(step) = max_lr * (step / warmup_steps)
```

**Cosine Decay:**
```
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr(step) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
```

**Cosine with Restarts:**
```
T_cur = step mod T_i           # Position within current cycle
T_i = T_0 * T_mult^i           # Length of cycle i
lr(step) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * T_cur / T_i))
```

### Monitoring LR During Training

The current learning rate is logged every 100 steps:

```
Step 100 | Loss: 1.45 | LR: 1.00e-05 | Speed: 35.2 samples/s
Step 200 | Loss: 1.32 | LR: 2.00e-05 | Speed: 36.1 samples/s
...
Step 1000 | Loss: 1.10 | LR: 1.00e-04 | Speed: 35.8 samples/s  ← Peak LR
Step 2000 | Loss: 0.95 | LR: 9.50e-05 | Speed: 35.5 samples/s  ← Starting decay
```

---

## Training Pipeline

### Step 1: Generate Next-Basket Samples

Creates (basket_t, basket_t+1) pairs from customer timelines:

```bash
python -m src.data_preparation.stage4_next_basket_samples
```

**What this does:**
- For each customer, extracts consecutive basket pairs
- Includes full history before time t
- Filters by time gap (max 12 weeks between visits)
- Splits by target basket time (train/val/test)

**Output files** (in `data/prepared/`):
- `train_next_basket.parquet`
- `validation_next_basket.parquet`
- `test_next_basket.parquet`

**Typical sample counts:**
- Train: ~15M pairs
- Validation: ~2M pairs
- Test: ~2M pairs

### Step 2: Train the Model

```bash
python -m src.training.train_next_basket \
    --epochs 20 \
    --batch-size 64 \
    --device mps \
    --gradient-checkpointing
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--batch-size` | 64 | Batch size (smaller due to multi-label loss memory) |
| `--lr` | 1e-4 | Peak learning rate |
| `--scheduler` | cosine | LR schedule: `cosine`, `onecycle`, `cosine_restarts` |
| `--warmup-ratio` | 0.1 | Fraction of steps for warmup |
| `--min-lr-ratio` | 0.01 | Minimum LR as fraction of peak LR |
| `--hidden-dim` | 512 | Model hidden dimension |
| `--encoder-layers` | 4 | Transformer encoder layers |
| `--device` | auto | cpu/cuda/mps |
| `--gradient-checkpointing` | off | Enable for memory efficiency |
| `--eval-every` | 500 | Validation frequency (steps) |

### Step 3: Monitor Training

Expected log output:

```
Step 100 | Loss: 1.4523 | Product: 1.2341 | Speed: 35.2 samples/s
>>> Validation | Loss: 1.3892 | P@10: 0.0312 | R@10: 0.0621 | F1@10: 0.0415 | HR@10: 0.2341
Step 200 | Loss: 1.3124 | Product: 1.1023 | Speed: 36.1 samples/s
...
>>> Validation | Loss: 0.9823 | P@10: 0.0892 | R@10: 0.1521 | F1@10: 0.1124 | HR@10: 0.4521
New best F1@10: 0.1124
```

---

## Evaluation Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision@k** | Of top-k predictions, how many were bought? | Higher is better |
| **Recall@k** | Of items bought, how many were in top-k? | Higher is better |
| **F1@k** | Harmonic mean of P@k and R@k | Higher is better |
| **Hit Rate@k** | Did at least one prediction match? | Higher is better |
| **NDCG@k** | Ranking quality (relevant items ranked higher) | Higher is better |

### Expected Progression

```
Training Progression (typical):
═══════════════════════════════════════════════════════════════════════

Initial (random):
    P@10: 0.002    R@10: 0.005    F1@10: 0.003    HR@10: 0.02

Epoch 5:
    P@10: 0.08     R@10: 0.15     F1@10: 0.10     HR@10: 0.45

Epoch 10:
    P@10: 0.12     R@10: 0.20     F1@10: 0.15     HR@10: 0.55

Epoch 20:
    P@10: 0.15     R@10: 0.25     F1@10: 0.18     HR@10: 0.62

Good model (cloud GPU):
    P@10: 0.20+    R@10: 0.30+    F1@10: 0.24+    HR@10: 0.70+
```

### Interpreting Metrics

- **F1@10 > 0.15**: Model is learning meaningful patterns
- **F1@10 > 0.20**: Good for simulation use
- **F1@10 > 0.25**: Excellent, production-ready
- **Hit Rate > 0.60**: Majority of predictions have at least one correct item

---

## Using the Model for RL

### Loading a Trained Model

```python
import torch
from src.training import NextBasketWorldModel, NextBasketModelConfig

# Create model with same config as training
config = NextBasketModelConfig(
    vocab_size=5000,
    hidden_dim=512,
    encoder_layers=4,
)
model = NextBasketWorldModel(config)

# Load checkpoint
checkpoint = torch.load('checkpoints/next_basket_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model with F1@10: {checkpoint['metrics']['f1@10']:.4f}")
```

### Getting Product Probabilities

```python
# Prepare inputs (these come from your dataset/environment)
input_embeddings = ...      # [B, S, 256] - current basket products
input_price_features = ...  # [B, S, 64] - current basket prices
input_attention_mask = ...  # [B, S] - valid positions
customer_context = ...      # [B, 192] - customer embedding
temporal_context = ...      # [B, 64] - time of NEXT visit (t+1)
store_context = ...         # [B, 96] - store of NEXT visit
trip_context = ...          # [B, 48] - expected trip type

# Get probability distribution
with torch.no_grad():
    probs = model.get_product_probabilities(
        input_embeddings=input_embeddings,
        input_price_features=input_price_features,
        input_attention_mask=input_attention_mask,
        customer_context=customer_context,
        temporal_context=temporal_context,
        store_context=store_context,
        trip_context=trip_context,
    )

# probs is [B, vocab_size] - probability of each product in next basket
# probs[0, 123] = P(product 123 in next basket | current state)
```

### Sampling a Next Basket

```python
# Method 1: Threshold
threshold = 0.5
predicted_basket = (probs > threshold).float()  # [B, V] binary

# Method 2: Top-k
k = 10
_, top_k_indices = probs.topk(k, dim=-1)  # [B, k] indices

# Method 3: Probabilistic sampling (for simulation diversity)
predicted_basket = torch.bernoulli(probs)  # [B, V] sampled
```

### RL Environment Integration

```python
class RetailEnvironment:
    def __init__(self, world_model):
        self.world_model = world_model

    def step(self, action):
        """
        action: Promotion/recommendation decisions
        returns: next_state, reward, done, info
        """
        # Apply action (e.g., modify prices, show recommendations)
        modified_context = self.apply_action(action)

        # Use world model to predict customer response
        next_basket_probs = self.world_model.get_product_probabilities(
            input_embeddings=self.current_basket_emb,
            input_price_features=modified_context['prices'],
            input_attention_mask=self.attention_mask,
            customer_context=self.customer_emb,
            temporal_context=modified_context['time'],
            store_context=self.store_emb,
            trip_context=modified_context['trip'],
        )

        # Sample next basket (or use expected value)
        next_basket = torch.bernoulli(next_basket_probs)

        # Calculate reward
        reward = self.calculate_reward(next_basket, action)

        # Update state
        self.current_basket_emb = self.encode_basket(next_basket)

        return self.get_state(), reward, False, {}
```

---

## Troubleshooting

### Issue: Low F1@10 (< 0.05 after epoch 5)

**Possible causes:**
1. Learning rate too high → Try `--lr 5e-5`
2. Batch size too small → Try `--batch-size 128`
3. Not enough training data → Check sample counts

### Issue: Out of Memory

**Solutions:**
1. Enable gradient checkpointing: `--gradient-checkpointing`
2. Reduce batch size: `--batch-size 32`
3. Reduce hidden dimension: `--hidden-dim 256`

### Issue: Training Too Slow

**Solutions:**
1. Use GPU: `--device cuda` or `--device mps`
2. Increase batch size (if memory allows)
3. Reduce validation frequency: `--eval-every 1000`

### Issue: Validation Metrics Not Improving

**Possible causes:**
1. Learning rate too low → Try `--lr 5e-4`
2. Model too small → Try `--hidden-dim 768`
3. Overfitting → Add dropout or reduce model size

---

## Files Reference

| Purpose | File |
|---------|------|
| Sample generation | `src/data_preparation/stage4_next_basket_samples.py` |
| Dataset | `src/training/dataset_next_basket.py` |
| Model | `src/training/model_next_basket.py` |
| Loss + Metrics | `src/training/losses_next_basket.py` |
| Training script | `src/training/train_next_basket.py` |
| Tests | `tests/test_training/test_next_basket.py` |

---

## Next Steps

Once you have a trained next-basket model:

1. **Build RL Environment**: Wrap the model as a gym-style environment
2. **Train RL Agent**: Use PPO, SAC, or other algorithms
3. **Counterfactual Analysis**: "What if we promoted product X?"
4. **A/B Test Simulation**: Compare strategies before live deployment

---

*See also: [section5.2_training_deep_dive.md](section5.2_training_deep_dive.md) for masked prediction training*
