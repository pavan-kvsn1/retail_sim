# Section 6: End-to-End Data Flow Documentation

This document provides a granular walkthrough of how data flows from raw transactions through all pipeline stages to the training of both world models (Store Visit Prediction and Next-Basket Prediction).

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RetailSim Data Flow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  raw_data/transactions.csv                                                  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────┐                                                       │
│  │  Section 2       │                                                       │
│  │  Data Pipeline   │ → prices_derived.parquet                              │
│  │  (run_pipeline)  │ → product_graph.pkl                                   │
│  │                  │ → customer_store_affinity.parquet                     │
│  │                  │ → customer_mission_patterns.parquet                   │
│  └────────┬─────────┘                                                       │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  Section 3       │                                                       │
│  │  Feature Eng.    │ → pseudo_brands.parquet                               │
│  │  (run_feature_   │ → price_features.parquet (64d per product-week)       │
│  │   engineering)   │ → product_embeddings.pkl (256d per product)           │
│  │                  │ → customer_embeddings.parquet (160d per customer)     │
│  │                  │ → store_features.parquet (96d per store)              │
│  └────────┬─────────┘                                                       │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  Section 4       │                                                       │
│  │  Tensor Prep     │ → Validates T1-T6 tensor dimensions                   │
│  │  (run_tensor_    │ → Tests dataset/dataloader                            │
│  │   preparation)   │                                                       │
│  └────────┬─────────┘                                                       │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  Data Prep       │                                                       │
│  │  (run_data_      │ → temporal_metadata.parquet (splits)                  │
│  │   preparation)   │ → customer_histories.parquet                          │
│  │                  │ → samples/ (bucketed)                                 │
│  │                  │ → tensor_cache/ (.npy files)                          │
│  └────────┬─────────┘                                                       │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  Next-Basket     │ → train_next_basket.parquet                           │
│  │  Sample Creator  │ → validation_next_basket.parquet                      │
│  │  (stage4_next_   │ → test_next_basket.parquet                            │
│  │   basket_samples)│                                                       │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│     ┌─────┴─────┐                                                           │
│     ▼           ▼                                                           │
│  ┌──────────┐  ┌──────────────┐                                             │
│  │ Stage 1  │  │   Stage 2    │                                             │
│  │ Store    │  │  Next-Basket │                                             │
│  │ Visit    │  │  Prediction  │                                             │
│  │ Training │  │  Training    │                                             │
│  └──────────┘  └──────────────┘                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Pipeline (`run_pipeline.py`)

**Input**: `raw_data/transactions.csv`

**Columns Used**:
```python
['PROD_CODE', 'PROD_CODE_10', 'PROD_CODE_20', 'PROD_CODE_30', 'PROD_CODE_40',
 'STORE_CODE', 'STORE_REGION', 'SHOP_WEEK',
 'SPEND', 'QUANTITY', 'CUST_CODE', 'BASKET_ID',
 'BASKET_TYPE', 'BASKET_DOMINANT_MISSION',
 'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE']
```

### Stage 1.1: Price Derivation

**Output**: `data/processed/prices_derived.parquet`

Computes unit prices using waterfall imputation:
1. Direct calculation: `SPEND / QUANTITY`
2. Same product-store-week median
3. Same product-week median
4. Same category median
5. Global product median

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| PROD_CODE | str | Product identifier |
| STORE_CODE | str | Store identifier |
| SHOP_WEEK | int | Week number (YYYYWW format) |
| unit_price | float | Derived unit price |
| price_source | str | Imputation source used |

### Stage 1.2: Product Graph

**Output**: `data/processed/product_graph.pkl`

Builds a heterogeneous NetworkX graph with:
- **Nodes**: Products (with hierarchy attributes)
- **Edges**: Co-purchase relationships (weight = count)

**Edge Weights**: Minimum co-purchase count = 5, top-k complements = 10

### Stage 1.3: Customer-Store Affinity

**Output**: `data/processed/customer_store_affinity.parquet`

Computes per-customer:
- Primary store (most visited)
- Store loyalty score
- Switching frequency
- Store visit distribution

### Stage 1.4: Mission Patterns

**Output**: `data/processed/customer_mission_patterns.parquet`

Per-customer mission distributions:
- Trip type frequencies (Full Shop, Top Up, etc.)
- Price sensitivity patterns
- Basket size distributions

---

## Stage 2: Feature Engineering (`run_feature_engineering.py`)

**Requires**: All Stage 1 outputs

### Layer 1: Pseudo-Brand Inference

**Output**: `data/features/pseudo_brands.parquet`

Clusters products into pseudo-brands based on:
- Co-purchase patterns
- Price similarity
- Category hierarchy

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| PROD_CODE | str | Product identifier |
| pseudo_brand_id | int | Inferred brand cluster |
| brand_confidence | float | Clustering confidence |

### Layer 2: Fourier Price Encoding

**Output**: `data/features/price_features.parquet`

**Dimension**: 64d per (product, store, week)

**Breakdown**:
| Component | Dimension | Description |
|-----------|-----------|-------------|
| Fourier components | 24d | Multi-scale price representation |
| Log price features | 16d | Log-transformed price metrics |
| Relative prices | 16d | Price vs category/store averages |
| Price velocity | 8d | Price change momentum |

**Example** (conceptual):
```python
# For product P001 at store S001 in week 202340:
price_features = {
    'fourier': [0.23, -0.15, 0.08, ...],    # 24 values
    'log': [2.3, 0.5, -0.2, ...],           # 16 values
    'relative': [0.95, 1.02, 0.88, ...],    # 16 values
    'velocity': [0.01, -0.03, 0.02, ...],   # 8 values
}
# Total: 64-dimensional vector
```

### Layer 3: Graph Embeddings (GraphSAGE)

**Output**: `data/features/product_embeddings.pkl`

**Dimension**: 256d per product

Uses GraphSAGE on the product graph:
1. Initialize with pseudo-brand + category features
2. 2-hop neighborhood aggregation
3. Mean pooling over neighbors
4. Final projection to 256d

**Example**:
```python
# Product embeddings dictionary
product_embeddings = {
    'PROD001': np.array([0.12, -0.34, 0.56, ...]),  # 256 floats
    'PROD002': np.array([0.08, 0.22, -0.11, ...]),  # 256 floats
    ...
}
```

### Layer 4: Customer History Encoding

**Output**:
- `data/features/customer_history_embeddings.pkl`
- `data/features/customer_embeddings.parquet`

**Dimension**: 160d per customer

**Breakdown**:
| Component | Dimension | Description |
|-----------|-----------|-------------|
| Behavioral history | 96d | Aggregated purchase patterns |
| Static demographics | 64d | Segment + cold-start features |

**Example**:
```python
# Customer embedding
customer_embedding = {
    'static': np.array([...]),    # 64d: segment encodings
    'history': np.array([...]),   # 96d: purchase behavior
}
# Combined: 160d per customer
```

### Layer 5: Store Context

**Output**: `data/features/store_features.parquet`

**Dimension**: 96d per store

**Breakdown**:
| Component | Dimension | Description |
|-----------|-----------|-------------|
| Format encoding | 16d | Store format (LS, MS, SS) |
| Region encoding | 32d | Geographic region |
| Operational features | 32d | Size, traffic, competition |
| Store identity | 16d | Learnable store embedding |

---

## Stage 3: Tensor Preparation (`run_tensor_preparation.py`)

Tests and validates the tensor encoding pipeline.

### Tensor Specifications

| Tensor | Dimension | Type | Components |
|--------|-----------|------|------------|
| **T1** | 192d | Dense (Customer) | static(64) + history(96) + affinity(32) |
| **T2** | 256d/item | Sequence (Products) | Product embeddings |
| **T3** | 64d | Dense (Temporal) | week(16) + weekday(8) + hour(8) + holiday(8) + season(8) + trend(8) + recency(8) |
| **T4** | 64d/item | Sequence (Prices) | fourier(24) + log(16) + relative(16) + velocity(8) |
| **T5** | 96d | Dense (Store) | format(16) + region(32) + operational(32) + identity(16) |
| **T6** | 48d | Dense (Trip) | mission_type(16) + mission_focus(16) + price_sens(8) + basket_size(8) |

### Combined Dimensions

```
Dense Context: T1 + T3 + T5 + T6 = 192 + 64 + 96 + 48 = 400d
Sequence Features: T2 + T4 = 256 + 64 = 320d per item
```

### T1: Customer Context [192d]

**Encoding Process**:
```python
def encode_customer(customer_id, history_embedding, static_embedding):
    # static_embedding: 64d (from customer segments)
    # history_embedding: 96d (from purchase history)
    # affinity: 32d (category preferences, can be zeros)

    return np.concatenate([
        static_embedding,    # [64]
        history_embedding,   # [96]
        affinity_scores,     # [32]
    ])  # Total: [192]
```

**Example**:
```python
T1 = [
    # Static (64d): Customer segment embeddings
    0.23, -0.15, 0.08, ...,  # Segment 1 encoding
    0.45, 0.12, -0.33, ...,  # Segment 2 encoding

    # History (96d): Aggregated purchase behavior
    0.12, 0.34, -0.22, ...,  # Category affinities
    0.56, 0.11, 0.88, ...,   # Purchase frequency features

    # Affinity (32d): Store/category preferences
    0.80, 0.15, 0.05, ...,   # Store preference distribution
]
```

### T2: Product Sequence [256d/item]

**Encoding Process**:
```python
def encode_product_sequence(product_ids, product_embeddings, max_len=50):
    # Lookup pre-computed 256d embeddings
    embeddings = np.zeros((max_len, 256))
    for i, prod_id in enumerate(product_ids[:max_len]):
        embeddings[i] = product_embeddings.get(prod_id, zeros_256)
    return embeddings  # [S, 256]
```

**Example** (3-item basket):
```python
T2 = [
    # Item 1: Milk
    [0.12, -0.34, 0.56, ..., 0.23],  # 256 values
    # Item 2: Bread
    [0.08, 0.22, -0.11, ..., -0.45], # 256 values
    # Item 3: Eggs
    [0.33, 0.15, 0.67, ..., 0.12],   # 256 values
    # Padding (if max_len > 3)
    [0.00, 0.00, 0.00, ..., 0.00],   # zeros
]
```

### T3: Temporal Context [64d]

**Encoding Process**:
```python
def encode_temporal(shop_week, shop_weekday, shop_hour):
    week_of_year = shop_week % 100  # 1-52

    return np.concatenate([
        week_embed[week_of_year],        # [16]
        weekday_embed[shop_weekday],     # [8]
        hour_embed[shop_hour],           # [8]
        holiday_pattern,                 # [8]
        season_embed[get_season(week)],  # [8]
        trend_embed,                     # [8] (fourier of normalized week)
        recency_embed,                   # [8] (days since last visit)
    ])  # Total: [64]
```

**Example** (Wednesday afternoon in December):
```python
T3 = [
    # Week embedding (16d): Week 50 pattern
    0.45, -0.23, 0.12, ...,

    # Weekday embedding (8d): Wednesday = 3
    0.15, 0.33, -0.22, 0.08, 0.45, -0.11, 0.23, 0.56,

    # Hour embedding (8d): 14:00
    0.34, 0.12, -0.45, 0.67, 0.23, -0.11, 0.08, 0.45,

    # Holiday pattern (8d): Near Christmas
    0.50, 0.00, 0.50, 0.00, 0.50, 0.00, 0.50, 0.00,

    # Season embedding (8d): Winter
    0.23, 0.45, -0.12, 0.34, 0.56, -0.23, 0.11, 0.45,

    # Trend (8d): Position in dataset timeline
    0.12, 0.88, -0.23, 0.77, 0.34, 0.66, -0.11, 0.55,

    # Recency (8d): 7 days since last visit
    0.45, 0.55, 0.35, 0.65, 0.25, 0.75, 0.15, 0.85,
]
```

### T4: Price Context [64d/item]

**Encoding Process**:
```python
def encode_price(product_id, store_id, week):
    # Lookup from price_features_indexed.parquet
    key = (product_id, store_id, week)
    if key in price_lookup:
        return price_lookup[key]  # [64]
    else:
        return random_64d()  # Fallback
```

**Breakdown per item**:
```python
T4_item = [
    # Fourier components (24d)
    0.23, -0.15, 0.08, 0.45, ...,  # Multi-scale price representation

    # Log features (16d)
    2.30, 0.50, -0.20, 0.15, ...,  # log(price), log(price/avg), etc.

    # Relative prices (16d)
    0.95, 1.02, 0.88, 1.05, ...,   # price/category_avg, price/store_avg

    # Velocity (8d)
    0.01, -0.03, 0.02, 0.00, ...,  # Price change momentum
]
```

### T5: Store Context [96d]

**Encoding Process**:
```python
def encode_store(store_id, store_format, store_region):
    return np.concatenate([
        format_embed[store_format],        # [16] LS/MS/SS
        region_embed[store_region],        # [32] E01, W02, etc.
        operational_features,              # [32] size, traffic, competition
        store_identity_embed[store_id],    # [16] learnable
    ])  # Total: [96]
```

**Example** (Large Store in East Region):
```python
T5 = [
    # Format (16d): Large Store
    0.45, 0.23, -0.12, 0.34, ...,

    # Region (32d): E01
    0.12, -0.34, 0.56, 0.23, ...,

    # Operational (32d)
    0.80,  # store_size (normalized)
    0.70,  # traffic_level
    0.30,  # competition_density
    0.90,  # store_age
    ...,

    # Identity (16d): Store-specific learned embedding
    0.11, 0.22, -0.33, 0.44, ...,
]
```

### T6: Trip Context [48d]

**Encoding Process**:
```python
def encode_trip(basket_type, mission, price_sensitivity, basket_size):
    return np.concatenate([
        mission_type_embed[basket_type],      # [16]
        mission_focus_embed[mission],         # [16]
        price_sens_embed[price_sensitivity],  # [8]
        basket_size_embed[basket_size],       # [8]
    ])  # Total: [48]
```

**Vocabulary**:
```python
MISSION_TYPES = ['Top Up', 'Full Shop', 'Small Shop', 'Emergency']
MISSION_FOCUS = ['Fresh', 'Grocery', 'Mixed', 'Nonfood', 'General']
PRICE_SENS = ['LA', 'MM', 'UM']  # Low-Average, Mid-Market, Upper-Market
BASKET_SIZE = ['S', 'M', 'L']    # Small, Medium, Large
```

---

## Stage 4: Data Preparation (`run_data_preparation.py`)

### Stage 4.1: Temporal Metadata

**Output**: `data/prepared/temporal_metadata.parquet`

Creates train/validation/test splits based on time:
- Train: weeks 1-80
- Validation: weeks 81-95
- Test: weeks 96+

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| basket_id | int | Unique basket identifier |
| customer_id | str | Customer identifier |
| store_id | str | Store identifier |
| week | int | Shopping week |
| split | str | 'train', 'validation', or 'test' |
| bucket | int | History bucket (for balanced batching) |

### Stage 4.2: Customer Histories

**Output**: `data/prepared/customer_histories.parquet`

Per-split history extraction for each customer.

### Stage 4.3: Training Samples

**Output**: `data/prepared/samples/`

Bucketed sample files for efficient batch construction.

### Stage 4.4: Tensor Cache

**Output**: `data/prepared/tensor_cache/`

Pre-computed embeddings as `.npy` files:
```
tensor_cache/
├── product_embeddings.npy       # [N_products, 256]
├── customer_history_embeddings.npy  # [N_customers, 160]
├── customer_static_embeddings.npy   # [N_customers, 64]
├── store_embeddings.npy         # [N_stores, 96]
├── price_features_indexed.parquet   # Indexed price features
└── vocab.json                   # ID mappings
```

---

## Stage 5: Next-Basket Samples (`stage4_next_basket_samples.py`)

Creates (basket_t, basket_t+1) pairs for autoregressive prediction.

**Output**:
- `data/prepared/train_next_basket.parquet`
- `data/prepared/validation_next_basket.parquet`
- `data/prepared/test_next_basket.parquet`

**Sample Schema**:
| Column | Type | Description |
|--------|------|-------------|
| customer_id | str | Customer identifier |
| input_basket_id | int | Basket at time t |
| input_week | int | Week of basket t |
| input_store_id | str | Store where basket t purchased |
| input_products | list[str] | Products in basket t (JSON serialized) |
| input_weekday | int | Day of week for basket t |
| input_hour | int | Hour of day for basket t |
| target_basket_id | int | Basket at time t+1 |
| target_week | int | Week of basket t+1 |
| target_store_id | str | Store where basket t+1 purchased |
| target_products | list[str] | Products in basket t+1 (JSON serialized) |
| target_weekday | int | Day of week for basket t+1 |
| target_hour | int | Hour of day for basket t+1 |
| num_history_baskets | int | Number of baskets before t |
| week_gap | int | Weeks between t and t+1 |
| split | str | 'train', 'validation', or 'test' |

**Filters Applied**:
- Minimum target basket size: 2 products
- Maximum time gap: 12 weeks
- Minimum history baskets: 0 (allow cold start)

**Example Row**:
```python
{
    'customer_id': 'CUST_12345',
    'input_basket_id': 987654,
    'input_week': 202340,
    'input_store_id': 'STORE_001',
    'input_products': ['PROD_A', 'PROD_B', 'PROD_C'],  # What they bought
    'target_basket_id': 987678,
    'target_week': 202341,
    'target_store_id': 'STORE_002',
    'target_products': ['PROD_A', 'PROD_D', 'PROD_E'],  # What they'll buy
    'num_history_baskets': 15,
    'week_gap': 1,
    'split': 'train',
}
```

---

## Training: Store Visit Prediction (`train_store_visit.py`)

### Model Architecture

**Stage 1 of Two-Stage World Model**

Predicts: `P(next_store | customer, time, previous_store, previous_basket)`

```
┌─────────────────────────────────────────────────────────────┐
│                   StoreVisitPredictor                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Inputs:                                                    │
│  ├── customer_context [B, 192]    ─┐                        │
│  ├── temporal_context [B, 64]      ├─→ Concat [B, 416]      │
│  ├── previous_store_emb [B, 96]   ─┤     (with basket)      │
│  └── basket_summary [B, 64]       ─┘                        │
│                                                             │
│  MLP Layers:                                                │
│  ├── Linear(416 → 256) + LayerNorm + GELU + Dropout         │
│  └── Linear(256 → 256) + LayerNorm + GELU + Dropout         │
│                                                             │
│  Output Head:                                               │
│  └── Linear(256 → num_stores) → Softmax                     │
│                                                             │
│  Output: store_probs [B, num_stores]                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Input Dimensions

| Input | Dimension | Source |
|-------|-----------|--------|
| customer_context | [B, 192] | T1: static(64) + history(96) + affinity(32) |
| temporal_context | [B, 64] | T3: Encoded target time t+1 |
| previous_store_idx | [B] | Index → embedding lookup → [B, 96] |
| previous_basket_embeddings | [B, S, 256] | T2: Products from basket t |
| previous_basket_mask | [B, S] | Valid positions in basket |

### Basket Summarizer

When `use_basket_summary=True`:
```python
class BasketSummarizer:
    # Input: [B, S, 256] product embeddings
    # Output: [B, 64] basket summary

    proj = Linear(256 → 64)
    norm = LayerNorm(64)

    # Masked mean pooling over valid items
```

### Data Loading

```python
# dataset_store_visit.py
class StoreVisitDataset:
    def get_batch(self, indices) -> StoreVisitBatch:
        return StoreVisitBatch(
            customer_context=...,      # [B, 192]
            temporal_context=...,      # [B, 64]
            previous_store_idx=...,    # [B]
            target_store_idx=...,      # [B] (label)
            previous_basket_embeddings=...,  # [B, S, 256]
            previous_basket_mask=...,  # [B, S]
        )
```

### Loss Function

```python
# Cross-entropy with label smoothing + class weights
loss = CrossEntropyLoss(
    weight=class_weights,      # Handle store imbalance
    label_smoothing=0.1,       # Regularization
)
```

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Move to device
        inputs = to_device(batch)
        targets = inputs.pop('target_store_idx')

        # Forward pass
        outputs = model(**inputs)
        logits = outputs['store_logits']  # [B, num_stores]

        # Compute loss
        loss = loss_fn(logits, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
```

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Exact match |
| Top-3 Accuracy | Target in top 3 predictions |
| Top-5 Accuracy | Target in top 5 predictions |
| Top-10 Accuracy | Target in top 10 predictions |
| MRR | Mean Reciprocal Rank |

---

## Training: Next-Basket Prediction (`train_next_basket.py`)

### Model Architecture

**Stage 2 of Two-Stage World Model**

Predicts: `P(products | customer, time, store, previous_basket)`

```
┌─────────────────────────────────────────────────────────────┐
│                   NextBasketWorldModel                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   InputEncoder                       │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                                                     │    │
│  │  Product Input:                                     │    │
│  │  ├── input_embeddings [B, S, 256]  ─┐               │    │
│  │  └── input_prices [B, S, 64]       ─┴─→ [B, S, 320] │    │
│  │                                                     │    │
│  │  product_proj: Linear(320 → 512)                    │    │
│  │  pos_encoder: Sinusoidal positions                  │    │
│  │                                                     │    │
│  │  TransformerEncoder:                                │    │
│  │  ├── 4 layers                                       │    │
│  │  ├── 8 heads                                        │    │
│  │  └── 1024 FFN dim                                   │    │
│  │                                                     │    │
│  │  Mean pooling → [B, 512]                            │    │
│  │                                                     │    │
│  │  Context:                                           │    │
│  │  ├── customer [B, 192] ─┐                           │    │
│  │  ├── temporal [B, 64]   ├─→ concat [B, 400]         │    │
│  │  ├── store [B, 96]      │                           │    │
│  │  └── trip [B, 48]      ─┘                           │    │
│  │                                                     │    │
│  │  context_proj: Linear(400 → 512)                    │    │
│  │                                                     │    │
│  │  Output: pooled + context_proj(context) → [B, 512]  │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  BasketPredictor                    │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                                                     │    │
│  │  MLP:                                               │    │
│  │  ├── Linear(512 → 1024) + GELU + Dropout            │    │
│  │  └── Linear(1024 → 1024) + GELU + Dropout           │    │
│  │                                                     │    │
│  │  Output Heads:                                      │    │
│  │  ├── product_logits: Linear(1024 → vocab_size)      │    │
│  │  ├── basket_size: Linear(512 → 4)                   │    │
│  │  ├── mission_type: Linear(512 → 5)                  │    │
│  │  ├── mission_focus: Linear(512 → 6)                 │    │
│  │  └── price_sensitivity: Linear(512 → 4)             │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Output: product_logits [B, vocab_size] → Sigmoid           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Input Dimensions

| Input | Dimension | Source |
|-------|-----------|--------|
| input_embeddings | [B, S, 256] | T2: Product embeddings for basket t |
| input_price_features | [B, S, 64] | T4: Price context for basket t |
| input_attention_mask | [B, S] | Valid positions in basket |
| customer_context | [B, 192] | T1: Customer embedding |
| temporal_context | [B, 64] | T3: Encoded target time t+1 |
| store_context | [B, 96] | T5: Target store embedding |
| trip_context | [B, 48] | T6: Trip type encoding |

### Data Loading

```python
# dataset_next_basket.py
class NextBasketDataset:
    def get_batch(self, indices) -> NextBasketBatch:
        return NextBasketBatch(
            # Context tensors
            customer_context=...,       # [B, 192]
            temporal_context=...,       # [B, 64]
            store_context=...,          # [B, 96]
            trip_context=...,           # [B, 48]

            # Input basket (time t)
            input_embeddings=...,       # [B, S, 256]
            input_price_features=...,   # [B, S, 64]
            input_attention_mask=...,   # [B, S]

            # Target (time t+1)
            target_products=...,        # [B, vocab_size] multi-hot
            auxiliary_labels=...,       # Dict of auxiliary targets
        )
```

### Loss Function

```python
# Multi-label BCE + auxiliary losses
class NextBasketLoss:
    def forward(self, outputs, targets, auxiliary_labels):
        # Main product loss (multi-label)
        product_loss = BCEWithLogitsLoss()(
            outputs['product_logits'],
            targets
        )

        # Auxiliary losses (cross-entropy)
        aux_loss = 0
        for task in ['basket_size', 'mission_type', 'mission_focus', 'price_sensitivity']:
            aux_loss += CrossEntropyLoss()(
                outputs[task],
                auxiliary_labels[task]
            )

        return product_loss + 0.1 * aux_loss
```

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Move to device (7 input tensors)
        input_embeddings = torch.tensor(batch.input_embeddings, device=device)
        input_price_features = torch.tensor(batch.input_price_features, device=device)
        input_attention_mask = torch.tensor(batch.input_attention_mask, device=device)
        customer_context = torch.tensor(batch.customer_context, device=device)
        temporal_context = torch.tensor(batch.temporal_context, device=device)
        store_context = torch.tensor(batch.store_context, device=device)
        trip_context = torch.tensor(batch.trip_context, device=device)
        targets = torch.tensor(batch.target_products, device=device)

        # Forward pass
        outputs = model(
            input_embeddings,
            input_price_features,
            input_attention_mask,
            customer_context,
            temporal_context,
            store_context,
            trip_context,
        )

        # Loss
        loss, loss_dict = loss_fn(outputs, targets, auxiliary_labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
```

### Metrics

| Metric | Description |
|--------|-------------|
| Precision@k | Fraction of predicted products that are correct |
| Recall@k | Fraction of actual products that are predicted |
| F1@k | Harmonic mean of P@k and R@k |
| Hit Rate@k | Did we predict at least one correct product? |

---

## Two-Stage Inference Pipeline

At inference time, the models work together:

```python
def predict_next_visit(customer_id, current_basket, current_store, current_time):
    # Encode inputs
    customer_context = encode_customer(customer_id)        # [1, 192]
    temporal_context = encode_time(next_time)              # [1, 64]
    basket_embeddings = encode_basket(current_basket)      # [1, S, 256]

    # Stage 1: Predict store
    store_outputs = store_visit_model(
        customer_context=customer_context,
        temporal_context=temporal_context,
        previous_store_idx=store_to_idx[current_store],
        previous_basket_embeddings=basket_embeddings,
    )
    predicted_store_idx = store_outputs['store_probs'].argmax()
    # Or sample: torch.multinomial(store_outputs['store_probs'], 1)

    # Stage 2: Predict basket at predicted store
    store_context = store_embeddings[predicted_store_idx]  # [1, 96]
    trip_context = encode_trip(predicted_mission)          # [1, 48]

    basket_outputs = next_basket_model(
        input_embeddings=basket_embeddings,
        input_price_features=price_features,
        input_attention_mask=attention_mask,
        customer_context=customer_context,
        temporal_context=temporal_context,
        store_context=store_context,
        trip_context=trip_context,
    )

    product_probs = torch.sigmoid(basket_outputs['product_logits'])
    predicted_products = (product_probs > 0.5).nonzero()
    # Or top-k: product_probs.topk(k=10)

    return {
        'predicted_store': idx_to_store[predicted_store_idx],
        'predicted_products': [idx_to_product[i] for i in predicted_products],
        'product_probabilities': product_probs,
    }
```

---

## Complete Data Flow Summary

```
Raw Transaction → Data Pipeline → Feature Engineering → Tensor Prep → Data Prep → Training

1. transactions.csv
   ↓
2. run_pipeline.py
   ├── prices_derived.parquet (unit prices)
   ├── product_graph.pkl (co-purchase graph)
   ├── customer_store_affinity.parquet (loyalty)
   └── customer_mission_patterns.parquet (trip types)
   ↓
3. run_feature_engineering.py
   ├── pseudo_brands.parquet (product clusters)
   ├── price_features.parquet (64d per product-store-week)
   ├── product_embeddings.pkl (256d per product)
   ├── customer_embeddings.parquet (160d per customer)
   └── store_features.parquet (96d per store)
   ↓
4. run_tensor_preparation.py
   └── Validates T1(192d), T2(256d), T3(64d), T4(64d), T5(96d), T6(48d)
   ↓
5. run_data_preparation.py
   ├── temporal_metadata.parquet (train/val/test splits)
   ├── customer_histories.parquet
   ├── samples/ (bucketed samples)
   └── tensor_cache/*.npy (pre-computed embeddings)
   ↓
6. stage4_next_basket_samples.py
   ├── train_next_basket.parquet
   ├── validation_next_basket.parquet
   └── test_next_basket.parquet
   ↓
7. Training
   ├── train_store_visit.py
   │   └── Input: T1(192) + T3(64) + prev_store(96) + basket_summary(64) = 416d
   │   └── Output: P(store) over ~761 stores
   │
   └── train_next_basket.py
       └── Input: T1(192) + T3(64) + T5(96) + T6(48) = 400d context
       │         + T2(256) + T4(64) = 320d/item sequence
       └── Output: P(product) over ~5000 products (multi-label)
```
