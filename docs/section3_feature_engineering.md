# Section 3: Feature Engineering

The feature engineering module transforms raw and processed data into rich embeddings and feature vectors suitable for the world model.

## Overview

```
Data Pipeline Outputs → Layer 1 → Layer 2 → Layer 3 → Layer 4 → Layer 5 → Features
                       (Brand)   (Price)   (Graph)   (Customer) (Store)
```

**Input**: Outputs from Section 2 Data Pipeline
**Output**: Feature files in `data/features/`

## Layer 1: Pseudo-Brand Inference

**File**: `src/feature_engineering/layer1_pseudo_brand.py`
**Class**: `PseudoBrandInference`

### Purpose
Infers pseudo-brand identities from price tiers and substitution patterns since the dataset lacks explicit brand information.

### Methodology

#### Price Tier Classification
Products are classified into price tiers within their category:

```python
# Within each sub-commodity (PROD_CODE_40)
Premium: price > 75th percentile
Mid:     25th percentile <= price <= 75th percentile
Value:   price < 25th percentile
```

#### Pseudo-Brand Assignment
Combines category and price tier to create pseudo-brand identifiers:

```python
pseudo_brand_id = f"{category}_{price_tier}_{cluster_id}"
```

### Output Schema
| Column | Type | Description |
|--------|------|-------------|
| `product_id` | str | Product code |
| `pseudo_brand_id` | str | Inferred brand identifier |
| `price_tier` | str | Premium/Mid/Value |
| `category` | str | Sub-commodity code |
| `brand_size` | int | Products in this pseudo-brand |

### Usage
```python
from src.feature_engineering.layer1_pseudo_brand import PseudoBrandInference

inferencer = PseudoBrandInference()
pseudo_brands = inferencer.run(transactions_df, prices_df, product_graph)
```

---

## Layer 2: Fourier Price Encoding

**File**: `src/feature_engineering/layer2_fourier_price.py`
**Class**: `FourierPriceEncoder`

### Purpose
Creates rich 64-dimensional price representations using Fourier features to capture cyclical price patterns.

### Feature Components (64d total)

#### Fourier Features [24d]
Captures cyclical price patterns at multiple frequencies:

```python
frequencies = [1/7, 1/14, 1/30, 1/90, 1/2, 1/3, 1/4, 1/5]
for freq in frequencies:
    features.append(sin(2π × freq × price))
    features.append(cos(2π × freq × price))
```

#### Log-Price Features [16d]
Captures price magnitude and relationships:

```python
log_actual = log(actual_price)
log_base = log(base_price)
log_diff = log_base - log_actual  # Discount in log space
# Plus polynomial expansions
```

#### Relative Price Features [16d]
Position within category price distribution:

```python
relative = actual_price / category_avg
# Premium indicator: relative > 1.3
# Value indicator: relative < 0.7
# Plus Fourier encodings of relative position
```

#### Velocity Features [8d]
Price change dynamics:

```python
velocity = (current_price - prior_price) / prior_price
# Increasing/Decreasing/Stable indicators
# Fourier encodings of velocity
```

### Output Schema
| Column | Type | Description |
|--------|------|-------------|
| `product_id` | str | Product code |
| `week` | int | Shopping week |
| `actual_price` | float | Transaction price |
| `fourier_0..23` | float | Fourier features |
| `log_0..15` | float | Log-price features |
| `relative_0..15` | float | Relative price features |
| `velocity_0..7` | float | Price velocity features |

### Usage
```python
from src.feature_engineering.layer2_fourier_price import FourierPriceEncoder

encoder = FourierPriceEncoder()
price_features = encoder.run(prices_df)
```

---

## Layer 3: Graph Embeddings (GraphSAGE)

**File**: `src/feature_engineering/layer3_graph_embeddings.py`
**Class**: `GraphSAGEEncoder`

### Purpose
Generates 256-dimensional product embeddings using GraphSAGE aggregation over the product graph.

### Architecture

#### Node Feature Initialization
Initial features from pseudo-brand and price information:

```python
node_features = [
    pseudo_brand_embedding,  # [64d]
    price_tier_embedding,    # [32d]
    category_embedding,      # [64d]
    random_init              # [96d]
]  # Total: 256d
```

#### GraphSAGE Aggregation
Two-layer message passing:

```python
# Layer 1: 1-hop neighborhood
h1 = σ(W1 × CONCAT(h0, AGG(neighbors_1hop)))

# Layer 2: 2-hop neighborhood
h2 = σ(W2 × CONCAT(h1, AGG(neighbors_2hop)))
```

**Aggregation Function**: Mean aggregation with edge-type weighting

```python
# Different weights for edge types
copurchase_weight = 1.0
substitution_weight = 0.8
hierarchy_weight = 0.5
```

### Output
```python
{
    'embeddings': {product_id: np.array([256d])},
    'output_dim': 256
}
```

### Usage
```python
from src.feature_engineering.layer3_graph_embeddings import GraphSAGEEncoder

encoder = GraphSAGEEncoder(embedding_dim=256, num_layers=2)
product_embeddings = encoder.run(product_graph, pseudo_brands)
encoder.save('data/features/product_embeddings.pkl')
```

---

## Layer 4: Customer History Encoding

**File**: `src/feature_engineering/layer4_customer_history.py`
**Class**: `CustomerHistoryEncoder`

### Purpose
Encodes customer purchase history into 160-dimensional behavioral signatures.

### Architecture

#### Trip-Level Encoding [176d per trip]
Each shopping trip is encoded with:

```python
trip_features = [
    basket_product_embedding,  # [256d] Mean of product embeddings
    temporal_features,         # [32d] Week, day, hour encoding
    basket_size_features,      # [16d] Item count, spend
    mission_features           # [32d] Type, focus indicators
]
# Projected to 176d
```

#### Sequence Aggregation
Attention-weighted aggregation over trip history:

```python
# Recency-biased attention
recency_weights = exp(linspace(-1, 0, num_trips))

# Content-based attention
content_scores = trip_matrix @ W_attention

# Combined attention
attention = softmax(recency_weights × content_scores)

# Weighted aggregation
history_embed = Σ(attention_i × trip_i)
```

#### Mission Statistics [32d]
Pre-computed mission pattern features:

```python
mission_stats = [
    mission_type_distribution,    # [8d]
    mission_focus_distribution,   # [8d]
    price_sensitivity_stats,      # [8d]
    basket_size_stats             # [8d]
]
```

### Cold-Start Handling
Adaptive blending based on trip count:

```python
if num_trips < 5:
    alpha = 0.8  # Rely more on population averages
else:
    alpha = 1 / log(num_trips + 1)

final_embed = alpha × population_avg + (1-alpha) × personal_embed
```

### Output Schema
Parquet file with columns:
- `customer_id`: Customer code
- `embed_0..159`: Embedding dimensions
- `total_trips`: Number of trips

### Usage
```python
from src.feature_engineering.layer4_customer_history import CustomerHistoryEncoder

encoder = CustomerHistoryEncoder(output_dim=160)
customer_embeddings = encoder.run(transactions_df, product_embeddings, mission_patterns)
encoder.save('data/features/customer_history_embeddings.pkl')
```

---

## Layer 5: Store Context Features

**File**: `src/feature_engineering/layer5_store_context.py`
**Class**: `StoreContextEncoder`

### Purpose
Generates 96-dimensional store representations combining identity, format, and operational features.

### Feature Components (96d total)

#### Store Identity [32d]
Hash-based consistent embedding:

```python
store_hash = hash(store_id)
np.random.seed(store_hash)
identity_embed = np.random.randn(32) * 0.1
```

#### Format Features [24d]
Store format encoding:

```python
formats = ['LS', 'MS', 'SS']  # Large/Medium/Small Super
format_embed = embedding_lookup[store_format]
```

#### Region Features [24d]
Geographic region encoding:

```python
regions = ['E01', 'E02', 'W01', 'W02', ...]
region_embed = embedding_lookup[store_region]
```

#### Operational Features [16d]
Derived from transaction patterns:

```python
operational = [
    baskets_per_week_norm,    # Traffic volume
    customers_per_week_norm,  # Unique visitors
    avg_spend_norm,           # Basket size
    tenure_weeks_norm         # Store age proxy
]
# Plus Fourier expansions
```

### Output Schema
| Column | Type | Description |
|--------|------|-------------|
| `store_id` | str | Store code |
| `format` | str | Store format (LS/MS/SS) |
| `region` | str | Region code |
| `identity_0..31` | float | Identity embedding |
| `format_0..23` | float | Format features |
| `region_0..23` | float | Region features |
| `operational_0..15` | float | Operational features |

### Usage
```python
from src.feature_engineering.layer5_store_context import StoreContextEncoder

encoder = StoreContextEncoder()
store_features = encoder.run(transactions_df, customer_affinity)
store_features.to_parquet('data/features/store_features.parquet')
```

---

## Running the Complete Pipeline

### Command Line
```bash
# Run with default 10,000 rows
python -m src.feature_engineering.run_feature_engineering

# Run with custom row limit
python -m src.feature_engineering.run_feature_engineering --nrows 50000
```

### Prerequisites
The data pipeline (Section 2) must be run first:
```bash
python -m src.data_pipeline.run_pipeline --nrows 10000
```

### Output Files
```
data/features/
├── pseudo_brands.parquet           # Layer 1: 574 pseudo-brands
├── price_features.parquet          # Layer 2: 64d price features
├── product_embeddings.pkl          # Layer 3: 256d product embeddings
├── customer_history_embeddings.pkl # Layer 4: 160d customer embeddings
├── customer_embeddings.parquet     # Layer 4: Parquet format for tensors
└── store_features.parquet          # Layer 5: 96d store features
```

---

## Feature Dimensions Summary

| Layer | Feature | Dimension | Entities |
|-------|---------|-----------|----------|
| 1 | Pseudo-Brand | categorical | ~574 brands |
| 2 | Price Encoding | 64d | per product-week |
| 3 | Product Embedding | 256d | ~3,939 products |
| 4 | Customer History | 160d | ~946 customers |
| 5 | Store Context | 96d | ~3 stores |

---

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- networkx >= 3.0
- scipy >= 1.11.0

Optional:
- torch >= 2.0.0 (for GPU-accelerated GraphSAGE)

## Performance Notes

- Layer 1 (Pseudo-Brand): O(p) where p = products
- Layer 2 (Price): O(n) where n = price records
- Layer 3 (Graph): O(p × d × L) where d = degree, L = layers
- Layer 4 (Customer): O(c × t) where c = customers, t = avg trips
- Layer 5 (Store): O(s) where s = stores

For 10,000 transactions: ~60 seconds total (Layer 3 dominates).
