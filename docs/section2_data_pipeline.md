# Section 2: Data Pipeline Architecture

The data pipeline processes raw Dunnhumby LGSR transaction data through four sequential stages to create foundational features for the RetailSim world model.

## Overview

```
Raw Transactions → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Processed Data
                   (Price)   (Graph)   (Affinity) (Mission)
```

**Input**: `raw_data/transactions.csv` (Dunnhumby Let's Get Sort of Real dataset)
**Output**: Processed parquet files in `data/processed/`

## Stage 1: Price Derivation Pipeline

**File**: `src/data_pipeline/stage1_price_derivation.py`
**Class**: `PriceDerivationPipeline`

### Purpose
Derives unit prices from transaction data using a waterfall imputation strategy to handle missing values.

### Price Derivation Logic
```
Unit Price = SPEND / QUANTITY
```

### Waterfall Imputation Strategy
When direct price calculation fails (division by zero, missing data), the pipeline falls back through increasingly general price estimates:

1. **Product-Week Price**: Average price for same product in same week
2. **Product Price**: Overall average price for the product
3. **Category-Week Price**: Average price for category (PROD_CODE_40) in same week
4. **Category Price**: Overall average price for category
5. **Global Median**: Dataset-wide median price (last resort)

### Output Schema
| Column | Type | Description |
|--------|------|-------------|
| `product_id` | str | Product code (PROD_CODE) |
| `week` | int | Shopping week (SHOP_WEEK) |
| `actual_price` | float | Derived unit price |
| `base_price` | float | Reference price (product median) |
| `discount_pct` | float | Discount percentage from base |
| `price_rank` | float | Percentile rank within category |

### Usage
```python
from src.data_pipeline.stage1_price_derivation import PriceDerivationPipeline

pipeline = PriceDerivationPipeline()
prices_df = pipeline.run(transactions_df)
prices_df.to_parquet('data/processed/prices_derived.parquet')
```

---

## Stage 2: Product Graph Construction

**File**: `src/data_pipeline/stage2_product_graph.py`
**Class**: `ProductGraphBuilder`

### Purpose
Builds a heterogeneous product graph capturing relationships between products for downstream graph neural network processing.

### Edge Types

#### 1. Co-purchase Edges
Products frequently bought together in the same basket.

```python
# PMI-based co-purchase scoring
pmi = log(P(A,B) / (P(A) * P(B)))
weight = max(0, pmi)  # Only positive associations
```

**Threshold**: Top co-purchases by PMI score

#### 2. Substitution Edges
Products that could replace each other (same category, similar price).

**Criteria**:
- Same sub-commodity (PROD_CODE_40)
- Price ratio between 0.5 and 2.0
- Different products

#### 3. Hierarchy Edges
Category structure from product codes.

```
PROD_CODE_10 (Department)
    └── PROD_CODE_20 (Category)
        └── PROD_CODE_30 (Sub-category)
            └── PROD_CODE_40 (Sub-commodity)
                └── PROD_CODE (Product)
```

### Graph Structure
- **Nodes**: Products (PROD_CODE) + Category nodes
- **Edges**: Typed edges with weights
- **Format**: NetworkX graph saved as pickle

### Output
```python
# Graph statistics example
Nodes: 4,313
Edges: 15,001
  - copurchase: ~10,000
  - substitution: ~2,000
  - hierarchy: ~3,000
```

### Usage
```python
from src.data_pipeline.stage2_product_graph import ProductGraphBuilder

builder = ProductGraphBuilder()
graph = builder.run(transactions_df, prices_df)
builder.save('data/processed/product_graph.pkl')
```

---

## Stage 3: Customer-Store Affinity

**File**: `src/data_pipeline/stage3_customer_store_affinity.py`
**Class**: `CustomerStoreAffinityPipeline`

### Purpose
Computes customer loyalty and store affinity metrics to understand shopping patterns.

### Metrics Computed

#### Herfindahl Index (Store Concentration)
Measures how concentrated a customer's shopping is across stores.

```python
HHI = Σ(share_i²)  # where share_i = visits to store i / total visits
# HHI = 1.0: shops at only one store
# HHI → 0: shops equally across many stores
```

#### Primary Store
The store where the customer shops most frequently.

#### Switching Rate
Proportion of consecutive trips where the customer changes stores.

```python
switching_rate = store_changes / (total_trips - 1)
```

#### Loyalty Score
Composite metric combining concentration and consistency.

```python
loyalty_score = 0.6 * hhi + 0.4 * (1 - switching_rate)
```

### Output Schema
| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | str | Customer code |
| `primary_store` | str | Most visited store |
| `store_concentration` | float | Herfindahl Index [0,1] |
| `switching_rate` | float | Store change frequency [0,1] |
| `loyalty_score` | float | Composite loyalty metric [0,1] |
| `total_trips` | int | Number of shopping trips |
| `unique_stores` | int | Number of distinct stores visited |

### Usage
```python
from src.data_pipeline.stage3_customer_store_affinity import CustomerStoreAffinityPipeline

pipeline = CustomerStoreAffinityPipeline(min_trips=3)
affinity_df = pipeline.run(transactions_df)
affinity_df.to_parquet('data/processed/customer_store_affinity.parquet')
```

---

## Stage 4: Mission Pattern Extraction

**File**: `src/data_pipeline/stage4_mission_patterns.py`
**Class**: `MissionPatternPipeline`

### Purpose
Extracts shopping mission patterns from customer transaction history to understand behavioral signatures.

### Mission Dimensions

#### Mission Type Distribution
Based on `BASKET_TYPE` field:
- **Top Up**: Quick replenishment trips
- **Full Shop**: Complete weekly/monthly shopping
- **Small Shop**: Medium-sized trips
- **Emergency**: Urgent small purchases

#### Mission Focus Distribution
Based on `BASKET_DOMINANT_MISSION`:
- **Fresh**: Focus on fresh produce, dairy, meat
- **Grocery**: Dry goods, pantry staples
- **Mixed**: Balanced basket
- **Nonfood**: Household items, personal care

#### Price Sensitivity
Based on `BASKET_PRICE_SENSITIVITY`:
- **LA**: Low sensitivity (premium shopper)
- **MM**: Medium sensitivity (mainstream)
- **UM**: High sensitivity (value seeker)

#### Basket Size Patterns
Based on `BASKET_SIZE`:
- **S**: Small baskets
- **M**: Medium baskets
- **L**: Large baskets

### Output Schema
| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | str | Customer code |
| `p_mission_*` | float | Mission type probabilities |
| `p_focus_*` | float | Mission focus probabilities |
| `mean_price_sensitivity` | float | Average price sensitivity |
| `mean_basket_size` | float | Average basket size |
| `total_baskets` | int | Number of baskets analyzed |

### Usage
```python
from src.data_pipeline.stage4_mission_patterns import MissionPatternPipeline

pipeline = MissionPatternPipeline(min_baskets=3)
patterns_df = pipeline.run(transactions_df)
patterns_df.to_parquet('data/processed/customer_mission_patterns.parquet')
```

---

## Running the Complete Pipeline

### Command Line
```bash
# Run with default 10,000 rows
python -m src.data_pipeline.run_pipeline

# Run with custom row limit
python -m src.data_pipeline.run_pipeline --nrows 50000
```

### Programmatic
```python
from src.data_pipeline.run_pipeline import run_data_pipeline

results = run_data_pipeline(nrows=10000)
# Returns dict with: prices, graph, affinity, mission_patterns
```

### Output Files
```
data/processed/
├── prices_derived.parquet      # Stage 1 output
├── product_graph.pkl           # Stage 2 output
├── customer_store_affinity.parquet  # Stage 3 output
└── customer_mission_patterns.parquet # Stage 4 output
```

---

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- networkx >= 3.0
- pyarrow >= 14.0.0

## Performance Notes

- Stage 1 (Price): O(n) where n = transactions
- Stage 2 (Graph): O(n²) for co-purchase computation (optimized with sparse matrices)
- Stage 3 (Affinity): O(c × t) where c = customers, t = avg trips
- Stage 4 (Mission): O(c × b) where c = customers, b = avg baskets

For 10,000 transactions: ~5 seconds total pipeline time.
