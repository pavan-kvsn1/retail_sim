# RetailSim: Technical Product Vision Document v7.0
## A World Model + Multi-Agent RL Platform for Retail Decision Simulation

**Built on Production-Scale Data: The "Let's Get Sort-of-Real" (LGSR) Dataset**

---

## Version History

**Version 7.0 - Major Pivot to Production-Scale Dataset**

Key Changes from v5.x/v6.x:
- **ABANDONED: Dunnhumby Complete Journey** (801 users - insufficient for deep learning)
- **ADOPTED: "Let's Get Sort-of-Real" Dataset** (500K users - production viable)
- **NEW: Derived Price/Promo Pipeline** - Imputing prices from transaction-level SPEND/QUANTITY
- **NEW: Segment-Based Cold-Start** - Using `seg_1`/`seg_2` instead of demographics
- **NEW: Multi-Task Learning Targets** - Basket size, price sensitivity, mission type
- **REMOVED: Coupon tensor** - Not available in LGSR dataset
- **UPDATED: Realistic scale estimates** - 300M transactions, 47M baskets

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Overview: LGSR](#2-dataset-overview-lgsr)
3. [Data Processing Pipeline](#3-data-processing-pipeline)
4. [Tensor Specifications](#4-tensor-specifications)
5. [Model Architecture](#5-model-architecture)
6. [Training Strategy](#6-training-strategy)
7. [Multi-Agent RL System](#7-multi-agent-rl-system)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Success Metrics](#9-success-metrics)
10. [Risk Mitigation](#10-risk-mitigation)

---

## 1. Executive Summary

### The Vision

RetailSim is a "Flight Simulator for Retail" - a comprehensive platform where AI agents learn optimal pricing, promotion, and inventory strategies through realistic simulation before real-world deployment.

### Why v7.0?

**Previous versions (v5.x, v6.x) had a fatal flaw:**

| Dataset | Users | Verdict | Problem |
|---------|-------|---------|---------|
| Dunnhumby Complete Journey | 801 | ❌ Unusable | 50-100x too small for deep learning |
| Proposed synthetic expansion | 2,500 | ❌ Still unusable | Model memorizes, doesn't generalize |
| **LGSR Dataset** | **500,000** | **✅ Production-viable** | Sufficient scale for transformers |

**The "Let's Get Sort-of-Real" (LGSR) dataset provides:**
- 500K distinct customers (625x more than Complete Journey)
- 300M transactions over 117 weeks
- 47M shopping baskets
- 5,000 products across 760 stores
- Rich behavioral signals (segments, missions, basket types)

### What We Gain

| Feature | Complete Journey | LGSR Dataset | Impact |
|---------|-----------------|--------------|--------|
| Users | 801 | 500,000 | ✅ Deep learning viable |
| Transactions | 650K | 300M | ✅ 460x more training data |
| Baskets | 82K | 47M | ✅ 573x more sequences |
| Demographics | ✅ Rich | ❌ Missing | ⚠️ Use segments instead |
| Prices | ✅ Explicit | ❌ Must derive | ⚠️ Imputation pipeline |
| Promos | ✅ Explicit | ❌ Must derive | ⚠️ Detection algorithm |
| Coupons | ✅ Available | ❌ Missing | ❌ Remove from model |

### What We Lose (and How We Compensate)

| Lost Feature | Compensation Strategy |
|--------------|----------------------|
| Demographics | Use `seg_1` (lifestage) + `seg_2` (lifestyle) as persona tokens |
| Explicit Prices | Derive from `SPEND/QUANTITY` with waterfall imputation |
| Explicit Promos | Detect via `Actual_Price < 0.95 × Base_Price` threshold |
| Coupons | Remove coupon tensor entirely; accept limitation |

---

## 2. Dataset Overview: LGSR

### 2.1 Scale Summary

```
┌─────────────────────────────────────────────────────────────┐
│              "Let's Get Sort-of-Real" Dataset               │
├─────────────────────────────────────────────────────────────┤
│  Temporal Scope                                             │
│  • 117 weeks of transactions (April 2006 - December 2008)   │
│  • ~2.25 years of shopping behavior                         │
├─────────────────────────────────────────────────────────────┤
│  Transaction Volume                                         │
│  • 300M total transaction rows                              │
│  • 47M distinct baskets                                     │
│  • 2.6M transactions per week (average)                     │
│  • 400K baskets per week (average)                          │
├─────────────────────────────────────────────────────────────┤
│  Entity Counts                                              │
│  • ~500,000 distinct customers (with loyalty cards)         │
│  • ~5,000 distinct products (SKUs)                          │
│  • ~760 distinct stores                                     │
│  • 3 store formats (LS, MS, SS)                             │
│  • Multiple regions (E01, E02, W01, W02, S01, etc.)         │
├─────────────────────────────────────────────────────────────┤
│  Behavioral Richness                                        │
│  • Customer segments: seg_1 × seg_2 (lifestyle/lifestage)   │
│  • Basket metadata: size, price sensitivity, type, mission  │
│  • Temporal signals: date, weekday, hour                    │
│  • Product hierarchy: 4 levels (D→G→DEP→CL→PRD)             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Schema

**Transaction Table (Primary Data Source):**

| Column | Type | Example | Use in RetailSim |
|--------|------|---------|------------------|
| `SHOP_WEEK` | Temporal | `200626` | Time step index (links to time.csv) |
| `SHOP_DATE` | Temporal | `20060823` | Day-of-week, recency calculations |
| `SHOP_WEEKDAY` | Temporal | `4` | Shopping pattern features |
| `SHOP_HOUR` | Temporal | `15` | Context signal (morning vs evening shopper) |
| `QUANTITY` | Numerical | `3` | Purchase quantity, unit price calculation |
| `SPEND` | Numerical | `5.04` | Revenue signal, price derivation |
| `PROD_CODE` | ID | `PRD0901543` | Product identifier (vocab token) |
| `PROD_CODE_10` | ID | `CL00151` | Hierarchy Level 1: Sub-commodity |
| `PROD_CODE_20` | ID | `DEP00052` | Hierarchy Level 2: Commodity |
| `PROD_CODE_30` | ID | `G00015` | Hierarchy Level 3: Sub-department |
| `PROD_CODE_40` | ID | `D00003` | Hierarchy Level 4: Department |
| `CUST_CODE` | ID | `CUST0000472158` | Customer identifier |
| `seg_1` | Categorical | `CT`, `AZ`, `BG` | Lifestage segment (cold-start persona) |
| `seg_2` | Categorical | `DI`, `FN`, `BU` | Lifestyle segment (price sensitivity init) |
| `BASKET_ID` | ID | `9.94E+14` | Session grouping key |
| `BASKET_SIZE` | Categorical | `L`, `M`, `S` | Multi-task target: trip size prediction |
| `BASKET_PRICE_SENSITIVITY` | Categorical | `LA`, `MM`, `UM` | Elasticity ground truth |
| `BASKET_TYPE` | Categorical | `Top Up`, `Full Shop` | Shopping mission |
| `BASKET_DOMINANT_MISSION` | Categorical | `Fresh`, `Grocery` | Category intent |
| `STORE_CODE` | ID | `STORE00001` | Store identifier |
| `STORE_FORMAT` | Categorical | `LS`, `MS`, `SS` | Store format (Large/Medium/Small Super) |
| `STORE_REGION` | Categorical | `E02`, `W01` | Geographic region |

**Time Dimension Table:**

| Column | Type | Example |
|--------|------|---------|
| `shop_week` | Integer | `200626` |
| `date_from` | Date | `20060821` |
| `date_to` | Date | `20060827` |

### 2.3 Segment Encoding (Cold-Start Personas)

**Lifestage Segments (`seg_1`):**
```
Observed values: CT, AZ, BG, DY, ...
Interpretation (hypothesized):
├─ CT: "Couples/Traditional" - established households
├─ AZ: "Active/Young" - younger demographics
├─ BG: "Budget/Growing" - value-focused families
├─ DY: "Dynamic/Young professionals" - urban singles
└─ (Exact mapping to be validated in EDA phase)
```

**Lifestyle Segments (`seg_2`):**
```
Observed values: DI, FN, BU, CZ, EQ, AT, ...
Interpretation (hypothesized):
├─ DI: "Discerning" - quality-focused
├─ FN: "Foodie/Natural" - fresh/organic preference
├─ BU: "Budget" - price-sensitive
├─ CZ: "Convenient/Zealous" - convenience-focused
├─ EQ: "Economy" - value-seeking
├─ AT: "Affluent/Traditional" - premium segment
└─ (Exact mapping to be validated in EDA phase)
```

**Cold-Start Strategy:**
```python
# Instead of demographics, use segment tokens
persona_embedding = concat(
    segment_embed(seg_1),  # Lifestage: [dim=32]
    segment_embed(seg_2)   # Lifestyle: [dim=32]
)  # Total: [dim=64]

# Feed as first tokens in customer sequence
customer_sequence = [<PERSONA>, seg_1_token, seg_2_token, <HISTORY>, ...]
```

---

## 3. Data Processing Pipeline

### 3.1 Price Derivation Pipeline

**The Challenge:**
LGSR dataset contains `SPEND` and `QUANTITY` but no explicit shelf prices or promotional flags.

**Solution: Derived Pricing with Waterfall Imputation**

```python
class PriceDerivationPipeline:
    """
    Derives shelf prices and promotional flags from transaction data.
    
    Core Logic:
    1. Actual_Price = Median(SPEND / QUANTITY) per (product, store, week)
    2. Base_Price = Max(Actual_Price) over rolling 4-week window
    3. Promo_Flag = (1 - Actual_Price / Base_Price) > 0.05
    """
    
    def __init__(self, promo_threshold: float = 0.05):
        self.promo_threshold = promo_threshold
        self.price_cache = {}
    
    def compute_actual_price(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Compute actual (effective) price per product-store-week.
        
        Formula: Median(SPEND / QUANTITY)
        Why Median: Robust to outliers (returns, data errors, extreme discounts)
        """
        transactions_df['unit_price'] = (
            transactions_df['SPEND'] / transactions_df['QUANTITY']
        )
        
        actual_prices = transactions_df.groupby(
            ['PROD_CODE', 'STORE_CODE', 'SHOP_WEEK']
        )['unit_price'].median().reset_index()
        
        actual_prices.columns = ['product_id', 'store_id', 'week', 'actual_price']
        
        return actual_prices
    
    def compute_base_price(self, actual_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Compute base (shelf) price using rolling maximum.
        
        Formula: Max(Actual_Price) over [week-2, week+1] window
        Rationale: Highest recent price represents non-promotional shelf price
        """
        # Sort by product, store, week
        actual_prices = actual_prices.sort_values(
            ['product_id', 'store_id', 'week']
        )
        
        # Rolling max with 4-week centered window
        actual_prices['base_price'] = actual_prices.groupby(
            ['product_id', 'store_id']
        )['actual_price'].transform(
            lambda x: x.rolling(window=4, center=True, min_periods=1).max()
        )
        
        return actual_prices
    
    def apply_waterfall_imputation(self, prices_df: pd.DataFrame, 
                                   store_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Handle missing prices with waterfall fallback.
        
        Priority:
        1. Store-specific price (if exists)
        2. Format-Region price (same store type + region)
        3. Global price (all stores)
        4. Forward-fill from previous week
        """
        # Merge store metadata
        prices_with_meta = prices_df.merge(
            store_metadata[['store_id', 'format', 'region']],
            on='store_id',
            how='left'
        )
        
        # Level 2: Format-Region fallback
        format_region_prices = prices_with_meta.groupby(
            ['product_id', 'week', 'format', 'region']
        )[['actual_price', 'base_price']].median().reset_index()
        format_region_prices.columns = [
            'product_id', 'week', 'format', 'region', 
            'fr_actual_price', 'fr_base_price'
        ]
        
        # Level 3: Global fallback
        global_prices = prices_with_meta.groupby(
            ['product_id', 'week']
        )[['actual_price', 'base_price']].median().reset_index()
        global_prices.columns = [
            'product_id', 'week', 'global_actual_price', 'global_base_price'
        ]
        
        # Merge fallbacks
        prices_with_fallbacks = prices_with_meta.merge(
            format_region_prices, 
            on=['product_id', 'week', 'format', 'region'],
            how='left'
        ).merge(
            global_prices,
            on=['product_id', 'week'],
            how='left'
        )
        
        # Apply waterfall
        prices_with_fallbacks['final_actual_price'] = (
            prices_with_fallbacks['actual_price']
            .fillna(prices_with_fallbacks['fr_actual_price'])
            .fillna(prices_with_fallbacks['global_actual_price'])
        )
        
        prices_with_fallbacks['final_base_price'] = (
            prices_with_fallbacks['base_price']
            .fillna(prices_with_fallbacks['fr_base_price'])
            .fillna(prices_with_fallbacks['global_base_price'])
        )
        
        # Level 4: Forward-fill remaining NaNs
        prices_with_fallbacks = prices_with_fallbacks.sort_values(
            ['product_id', 'store_id', 'week']
        )
        prices_with_fallbacks['final_actual_price'] = (
            prices_with_fallbacks.groupby(['product_id', 'store_id'])
            ['final_actual_price'].ffill()
        )
        prices_with_fallbacks['final_base_price'] = (
            prices_with_fallbacks.groupby(['product_id', 'store_id'])
            ['final_base_price'].ffill()
        )
        
        return prices_with_fallbacks
    
    def compute_promo_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Derive promotional features from price differences.
        
        Discount_Depth = 1 - (Actual_Price / Base_Price)
        Promo_Flag = Discount_Depth > threshold (default 5%)
        """
        prices_df['discount_depth'] = (
            1 - prices_df['final_actual_price'] / prices_df['final_base_price']
        ).clip(lower=0)  # No negative discounts
        
        prices_df['promo_flag'] = (
            prices_df['discount_depth'] > self.promo_threshold
        ).astype(int)
        
        # Categorize discount depth for embedding
        prices_df['discount_bin'] = pd.cut(
            prices_df['discount_depth'],
            bins=[-0.01, 0.05, 0.15, 0.30, 1.0],
            labels=['none', 'shallow', 'medium', 'deep']
        )
        
        return prices_df
```

### 3.2 Data Quality Handling

**Anonymous Transactions:**
```python
# Transactions without CUST_CODE (cash customers)
# Option A: Drop (lose ~15-20% of data)
# Option B: Treat as single "ANONYMOUS" customer (bad - mixes behaviors)
# Option C: Use for product-level patterns only, exclude from customer modeling

# RECOMMENDATION: Option C
# - Include in price derivation (more data = better prices)
# - Exclude from customer sequence modeling (no identity)
anonymous_mask = transactions_df['CUST_CODE'].isna()
customer_transactions = transactions_df[~anonymous_mask]  # For customer model
all_transactions = transactions_df  # For price derivation
```

**Outlier Handling:**
```python
# Price outliers (data errors, extreme bundles)
def filter_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove transactions with implausible unit prices."""
    df['unit_price'] = df['SPEND'] / df['QUANTITY']
    
    # Per-product outlier detection (IQR method)
    q1 = df.groupby('PROD_CODE')['unit_price'].transform('quantile', 0.05)
    q3 = df.groupby('PROD_CODE')['unit_price'].transform('quantile', 0.95)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    valid_mask = (df['unit_price'] >= lower_bound) & (df['unit_price'] <= upper_bound)
    
    return df[valid_mask]
```

### 3.3 Data Splits

**Temporal Split (Recommended):**
```
117 weeks total (April 2006 - December 2008)

Training:   Weeks 1-80   (68%)  - Learn patterns
Validation: Weeks 81-95  (13%)  - Hyperparameter tuning
Test:       Weeks 96-117 (19%)  - Final evaluation

Rationale:
- Temporal split prevents data leakage
- Test set includes holiday season (weeks 96+)
- Validation captures seasonality transition
```

**Customer Split (For Cold-Start Evaluation):**
```
Total: 500,000 customers

Seen Customers:     400,000 (80%) - In training set
Unseen Customers:   100,000 (20%) - First appear in test set

Cold-start evaluation:
- Unseen customers have segment info (seg_1, seg_2)
- No transaction history
- Tests segment-based persona embeddings
```

---

## 4. Tensor Specifications

### 4.1 Overview: What Changed from v6

| Tensor | v6 (Complete Journey) | v7 (LGSR) | Change Reason |
|--------|----------------------|-----------|---------------|
| T1: Customer | Demographics (age, income, etc.) | Segments (seg_1, seg_2) | Demographics not available |
| T2: Transaction | Dense [801 × 3000 × F × T] | **Sparse/Ragged** [500K × var × var] | Memory: 9.8GB dense → ~500MB sparse |
| T3: Product | Same | Same | No change |
| T4: Context | Same + derived prices | Same + derived prices | Price derivation pipeline |
| T5: Coupon | Full coupon data | **REMOVED** | Not available in LGSR |
| T6: Store | Store attributes | Format + Region only | Store ID too sparse |

### 4.2 Tensor T1: Customer Representation

**Format:** Sparse lookup + learned embeddings

```python
# Customer features available in LGSR
class CustomerTensor:
    """
    Customer representation using segment-based personas.
    
    Shape: [C] → [C × d_customer]
    Where: C = 500,000 customers, d_customer = 64
    """
    
    def __init__(self, n_customers=500_000, d_model=64):
        # Segment embeddings (cold-start compatible)
        self.seg1_embedding = nn.Embedding(
            num_embeddings=10,   # ~10 lifestage segments
            embedding_dim=32
        )
        self.seg2_embedding = nn.Embedding(
            num_embeddings=15,   # ~15 lifestyle segments  
            embedding_dim=32
        )
        
        # Learned customer embeddings (for known customers)
        self.customer_embedding = nn.Embedding(
            num_embeddings=n_customers,
            embedding_dim=d_model
        )
        
        # Blend weights (learned)
        self.blend_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, customer_ids, seg1_ids, seg2_ids, transaction_counts):
        """
        Compute customer representation with cold-start handling.
        
        For new customers (0 transactions): Use segment embeddings only
        For known customers: Blend segment + learned embeddings
        """
        # Segment-based persona (always available)
        segment_repr = torch.cat([
            self.seg1_embedding(seg1_ids),  # [batch, 32]
            self.seg2_embedding(seg2_ids)   # [batch, 32]
        ], dim=-1)  # [batch, 64]
        
        # Learned customer embedding (if known)
        learned_repr = self.customer_embedding(customer_ids)  # [batch, 64]
        
        # Adaptive blending based on transaction count
        # More transactions → more weight on learned embedding
        blend_weight = self.blend_mlp(
            torch.log1p(transaction_counts.float()).unsqueeze(-1)
        )  # [batch, 1]
        
        # Cold-start: blend_weight → 0 (use segments)
        # Established: blend_weight → 1 (use learned)
        final_repr = blend_weight * learned_repr + (1 - blend_weight) * segment_repr
        
        return final_repr  # [batch, 64]
```

**Memory Estimate:**
```
Segment embeddings: (10 + 15) × 32 × 4 bytes = 3.2 KB
Customer embeddings: 500,000 × 64 × 4 bytes = 128 MB
Total: ~128 MB (manageable)
```

### 4.3 Tensor T2: Transaction History (CRITICAL - SPARSE)

**The Memory Problem:**
```python
# WRONG - Dense tensor would require:
# [500,000 customers × 5,000 products × 10 features × 117 weeks]
# = 2.9 Trillion elements = 11.7 TB (impossible)

# RIGHT - Sparse/Ragged representation
```

**Solution: Ragged Tensor Format**

```python
class TransactionTensor:
    """
    Sparse transaction history using ragged tensors.
    
    Structure:
    {
        customer_id: {
            week: [
                (product_id, quantity, spend, hour, weekday),
                ...
            ]
        }
    }
    
    Memory: O(actual_transactions) not O(customers × products × weeks)
    ~300M transactions × 5 features × 4 bytes = ~6 GB (vs 11.7 TB dense)
    """
    
    def __init__(self):
        self.history = defaultdict(lambda: defaultdict(list))
    
    def add_transaction(self, customer_id, week, product_id, 
                       quantity, spend, hour, weekday):
        self.history[customer_id][week].append(
            (product_id, quantity, spend, hour, weekday)
        )
    
    def get_customer_sequence(self, customer_id: str, 
                              max_weeks: int = 52,
                              max_items_per_week: int = 50) -> torch.Tensor:
        """
        Convert ragged history to padded tensor for transformer input.
        
        Returns: [max_weeks, max_items_per_week, features]
        """
        customer_history = self.history[customer_id]
        
        # Get most recent max_weeks
        weeks = sorted(customer_history.keys())[-max_weeks:]
        
        sequences = []
        for week in weeks:
            items = customer_history[week][:max_items_per_week]
            
            # Pad to max_items_per_week
            while len(items) < max_items_per_week:
                items.append((0, 0, 0, 0, 0))  # Padding token
            
            sequences.append(items)
        
        # Pad weeks if needed
        while len(sequences) < max_weeks:
            sequences.insert(0, [(0, 0, 0, 0, 0)] * max_items_per_week)
        
        return torch.tensor(sequences)  # [max_weeks, max_items, 5]


class EfficientTransactionLoader:
    """
    Memory-efficient batch loading for training.
    
    Strategy: Load customer histories on-demand, not all at once.
    """
    
    def __init__(self, transaction_files: List[str], batch_size: int = 256):
        self.files = transaction_files  # One file per week
        self.batch_size = batch_size
        
        # Build customer index (which weeks have data for each customer)
        self.customer_week_index = self._build_index()
    
    def _build_index(self) -> Dict[str, List[int]]:
        """Scan files to build customer → weeks mapping."""
        index = defaultdict(list)
        
        for week_file in self.files:
            week_id = self._extract_week(week_file)
            
            # Read only customer_id column
            customers = pd.read_csv(
                week_file, 
                usecols=['CUST_CODE']
            )['CUST_CODE'].dropna().unique()
            
            for cust in customers:
                index[cust].append(week_id)
        
        return index
    
    def load_customer_batch(self, customer_ids: List[str], 
                           target_week: int) -> Dict[str, torch.Tensor]:
        """
        Load transaction history for a batch of customers.
        Only loads weeks before target_week (no leakage).
        """
        batch_data = {}
        
        for cust_id in customer_ids:
            # Get weeks before target (no leakage)
            valid_weeks = [w for w in self.customer_week_index[cust_id] 
                         if w < target_week]
            
            # Load from relevant files only
            customer_history = self._load_customer_weeks(cust_id, valid_weeks)
            batch_data[cust_id] = customer_history
        
        return batch_data
```

**Memory Summary:**
```
Actual Data:
├─ 300M transaction rows
├─ ~6 features per row (product, qty, spend, hour, weekday, week)
├─ Storage: ~7.2 GB on disk (compressed parquet)
├─ In-memory per batch: ~50-100 MB

vs Dense Tensor:
├─ 500K × 5K × 117 = 292.5 Billion cells
├─ Storage: 1.17 TB (FP32)
├─ IMPOSSIBLE

Sparse wins by: 165x compression
```

### 4.4 Tensor T3: Product Catalog

**Format:** Hierarchical embeddings (unchanged from v6)

```python
class ProductTensor:
    """
    Product representations with 4-level hierarchy.
    
    Hierarchy (from data):
    D00003 (Department) → G00016 (Sub-Dept) → DEP00055 (Commodity) → 
    CL00163 (Sub-Commodity) → PRD0900032 (SKU)
    """
    
    def __init__(self, n_products=5000, d_model=256):
        # Hierarchy level embeddings
        self.sku_embedding = nn.Embedding(n_products, d_model)      # ~5000 SKUs
        self.subcommodity_embedding = nn.Embedding(300, d_model)    # ~300 CL codes
        self.commodity_embedding = nn.Embedding(100, d_model)       # ~100 DEP codes
        self.subdept_embedding = nn.Embedding(30, d_model)          # ~30 G codes
        self.dept_embedding = nn.Embedding(10, d_model)             # ~10 D codes
        
        # Hierarchy aggregation
        self.hierarchy_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, product_ids, hierarchy_ids):
        """
        Compute product embeddings with hierarchy-aware aggregation.
        
        Args:
            product_ids: [batch] - SKU codes
            hierarchy_ids: [batch, 4] - (CL, DEP, G, D) codes per product
        
        Returns:
            [batch, d_model] - Product representations
        """
        # Get all level embeddings
        sku_emb = self.sku_embedding(product_ids)  # [batch, d]
        subcommodity_emb = self.subcommodity_embedding(hierarchy_ids[:, 0])
        commodity_emb = self.commodity_embedding(hierarchy_ids[:, 1])
        subdept_emb = self.subdept_embedding(hierarchy_ids[:, 2])
        dept_emb = self.dept_embedding(hierarchy_ids[:, 3])
        
        # Stack hierarchy levels
        hierarchy_stack = torch.stack([
            sku_emb, subcommodity_emb, commodity_emb, subdept_emb, dept_emb
        ], dim=1)  # [batch, 5, d]
        
        # Attention-based aggregation
        aggregated, _ = self.hierarchy_attention(
            query=sku_emb.unsqueeze(1),  # SKU queries hierarchy
            key=hierarchy_stack,
            value=hierarchy_stack
        )
        
        return aggregated.squeeze(1)  # [batch, d]
```

### 4.5 Tensor T4: Context (Prices, Promos, Time)

**Format:** Derived from transaction data

```python
class ContextTensor:
    """
    Market context including derived prices and promos.
    
    Shape: [S × P × T × F]
    Where: S=760 stores, P=5000 products, T=117 weeks, F=context features
    
    NOTE: This IS sparse (not all products in all stores every week)
    Use same sparse strategy as T2.
    """
    
    def __init__(self):
        # Price features (derived)
        self.price_features = [
            'actual_price',      # Derived from SPEND/QUANTITY
            'base_price',        # Rolling 4-week max
            'discount_depth',    # 1 - actual/base
            'promo_flag',        # discount_depth > 0.05
        ]
        
        # Temporal features
        self.time_features = [
            'week_of_year',      # 1-52 (sinusoidal encoding)
            'is_holiday_week',   # Christmas, Easter, etc.
            'days_to_holiday',   # Countdown feature
        ]
        
        # Store features
        self.store_features = [
            'format_embed',      # LS/MS/SS embedding
            'region_embed',      # E01/E02/W01/etc. embedding
        ]
    
    def build_context(self, week: int, store_id: str) -> torch.Tensor:
        """
        Build context tensor for a specific store-week.
        
        Returns sparse tensor: [P, F] for products with data
        """
        # Load derived prices for this store-week
        prices = self.price_cache[(store_id, week)]
        
        # Temporal encoding
        week_of_year = week % 52
        time_encoding = sinusoidal_encoding(week_of_year, dim=16)
        
        # Store encoding
        store_format = self.store_metadata[store_id]['format']
        store_region = self.store_metadata[store_id]['region']
        store_encoding = torch.cat([
            self.format_embedding(store_format),
            self.region_embedding(store_region)
        ])
        
        # Combine features per product
        context_vectors = []
        for product_id in prices['product_id'].unique():
            product_prices = prices[prices['product_id'] == product_id]
            
            price_vector = torch.tensor([
                np.log(product_prices['actual_price'].values[0] + 1e-6),
                np.log(product_prices['base_price'].values[0] + 1e-6),
                product_prices['discount_depth'].values[0],
                product_prices['promo_flag'].values[0],
            ])
            
            context = torch.cat([price_vector, time_encoding, store_encoding])
            context_vectors.append((product_id, context))
        
        return context_vectors  # Sparse: List of (product_id, features)
```

### 4.6 Tensor T5: Coupons - REMOVED

```
Status: REMOVED from v7

Reason: Coupon data not available in LGSR dataset

Mitigation:
- Price sensitivity captured via BASKET_PRICE_SENSITIVITY field
- Promotional effects captured via derived discount_depth
- Personalization via seg_1/seg_2 segments

Future: If coupon data becomes available, add back per v6 spec.
```

### 4.7 Tensor T6: Store Context

```python
class StoreTensor:
    """
    Store representations using Format + Region only.
    
    Why not Store ID?
    - 760 stores with 500K customers = sparse
    - Most customers visit 1-3 stores
    - Store ID would overfit
    
    Instead: Use Format × Region as proxy for store "persona"
    """
    
    def __init__(self, d_model=32):
        # Format embeddings (3 formats)
        self.format_embedding = nn.Embedding(
            num_embeddings=3,  # LS, MS, SS
            embedding_dim=d_model // 2
        )
        
        # Region embeddings (~10 regions)
        self.region_embedding = nn.Embedding(
            num_embeddings=15,  # E01, E02, W01, W02, S01, etc.
            embedding_dim=d_model // 2
        )
    
    def forward(self, format_ids, region_ids):
        return torch.cat([
            self.format_embedding(format_ids),
            self.region_embedding(region_ids)
        ], dim=-1)  # [batch, d_model]
```

---

## 5. Model Architecture

### 5.1 Architecture Overview

**Key Changes from v6:**
- Scale for 500K customers (vs 801)
- Segment-based cold-start (vs demographics)
- Multi-task learning (basket size, price sensitivity, mission)
- Sparse tensor handling throughout

```
┌─────────────────────────────────────────────────────────────────┐
│                    RetailSim v7 Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Customer   │    │   Product    │    │   Context    │      │
│  │   Encoder    │    │   Encoder    │    │   Encoder    │      │
│  │              │    │              │    │              │      │
│  │ seg_1 + seg_2│    │ 4-level      │    │ Prices +     │      │
│  │ + history    │    │ hierarchy    │    │ promos +     │      │
│  │              │    │              │    │ time         │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └─────────┬─────────┴─────────┬─────────┘               │
│                   │                   │                         │
│                   ▼                   ▼                         │
│         ┌─────────────────────────────────────┐                │
│         │        Cross-Attention Fusion        │                │
│         │   (Customer queries Products+Context)│                │
│         └─────────────────┬───────────────────┘                │
│                           │                                     │
│                           ▼                                     │
│         ┌─────────────────────────────────────┐                │
│         │         World Model Backbone         │                │
│         │   (Transformer Encoder-Decoder)      │                │
│         │                                      │                │
│         │   Encoder: 4 layers                  │                │
│         │   Decoder: 2 layers (with KV-cache)  │                │
│         └─────────────────┬───────────────────┘                │
│                           │                                     │
│         ┌─────────┬───────┴───────┬─────────┐                  │
│         ▼         ▼               ▼         ▼                  │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│    │ Basket  │ │ Basket  │ │  Price  │ │ Mission │            │
│    │ Items   │ │  Size   │ │  Sens.  │ │  Type   │            │
│    │ (main)  │ │ (aux)   │ │ (aux)   │ │ (aux)   │            │
│    └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
│                                                                 │
│    Primary Output: Next-item prediction (5000-way softmax)     │
│    Auxiliary Outputs: Multi-task learning targets              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Model 1: Customer Encoder

```python
class CustomerEncoder(nn.Module):
    """
    Encodes customer representation from segments + transaction history.
    
    Cold-start: Uses seg_1/seg_2 only
    Warm: Blends segments with learned embeddings from history
    """
    
    def __init__(self, d_model=256, n_customers=500_000):
        super().__init__()
        
        # Segment embeddings (always available)
        self.seg1_embed = nn.Embedding(10, d_model // 4)   # 64 dim
        self.seg2_embed = nn.Embedding(15, d_model // 4)   # 64 dim
        
        # Transaction history encoder
        self.history_encoder = TransactionHistoryEncoder(d_model)
        
        # Adaptive blending
        self.blend_network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model // 2 + d_model, d_model)
    
    def forward(self, seg1, seg2, transaction_history, history_mask):
        """
        Args:
            seg1: [batch] - Lifestage segment IDs
            seg2: [batch] - Lifestyle segment IDs
            transaction_history: [batch, T, max_items, features] - Sparse history
            history_mask: [batch, T, max_items] - Valid item mask
        
        Returns:
            customer_repr: [batch, d_model]
        """
        batch_size = seg1.size(0)
        
        # Segment representation (cold-start compatible)
        segment_repr = torch.cat([
            self.seg1_embed(seg1),
            self.seg2_embed(seg2)
        ], dim=-1)  # [batch, d_model // 2]
        
        # History representation (if available)
        history_repr = self.history_encoder(transaction_history, history_mask)
        # [batch, d_model]
        
        # Count valid transactions for blend weight
        n_transactions = history_mask.sum(dim=[1, 2]).float()  # [batch]
        blend_weight = self.blend_network(
            torch.log1p(n_transactions).unsqueeze(-1)
        )  # [batch, 1]
        
        # Blend: more history → more weight on history_repr
        combined = torch.cat([
            segment_repr,
            blend_weight * history_repr
        ], dim=-1)  # [batch, d_model // 2 + d_model]
        
        return self.output_proj(combined)  # [batch, d_model]


class TransactionHistoryEncoder(nn.Module):
    """
    Encodes variable-length transaction history using transformer.
    
    Input: Ragged tensor of (product_id, quantity, spend, hour, weekday)
    Output: Fixed-size customer representation
    """
    
    def __init__(self, d_model=256, n_heads=8, n_layers=2):
        super().__init__()
        
        # Item-level encoding
        self.product_embed = nn.Embedding(5000, d_model // 2)
        self.quantity_embed = nn.Linear(1, d_model // 8)
        self.spend_embed = nn.Linear(1, d_model // 8)
        self.hour_embed = nn.Embedding(24, d_model // 8)
        self.weekday_embed = nn.Embedding(7, d_model // 8)
        
        # Temporal position encoding (which week)
        self.week_position = SinusoidalPositionEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, history, mask):
        """
        Args:
            history: [batch, T, max_items, 5] - (prod, qty, spend, hour, wday)
            mask: [batch, T, max_items] - Valid items
        
        Returns:
            [batch, d_model] - Aggregated history representation
        """
        batch_size, T, max_items, _ = history.shape
        
        # Flatten to [batch, T * max_items, features]
        history_flat = history.view(batch_size, T * max_items, -1)
        mask_flat = mask.view(batch_size, T * max_items)
        
        # Embed each feature
        product_emb = self.product_embed(history_flat[:, :, 0].long())
        quantity_emb = self.quantity_embed(history_flat[:, :, 1:2])
        spend_emb = self.spend_embed(history_flat[:, :, 2:3])
        hour_emb = self.hour_embed(history_flat[:, :, 3].long())
        weekday_emb = self.weekday_embed(history_flat[:, :, 4].long())
        
        # Combine
        item_emb = torch.cat([
            product_emb, quantity_emb, spend_emb, hour_emb, weekday_emb
        ], dim=-1)  # [batch, T * max_items, d_model]
        
        # Add week position encoding
        week_positions = torch.arange(T).unsqueeze(1).expand(T, max_items).flatten()
        week_positions = week_positions.unsqueeze(0).expand(batch_size, -1)
        item_emb = item_emb + self.week_position(week_positions)
        
        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, item_emb], dim=1)
        
        # Attention mask (CLS can attend to everything)
        cls_mask = torch.ones(batch_size, 1, device=mask.device)
        full_mask = torch.cat([cls_mask, mask_flat], dim=1)
        
        # Encode
        encoded = self.encoder(
            sequence, 
            src_key_padding_mask=~full_mask.bool()
        )
        
        # Return CLS token as representation
        return encoded[:, 0, :]  # [batch, d_model]
```

### 5.3 Model 2: World Model (Encoder-Decoder Transformer)

**Architecture: 4 Encoder + 2 Decoder Layers**

```python
class WorldModel(nn.Module):
    """
    Transformer-based world model for basket prediction.
    
    Input: Customer representation + Context (prices, promos, time)
    Output: Basket sequence + auxiliary predictions
    
    Scale: Designed for 500K customers, 5K products
    Parameters: ~15M (appropriate for data scale)
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Sub-models
        self.customer_encoder = CustomerEncoder(
            d_model=config.d_model,
            n_customers=config.n_customers
        )
        self.product_encoder = ProductTensor(
            n_products=config.n_products,
            d_model=config.d_model
        )
        self.context_encoder = ContextEncoder(
            d_model=config.d_model
        )
        
        # Cross-attention: Customer queries Products+Context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            batch_first=True
        )
        
        # Encoder (context understanding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.n_encoder_layers  # 4
        )
        
        # Decoder (basket generation)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_decoder_layers  # 2
        )
        
        # Output heads
        self.item_head = nn.Linear(config.d_model, config.n_products)  # 5000-way
        
        # Multi-task auxiliary heads
        self.basket_size_head = nn.Linear(config.d_model, 3)   # S, M, L
        self.price_sens_head = nn.Linear(config.d_model, 3)    # LA, MM, UM
        self.mission_head = nn.Linear(config.d_model, 4)       # Fresh, Grocery, Mixed, Nonfood
    
    def encode(self, customer_repr, product_catalog, context):
        """
        Encoder pass: Build context-aware product representations.
        
        Args:
            customer_repr: [batch, d_model] - From CustomerEncoder
            product_catalog: [batch, P, d_model] - Product embeddings
            context: [batch, P, d_context] - Prices, promos, time
        
        Returns:
            encoder_output: [batch, P, d_model] - Customer-conditioned products
        """
        # Fuse product and context
        product_context = torch.cat([product_catalog, context], dim=-1)
        product_context = self.context_proj(product_context)  # [batch, P, d_model]
        
        # Cross-attention: Customer queries product catalog
        customer_query = customer_repr.unsqueeze(1)  # [batch, 1, d_model]
        
        attended_products, attn_weights = self.cross_attention(
            query=customer_query,
            key=product_context,
            value=product_context
        )  # [batch, 1, d_model], [batch, 1, P]
        
        # Broadcast customer context to all products
        customer_broadcast = customer_repr.unsqueeze(1).expand(-1, product_context.size(1), -1)
        encoder_input = product_context + 0.1 * customer_broadcast  # Residual
        
        # Transformer encoder
        encoder_output = self.encoder(encoder_input)  # [batch, P, d_model]
        
        return encoder_output, attn_weights
    
    def decode(self, encoder_output, target_sequence=None, kv_cache=None):
        """
        Decoder pass: Generate basket sequence autoregressively.
        
        Training: Teacher forcing with target_sequence
        Inference: Autoregressive with KV-cache
        """
        if self.training and target_sequence is not None:
            # Teacher forcing
            return self._decode_teacher_forcing(encoder_output, target_sequence)
        else:
            # Autoregressive with KV-cache
            return self._decode_autoregressive(encoder_output, kv_cache)
    
    def _decode_teacher_forcing(self, encoder_output, target_sequence):
        """Teacher forcing for training."""
        # Embed target products
        target_embeds = self.product_encoder.sku_embedding(target_sequence)
        
        # Causal mask
        seq_len = target_sequence.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=target_sequence.device),
            diagonal=1
        ).bool()
        
        # Decode
        decoder_output = self.decoder(
            tgt=target_embeds,
            memory=encoder_output,
            tgt_mask=causal_mask
        )
        
        # Item predictions (sampled softmax loss)
        item_logits = self.item_head(decoder_output)
        
        # Auxiliary predictions (from first token)
        aux_repr = decoder_output[:, 0, :]
        basket_size_logits = self.basket_size_head(aux_repr)
        price_sens_logits = self.price_sens_head(aux_repr)
        mission_logits = self.mission_head(aux_repr)
        
        return {
            'item_logits': item_logits,
            'basket_size_logits': basket_size_logits,
            'price_sens_logits': price_sens_logits,
            'mission_logits': mission_logits
        }
    
    def _decode_autoregressive(self, encoder_output, kv_cache=None):
        """Autoregressive decoding with KV-cache for inference."""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Start with START token
        current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        generated = [current_token]
        
        if kv_cache is None:
            kv_cache = [None] * self.config.n_decoder_layers
        
        for step in range(self.config.max_basket_size):
            # Embed current token
            token_embed = self.product_encoder.sku_embedding(current_token)
            
            # Decode with cache
            decoder_output, kv_cache = self._cached_decode_step(
                token_embed, encoder_output, kv_cache
            )
            
            # Predict next item
            item_logits = self.item_head(decoder_output[:, -1:, :])
            next_token = item_logits.argmax(dim=-1)  # Greedy
            
            # Check for END token
            if (next_token == self.config.end_token_id).all():
                break
            
            generated.append(next_token)
            current_token = next_token
        
        return torch.cat(generated, dim=1)


@dataclass
class WorldModelConfig:
    """Configuration for world model."""
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    d_feedforward: int = 1024
    dropout: float = 0.1
    
    n_customers: int = 500_000
    n_products: int = 5_000
    max_basket_size: int = 50
    
    # Special tokens
    start_token_id: int = 0
    end_token_id: int = 1
    pad_token_id: int = 2
```

### 5.4 Parameter Count (Updated for LGSR Scale)

```python
def count_parameters():
    """
    Calculate parameter count for v7 architecture.
    """
    # Customer Encoder
    customer_params = {
        'seg1_embedding': 10 * 64,           # 640
        'seg2_embedding': 15 * 64,           # 960
        'history_encoder': 2 * (4 * 256**2), # ~524K (2 transformer layers)
        'blend_network': 64 + 64,            # 128
        'output_proj': 384 * 256,            # 98K
    }
    # Total: ~625K
    
    # Product Encoder
    product_params = {
        'sku_embedding': 5000 * 256,         # 1.28M
        'hierarchy_embeddings': 450 * 256,   # 115K
        'hierarchy_attention': 256 * 256 * 3,# 196K
    }
    # Total: ~1.6M
    
    # World Model Backbone
    transformer_params = {
        'encoder_layer': 4 * (4 * 256**2 + 256 * 1024 * 2),  # ~3.15M
        'decoder_layer': 2 * (4 * 256**2 + 256 * 1024 * 2 + 256**2), # ~2.2M
        'cross_attention': 256**2 * 3,       # 196K
    }
    # Total: ~5.5M
    
    # Output Heads
    output_params = {
        'item_head': 256 * 5000,             # 1.28M
        'basket_size_head': 256 * 3,         # 768
        'price_sens_head': 256 * 3,          # 768
        'mission_head': 256 * 4,             # 1K
    }
    # Total: ~1.3M
    
    # GRAND TOTAL
    total = 625_000 + 1_600_000 + 5_500_000 + 1_300_000
    # = ~9M parameters
    
    return total

# Result: ~9M parameters
# With 300M transactions → 33x parameters
# This is HEALTHY for deep learning (need 10-100x)
```

**Parameter-to-Data Ratio:**
```
Parameters: ~9M
Training signals: 300M transaction rows × ~6 predictions per basket
               = ~1.8B training signals

Ratio: 1.8B / 9M = 200x

Target: 10-100x
Actual: 200x ✅ EXCELLENT - room for larger model if needed
```

---

## 6. Training Strategy

### 6.1 Loss Function: Sampled Softmax + Multi-Task

```python
class RetailSimLoss(nn.Module):
    """
    Combined loss for basket prediction + auxiliary tasks.
    
    Components:
    1. Sampled Softmax for item prediction (handles 5K vocab sparsity)
    2. Cross-entropy for auxiliary tasks (basket size, price sens, mission)
    """
    
    def __init__(self, n_products=5000, n_samples=512, 
                 aux_weight=0.2):
        super().__init__()
        self.n_products = n_products
        self.n_samples = n_samples
        self.aux_weight = aux_weight
    
    def forward(self, model_output, targets):
        """
        Args:
            model_output: Dict with logits
            targets: Dict with ground truth labels
        
        Returns:
            total_loss, loss_breakdown
        """
        # 1. Sampled Softmax for items
        item_loss = self.sampled_softmax_loss(
            model_output['item_logits'],
            targets['item_ids']
        )
        
        # 2. Auxiliary losses (standard cross-entropy)
        basket_size_loss = F.cross_entropy(
            model_output['basket_size_logits'],
            targets['basket_size']
        )
        
        price_sens_loss = F.cross_entropy(
            model_output['price_sens_logits'],
            targets['price_sensitivity']
        )
        
        mission_loss = F.cross_entropy(
            model_output['mission_logits'],
            targets['mission']
        )
        
        aux_loss = (basket_size_loss + price_sens_loss + mission_loss) / 3
        
        # Combined loss
        total_loss = item_loss + self.aux_weight * aux_loss
        
        return total_loss, {
            'item_loss': item_loss.item(),
            'basket_size_loss': basket_size_loss.item(),
            'price_sens_loss': price_sens_loss.item(),
            'mission_loss': mission_loss.item(),
            'aux_loss': aux_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def sampled_softmax_loss(self, logits, targets):
        """
        Efficient loss for large vocabulary.
        
        Instead of computing softmax over 5000 products,
        sample 512 negatives + 1 positive.
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten for sampling
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Sample negatives per position
        sampled_indices = self._sample_negatives(targets_flat)
        # [batch*seq, n_samples+1] - includes positive
        
        # Gather sampled logits
        sampled_logits = torch.gather(
            logits_flat, 
            dim=1, 
            index=sampled_indices
        )
        
        # Positive is always at index 0
        sampled_targets = torch.zeros(
            batch_size * seq_len, 
            dtype=torch.long, 
            device=logits.device
        )
        
        # Cross-entropy on sampled set
        loss = F.cross_entropy(sampled_logits, sampled_targets)
        
        return loss
    
    def _sample_negatives(self, targets):
        """Sample negative products for each position."""
        batch_size = targets.size(0)
        
        # Random negatives
        negatives = torch.randint(
            0, self.n_products,
            (batch_size, self.n_samples),
            device=targets.device
        )
        
        # Ensure positive is included at index 0
        sampled = torch.cat([
            targets.unsqueeze(1),
            negatives
        ], dim=1)
        
        return sampled
```

### 6.2 Training Schedule

```python
class TrainingSchedule:
    """
    Multi-phase training for RetailSim v7.
    """
    
    # Phase 1: Warm-up (Weeks 1-10)
    phase1 = {
        'epochs': 5,
        'learning_rate': 1e-4,
        'batch_size': 256,
        'warmup_steps': 10000,
        'description': 'Learn basic patterns with frozen product embeddings'
    }
    
    # Phase 2: Main Training (Weeks 11-70)
    phase2 = {
        'epochs': 15,
        'learning_rate': 3e-4,
        'batch_size': 512,
        'scheduler': 'cosine',
        'description': 'Full training with all parameters'
    }
    
    # Phase 3: Fine-tuning (Weeks 71-80)
    phase3 = {
        'epochs': 3,
        'learning_rate': 1e-5,
        'batch_size': 512,
        'description': 'Fine-tune on recent data before evaluation'
    }
```

### 6.3 Data Loading Strategy

```python
class EfficientDataLoader:
    """
    Memory-efficient loading for 300M transactions.
    
    Strategy:
    1. Keep data on disk (parquet files per week)
    2. Load batches on-demand
    3. Cache recent weeks in memory
    4. Use memory-mapped files for large lookups
    """
    
    def __init__(self, data_dir: str, cache_weeks: int = 10):
        self.data_dir = data_dir
        self.cache_weeks = cache_weeks
        
        # File index (which weeks exist)
        self.week_files = self._scan_week_files()
        
        # LRU cache for recent weeks
        self.week_cache = OrderedDict()
        
        # Customer index (memory-mapped)
        self.customer_index = self._build_customer_index()
    
    def _scan_week_files(self) -> Dict[int, str]:
        """Build mapping of week_id → file_path."""
        files = {}
        for f in Path(self.data_dir).glob('transactions_*.parquet'):
            week_id = int(f.stem.split('_')[1])
            files[week_id] = str(f)
        return files
    
    def _build_customer_index(self) -> np.memmap:
        """
        Build memory-mapped index: customer_id → (file_offsets).
        Allows O(1) lookup of customer transactions.
        """
        # Implementation: scan all files, build index, save to disk
        # Load as memmap for memory efficiency
        pass
    
    def get_batch(self, customer_ids: List[str], target_week: int) -> Dict:
        """
        Load a training batch.
        
        Returns:
            Dict with customer histories, targets, context
        """
        batch = {
            'customer_ids': customer_ids,
            'histories': [],
            'targets': [],
            'context': []
        }
        
        for cust_id in customer_ids:
            # Load history (weeks before target_week only - no leakage)
            history = self._load_customer_history(
                cust_id, 
                max_week=target_week - 1
            )
            
            # Load target basket
            target = self._load_customer_basket(cust_id, target_week)
            
            # Load context (prices, promos for target_week)
            context = self._load_week_context(target_week)
            
            batch['histories'].append(history)
            batch['targets'].append(target)
            batch['context'].append(context)
        
        return self._collate_batch(batch)
```

### 6.4 Distributed Training (Optional)

```python
# For faster training on multiple GPUs

class DistributedTrainer:
    """
    Multi-GPU training using PyTorch DDP.
    
    With 300M transactions and ~9M parameters:
    - Single GPU (A100): ~48-72 hours
    - 4x GPU (DDP): ~12-18 hours
    - 8x GPU (DDP): ~6-9 hours
    """
    
    def __init__(self, model, world_size=4):
        self.model = DDP(model)
        self.world_size = world_size
    
    def train_epoch(self, dataloader):
        # Standard DDP training loop
        pass
```

---

## 7. Multi-Agent RL System

### 7.1 Overview

**Agents remain similar to v6, with adjustments for LGSR data:**

| Agent | State | Actions | Data Source |
|-------|-------|---------|-------------|
| Pricing Agent | Current prices, demand forecast | Price per product | Derived prices |
| Promo Agent | Promo budget, product lift history | Which products to promote | Derived promo flags |
| Inventory Agent | Stock levels, demand forecast | Order quantities | QUANTITY field |

### 7.2 Pricing Agent (Updated for Derived Prices)

```python
class PricingAgentV7(nn.Module):
    """
    Pricing agent adapted for derived price data.
    
    Key difference from v6:
    - No explicit MSRP (use 95th percentile as proxy)
    - No explicit cost (use 20th percentile as proxy for minimum margin)
    - Constraints derived from observed price distribution
    """
    
    def __init__(self, n_products=5000, d_model=128):
        super().__init__()
        
        # Price bounds (computed from data)
        self.register_buffer('price_min', torch.zeros(n_products))  # 20th percentile
        self.register_buffer('price_max', torch.zeros(n_products))  # 95th percentile
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_products)
        )
    
    def compute_price_bounds(self, historical_prices: pd.DataFrame):
        """
        Compute price bounds from historical data.
        
        Assumption:
        - Min viable price = 20th percentile (covers cost + thin margin)
        - Max viable price = 95th percentile (MSRP proxy)
        """
        for product_id in range(self.n_products):
            product_prices = historical_prices[
                historical_prices['product_id'] == product_id
            ]['actual_price']
            
            if len(product_prices) > 0:
                self.price_min[product_id] = product_prices.quantile(0.20)
                self.price_max[product_id] = product_prices.quantile(0.95)
            else:
                # Fallback: global percentiles
                self.price_min[product_id] = historical_prices['actual_price'].quantile(0.20)
                self.price_max[product_id] = historical_prices['actual_price'].quantile(0.95)
    
    def forward(self, state):
        """
        Output prices within learned bounds.
        
        Uses squashing to guarantee constraint satisfaction.
        """
        # Raw policy output (unbounded)
        raw_prices = self.policy(state)  # [batch, n_products]
        
        # Squash to [0, 1] then scale to [min, max]
        normalized = torch.sigmoid(raw_prices)  # [0, 1]
        
        prices = (
            self.price_min + 
            normalized * (self.price_max - self.price_min)
        )
        
        return prices
```

### 7.3 Reward Shaping (Unchanged from v6)

```python
class PricingReward:
    """
    Multi-component reward to prevent constraint exploitation.
    
    Components:
    1. Profit (primary)
    2. Boundary penalty (discourage edge-hugging)
    3. Diversity bonus (varied margins)
    4. Inventory alignment (clear excess stock)
    """
    
    def compute_reward(self, prices, demand, costs_proxy, inventory):
        # Profit (primary objective)
        revenue = prices * demand
        cost = costs_proxy * demand  # Using derived min prices as cost proxy
        profit = (revenue - cost).sum()
        
        # Boundary penalty
        price_range = self.price_max - self.price_min
        normalized_position = (prices - self.price_min) / (price_range + 1e-6)
        boundary_penalty = (1 - 4 * normalized_position * (1 - normalized_position)).mean()
        
        # Diversity bonus
        diversity_bonus = normalized_position.std()
        
        # Inventory alignment
        excess_inventory = (inventory > 14).float()
        inventory_bonus = (excess_inventory * (1 - normalized_position)).mean()
        
        total_reward = (
            profit 
            - 0.1 * boundary_penalty 
            + 0.05 * diversity_bonus
            + 0.05 * inventory_bonus
        )
        
        return total_reward
```

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Data Pipeline (Weeks 1-4)

```
Week 1-2: Data Ingestion
├─ Download full LGSR dataset (117 weeks)
├─ Set up storage (parquet files per week)
├─ Build week_id → file_path index
├─ Profile data quality (null rates, outliers)
└─ Deliverable: Data lake with 117 parquet files

Week 3-4: Price Derivation Pipeline
├─ Implement unit_price = SPEND / QUANTITY
├─ Implement rolling 4-week base price
├─ Implement waterfall imputation
├─ Implement promo detection (>5% discount)
├─ Validate against common-sense checks
└─ Deliverable: Price/promo tensors for all store-product-weeks
```

### 8.2 Phase 2: Feature Engineering (Weeks 5-6)

```
Week 5: Customer Features
├─ Extract seg_1/seg_2 mappings
├─ Build customer → transaction history index
├─ Compute customer-level aggregates (RFM-like)
├─ Validate cold-start handling
└─ Deliverable: Customer feature store

Week 6: Product & Context Features
├─ Build product hierarchy mappings (PRD → CL → DEP → G → D)
├─ Compute product-level price statistics
├─ Generate temporal features (week_of_year, holidays)
├─ Build store format/region mappings
└─ Deliverable: Product and context feature stores
```

### 8.3 Phase 3: Model Development (Weeks 7-12)

```
Week 7-8: Customer Encoder
├─ Implement segment embeddings
├─ Implement transaction history encoder
├─ Implement adaptive blending
├─ Unit tests for cold-start behavior
└─ Deliverable: Tested CustomerEncoder module

Week 9-10: World Model Backbone
├─ Implement encoder (4 layers)
├─ Implement decoder (2 layers) with KV-cache
├─ Implement sampled softmax loss
├─ Implement multi-task heads
└─ Deliverable: Tested WorldModel module

Week 11-12: Training Loop
├─ Implement EfficientDataLoader
├─ Implement training schedule (3 phases)
├─ Implement validation metrics (P@10, P@20)
├─ Set up experiment tracking (MLflow)
└─ Deliverable: End-to-end training pipeline
```

### 8.4 Phase 4: Training & Evaluation (Weeks 13-16)

```
Week 13-14: World Model Training
├─ Train on weeks 1-80
├─ Validate on weeks 81-95
├─ Hyperparameter tuning
├─ Monitor for overfitting
└─ Target: P@10 > 0.60

Week 15-16: Evaluation
├─ Test on weeks 96-117
├─ Cold-start evaluation (unseen customers)
├─ Interpretability analysis (attention weights)
├─ Generate evaluation report
└─ Deliverable: Trained world model with evaluation metrics
```

### 8.5 Phase 5: RL Integration (Weeks 17-20)

```
Week 17-18: RL Environment
├─ Implement Gymnasium wrapper
├─ Integrate world model as simulator
├─ Implement pricing agent (PPO)
├─ Implement reward shaping
└─ Deliverable: RL training environment

Week 19-20: RL Training & Demo
├─ Train pricing agent
├─ Validate against heuristic baselines
├─ Build Streamlit demo
├─ Create counterfactual scenarios
└─ Deliverable: Working RetailSim demo
```

---

## 9. Success Metrics

### 9.1 World Model Metrics

| Metric | Target | Stretch | Notes |
|--------|--------|---------|-------|
| **Precision@10** | > 0.60 | > 0.68 | Core basket prediction accuracy |
| **Recall@20** | > 0.55 | > 0.62 | Coverage of actual basket |
| **MRR** | > 0.35 | > 0.42 | Mean reciprocal rank |
| **Cold-Start P@10** | > 0.40 | > 0.50 | Unseen customers (segment only) |
| **Basket Size Accuracy** | > 0.70 | > 0.80 | Auxiliary task |
| **Price Sens. Accuracy** | > 0.65 | > 0.75 | Auxiliary task |

### 9.2 RL Agent Metrics

| Metric | Target | Stretch | Notes |
|--------|--------|---------|-------|
| **Profit Lift** | > 5% | > 10% | vs. historical pricing |
| **Revenue Stability** | < 8% variance | < 5% | Week-over-week |
| **Price Violations** | 0% | 0% | Must be zero |
| **Constraint Exploitation** | < 20% at edges | < 10% | Healthy exploration |

### 9.3 System Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Training Time** | < 48 hours | Single GPU (A100) |
| **Inference Latency** | < 2 sec/step | With KV-cache |
| **Memory Usage** | < 16 GB | During training |
| **Data Pipeline** | < 24 hours | Full preprocessing |

---

## 10. Risk Mitigation

### 10.1 Data Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Derived prices inaccurate | Medium | High | Validate against business rules; sanity checks |
| Missing customer segments | Low | Medium | Fallback to global segment |
| Data quality issues | Medium | Medium | Extensive EDA; outlier detection |
| Temporal leakage | Medium | High | Strict train/val/test splits by week |

### 10.2 Model Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Overfitting | Low | High | 200x data-to-param ratio; regularization |
| Cold-start failure | Medium | Medium | Segment embeddings; adaptive blending |
| Slow convergence | Medium | Medium | Warm-up phase; learning rate schedule |
| Memory issues | Low | Medium | Sparse tensors; efficient data loading |

### 10.3 RL Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| World model errors compound | High | High | Conservative RL; ensemble world models |
| Constraint exploitation | Medium | Medium | Reward shaping; boundary penalties |
| Reward hacking | Medium | High | Multi-component rewards; human validation |

---

## 11. Appendices

### Appendix A: Segment Code Mappings (To Be Validated)

```
seg_1 (Lifestage):
├─ CT: Couples/Traditional
├─ AZ: Active/Young
├─ BG: Budget/Growing
├─ DY: Dynamic/Young professionals
└─ (Others to be discovered in EDA)

seg_2 (Lifestyle):
├─ DI: Discerning
├─ FN: Foodie/Natural
├─ BU: Budget
├─ CZ: Convenient/Zealous
├─ EQ: Economy
├─ AT: Affluent/Traditional
└─ (Others to be discovered in EDA)
```

### Appendix B: Hierarchy Mappings

```
Product Hierarchy (from data):
D00003 (Department, ~10 codes)
└─ G00016 (Sub-Department, ~30 codes)
   └─ DEP00055 (Commodity, ~100 codes)
      └─ CL00163 (Sub-Commodity, ~300 codes)
         └─ PRD0900032 (SKU, ~5000 codes)

Store Hierarchy:
STORE_REGION (E01, E02, W01, W02, S01, ...)
└─ STORE_FORMAT (LS, MS, SS)
   └─ STORE_CODE (STORE00001, ..., STORE00760)
```

### Appendix C: File Format Specifications

```
Transaction Files (per week):
├─ Format: Parquet (compressed)
├─ Naming: transactions_YYYYWW.parquet (e.g., transactions_200626.parquet)
├─ Size: ~50-100 MB per week
├─ Total: ~6-12 GB for 117 weeks

Derived Price Files:
├─ Format: Parquet
├─ Naming: prices_YYYYWW.parquet
├─ Columns: product_id, store_id, actual_price, base_price, discount_depth, promo_flag

Feature Stores:
├─ Customer features: customers.parquet (~50 MB)
├─ Product features: products.parquet (~5 MB)
├─ Store features: stores.parquet (~1 MB)
```

---

## Document Metadata

**Version:** 7.0  
**Status:** Implementation Ready  
**Last Updated:** November 2025  
**Authors:** KVSN & Product Leadership

**Changelog v6.x → v7.0:**
- **MAJOR:** Pivoted from Dunnhumby Complete Journey (801 users) to LGSR (500K users)
- Added derived price/promo pipeline specification
- Updated tensor specs for sparse representation (T2)
- Replaced demographics with segment embeddings (T1)
- Removed coupon tensor (T5) - not available in LGSR
- Updated parameter calculations for new scale (~9M params)
- Updated success metrics for LGSR characteristics
- Added multi-task learning (basket size, price sensitivity, mission)

**Key Decisions:**
- Sparse tensor format mandatory for T2 (memory)
- Segment-based cold-start replaces demographics
- Sampled softmax (512 samples) for 5K vocabulary
- 4+2 encoder-decoder depth appropriate for scale
- Derived prices from SPEND/QUANTITY with waterfall imputation
