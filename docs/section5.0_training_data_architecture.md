# Training Data Architecture: Current vs Proposed

## Executive Summary

This document compares the current disconnected data pipeline with a proposed unified architecture for world model training.

| Aspect | Current | Proposed |
|--------|---------|----------|
| Train/Val/Test Splits | Two separate systems | Unified temporal splits |
| Memory Usage | 10GB (optimized) | 10GB (same) |
| Encoding Overhead | Per-batch CPU encoding | Cached static + minimal dynamic |
| Disk Space | ~0 GB | ~20-50 GB tensor cache |
| Code Complexity | Two parallel systems | Single integrated pipeline |

---

## Current Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT STATE (Disconnected)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PATH A: data_preparation/ (Section 4.7)                                    │
│  ════════════════════════════════════════                                   │
│                                                                             │
│  raw_data/transactions.csv                                                  │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────┐                                                    │
│  │ stage1_temporal     │ → data/prepared/temporal_metadata.parquet          │
│  │ (split assignment)  │   (basket → train/val/test assignment)             │
│  └─────────────────────┘                                                    │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────┐                                                    │
│  │ stage2_customer     │ → data/prepared/customer_histories.parquet         │
│  │ (history extraction)│   (per-basket history sequences)                   │
│  └─────────────────────┘                                                    │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────┐    data/prepared/samples/                          │
│  │ stage3_training     │ → ├── train_bucket_1_samples.parquet               │
│  │ (bucketed samples)  │   ├── train_bucket_2_samples.parquet               │
│  └─────────────────────┘   ├── ...                                          │
│           │                ├── validation_samples.parquet                   │
│           ▼                └── test_samples.parquet                         │
│  ┌─────────────────────┐                                                    │
│  │ stage4_tensor_cache │ → data/prepared/tensor_cache/                      │
│  │ (static embeddings) │   ├── vocab.json                                   │
│  └─────────────────────┘   ├── product_embeddings.pt                        │
│           │                ├── store_embeddings.pt                          │
│           ▼                └── ...                                          │
│  ┌─────────────────────┐                                                    │
│  │ dataloader.py       │ → RetailSimDataset (loads prepared samples)        │
│  │ (NOT using T1-T6)   │   BucketBatchSampler                               │
│  └─────────────────────┘   BUT: Does NOT use tensor_preparation encoders!   │
│                                                                             │
│                                                                             │
│  PATH B: tensor_preparation/ (Section 4)                                    │
│  ═══════════════════════════════════════                                    │
│                                                                             │
│  raw_data/transactions.csv ◄──────────────────┐                             │
│           │                                   │                             │
│           ▼                                   │ Loads directly,             │
│  data/features/                               │ ignores prepared/           │
│  ├── product_embeddings.pkl                   │                             │
│  ├── customer_embeddings.parquet              │                             │
│  └── ...                                      │                             │
│           │                                   │                             │
│           ▼                                   │                             │
│  ┌─────────────────────┐                      │                             │
│  │ T1-T6 Encoders      │ Proper tensor specs  │                             │
│  │ (CustomerContext,   │ 192d, 256d, 64d...   │                             │
│  │  ProductSequence,   │                      │                             │
│  │  TemporalContext,   │                      │                             │
│  │  etc.)              │                      │                             │
│  └─────────────────────┘                      │                             │
│           │                                   │                             │
│           ▼                                   │                             │
│  ┌─────────────────────┐                      │                             │
│  │ dataset_optimized.py│ Memory efficient     │                             │
│  │ RetailSimDataset    │ 10GB for 223M rows   │                             │
│  │ Optimized           │ BUT: No train/val/   │                             │
│  └─────────────────────┘ test splits!         │                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Problems with Current Architecture

#### Problem 1: Two Disconnected Systems

```python
# data_preparation/dataloader.py - Has train/val/test splits but basic encoding
class RetailSimDataset:
    def __init__(self, samples_path, tensor_cache_path, ...):
        self.samples = pd.read_parquet(samples_path)  # Has proper splits!
        # BUT: Uses simple embedding lookups, not T1-T6 encoders

# tensor_preparation/dataset_optimized.py - Has proper T1-T6 encoding but no splits
class RetailSimDatasetOptimized:
    def __init__(self, project_root, ...):
        self.transactions = pd.read_csv('raw_data/transactions.csv')  # Raw data!
        # Has T1-T6 encoders BUT: No train/val/test splits!
```

#### Problem 2: No Temporal Splits in tensor_preparation

The optimized dataset loads ALL transactions and has no concept of:
- Training set (weeks 1-80)
- Validation set (weeks 81-95)
- Test set (weeks 96-117)

This means **data leakage** if used directly for training - the model could see future data!

#### Problem 3: data_preparation Doesn't Use Proper Tensor Specs

The `data_preparation/dataloader.py` doesn't use the T1-T6 tensor encoders that implement:
- Cold-start handling for customers
- BERT-style masking for products
- Proper positional encodings
- Trip context for auxiliary prediction

#### Problem 4: Duplicate Work

Both systems:
- Load transactions separately
- Build their own embedding lookups
- Have their own batch collation logic

---

## Proposed Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROPOSED ARCHITECTURE (Unified)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: Data Preparation (Run Once)                                       │
│  ═════════════════════════════════════                                      │
│                                                                             │
│  raw_data/transactions.csv                                                  │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │ Temporal Split Assignment                               │                │
│  │ - Normalize SHOP_WEEK (YYYYWW → sequential 1-117)       │                │
│  │ - Assign: train (1-80), val (81-95), test (96-117)      │                │
│  │ - Flag first-time customers per split                   │                │
│  └─────────────────────────────────────────────────────────┘                │
│           │                                                                 │
│           ▼                                                                 │
│  data/prepared/                                                             │
│  ├── train_samples.parquet      # Basket IDs + metadata for weeks 1-80     │
│  ├── validation_samples.parquet # Basket IDs + metadata for weeks 81-95    │
│  └── test_samples.parquet       # Basket IDs + metadata for weeks 96-117   │
│                                                                             │
│                                                                             │
│  STAGE 2: Tensor Cache (Run Once)                                           │
│  ═════════════════════════════════                                          │
│                                                                             │
│  data/features/*.parquet + *.pkl                                            │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │ Pre-compute Static Embeddings                           │                │
│  │ - Product embeddings matrix [N_products, 256]           │                │
│  │ - Customer embeddings matrix [N_customers, 160]         │                │
│  │ - Store embeddings matrix [N_stores, 96]                │                │
│  │ - Segment embeddings [N_segments, 64]                   │                │
│  │ - Positional encodings [max_seq_len, 256]               │                │
│  └─────────────────────────────────────────────────────────┘                │
│           │                                                                 │
│           ▼                                                                 │
│  data/prepared/tensor_cache/                                                │
│  ├── product_embeddings.npy     # [5000, 256] - memory-mapped              │
│  ├── customer_embeddings.npy    # [100000, 160] - memory-mapped            │
│  ├── store_embeddings.npy       # [500, 96]                                │
│  ├── positional_encodings.npy   # [50, 256]                                │
│  └── vocab.json                 # ID → index mappings                      │
│                                                                             │
│                                                                             │
│  STAGE 3: Training DataLoader (Used During Training)                        │
│  ═══════════════════════════════════════════════════                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │ WorldModelDataset                                       │                │
│  │                                                         │                │
│  │ Inputs:                                                 │                │
│  │ - split: 'train' | 'validation' | 'test'                │                │
│  │ - tensor_cache_path: data/prepared/tensor_cache/        │                │
│  │ - transactions: raw_data/transactions.csv (optimized)   │                │
│  │                                                         │                │
│  │ On Init:                                                │                │
│  │ 1. Load sample indices for split (memory-efficient)     │                │
│  │ 2. Memory-map static embeddings (zero copy)             │                │
│  │ 3. Load transactions with category dtypes               │                │
│  │ 4. Initialize VectorizedTensorEncoder                   │                │
│  │                                                         │                │
│  │ On __getitem__(idx):                                    │                │
│  │ 1. Get basket_id from split samples                     │                │
│  │ 2. Lookup static embeddings (matrix indexing)           │                │
│  │ 3. Encode dynamic features (temporal, price)            │                │
│  │ 4. Return complete T1-T6 tensors                        │                │
│  └─────────────────────────────────────────────────────────┘                │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │ WorldModelDataLoader                                    │                │
│  │                                                         │                │
│  │ Features:                                               │                │
│  │ - Bucket batching (similar sequence lengths)            │                │
│  │ - BERT-style masking (configurable %)                   │                │
│  │ - Prefetching for GPU overlap                           │                │
│  │ - Multi-worker loading                                  │                │
│  │                                                         │                │
│  │ Yields: WorldModelBatch                                 │                │
│  │ - customer_context [B, 192]    T1                       │                │
│  │ - product_embeddings [B, S, 256] T2                     │                │
│  │ - temporal_context [B, 64]     T3                       │                │
│  │ - price_features [B, S, 64]    T4                       │                │
│  │ - store_context [B, 96]        T5                       │                │
│  │ - trip_context [B, 48]         T6                       │                │
│  │ - attention_mask [B, S]                                 │                │
│  │ - masked_positions, masked_targets (for MLM)            │                │
│  │ - trip_labels (for auxiliary prediction)                │                │
│  └─────────────────────────────────────────────────────────┘                │
│           │                                                                 │
│           ▼                                                                 │
│       Training Loop                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Sample Index Files (Lightweight)

Instead of storing full tensors, store only basket IDs and metadata:

```python
# train_samples.parquet (~500MB for 20M baskets)
columns = [
    'basket_id',           # Primary key
    'customer_id',         # For embedding lookup
    'store_id',            # For embedding lookup
    'shop_week',           # For temporal encoding
    'shop_weekday',        # For temporal encoding
    'shop_hour',           # For temporal encoding
    'n_products',          # For bucketing by sequence length
    'history_length',      # For cold-start handling
    'bucket_id',           # Pre-computed bucket assignment
]
```

#### 2. Static Embedding Cache (Memory-Mapped)

```python
# tensor_cache/ structure
product_embeddings.npy    # Shape: [N_products + 3, 256]  ~5MB
                          # Index 0: PAD, 1: UNK, 2: MASK, 3+: products

customer_embeddings.npy   # Shape: [N_customers + 1, 160] ~64MB
                          # Index 0: UNK, 1+: customers

store_embeddings.npy      # Shape: [N_stores + 1, 96]     ~200KB
                          # Pre-computed from StoreContextEncoder

segment_embeddings.npy    # Shape: [N_segments, 64]       ~50KB
                          # For cold-start blending

positional_encodings.npy  # Shape: [max_seq_len, 256]     ~50KB
                          # Pre-computed sinusoidal

vocab.json                # ID → index mappings
{
    "products": {"PROD001": 3, "PROD002": 4, ...},
    "customers": {"CUST001": 1, "CUST002": 2, ...},
    "stores": {"STORE001": 1, ...}
}
```

#### 3. WorldModelDataset

```python
class WorldModelDataset:
    """Unified dataset for world model training."""

    def __init__(
        self,
        split: str,  # 'train', 'validation', 'test'
        project_root: Path,
        max_seq_len: int = 50,
        mask_prob: float = 0.15,
    ):
        # Load split-specific sample indices (lightweight)
        self.samples = pd.read_parquet(
            project_root / f'data/prepared/{split}_samples.parquet'
        )

        # Memory-map static embeddings (zero-copy)
        cache_dir = project_root / 'data/prepared/tensor_cache'
        self.product_embeds = np.load(cache_dir / 'product_embeddings.npy', mmap_mode='r')
        self.customer_embeds = np.load(cache_dir / 'customer_embeddings.npy', mmap_mode='r')
        self.store_embeds = np.load(cache_dir / 'store_embeddings.npy', mmap_mode='r')

        # Load vocab for ID → index mapping
        with open(cache_dir / 'vocab.json') as f:
            self.vocab = json.load(f)

        # Load transactions with optimized dtypes (for product sequences)
        self.transactions = pd.read_csv(
            project_root / 'raw_data/transactions.csv',
            dtype=TRANSACTION_DTYPES
        )

        # Build basket → transaction index mapping
        self._build_basket_index()

        # Initialize dynamic encoder (for temporal, trip context)
        self.encoder = VectorizedTensorEncoder()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]

        # T1: Customer context (static lookup + cold-start handling)
        cust_idx = self.vocab['customers'].get(sample['customer_id'], 0)
        customer_embed = self.customer_embeds[cust_idx]
        t1 = self._encode_customer_context(customer_embed, sample['history_length'])

        # T2: Product sequence (static lookup + positional encoding)
        product_ids = self._get_basket_products(sample['basket_id'])
        t2 = self._encode_product_sequence(product_ids)

        # T3: Temporal context (dynamic encoding)
        t3 = self.encoder.encode_temporal(
            sample['shop_week'], sample['shop_weekday'], sample['shop_hour']
        )

        # T4: Price context (dynamic encoding)
        t4 = self._encode_price_context(sample['basket_id'])

        # T5: Store context (static lookup)
        store_idx = self.vocab['stores'].get(sample['store_id'], 0)
        t5 = self.store_embeds[store_idx]

        # T6: Trip context (dynamic encoding)
        t6 = self.encoder.encode_trip(sample)

        return {
            't1': t1, 't2': t2, 't3': t3, 't4': t4, 't5': t5, 't6': t6,
            'attention_mask': ...,
            'trip_labels': ...,
        }
```

#### 4. WorldModelDataLoader

```python
class WorldModelDataLoader:
    """DataLoader with bucket batching and masking."""

    def __init__(
        self,
        dataset: WorldModelDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        mask_prob: float = 0.15,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mask_prob = mask_prob

        # Create bucket sampler for efficient batching
        self.sampler = BucketBatchSampler(
            dataset.samples['n_products'].values,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def __iter__(self):
        for batch_indices in self.sampler:
            # Collate samples into batch
            batch = self._collate([self.dataset[i] for i in batch_indices])

            # Apply BERT-style masking
            if self.mask_prob > 0:
                batch = self._apply_masking(batch)

            yield batch
```

---

## Comparison

### Memory Usage

| Dataset Size | Current (tensor_prep) | Proposed |
|--------------|----------------------|----------|
| 100K transactions | 0.01 GB | 0.01 GB |
| 10M transactions | 1 GB | 1 GB |
| 223M transactions | 10 GB | 10 GB + 70MB cache |

Memory usage is essentially the same - the tensor cache is memory-mapped so it doesn't increase RAM usage.

### Disk Space

| Component | Current | Proposed |
|-----------|---------|----------|
| Raw transactions | 29.6 GB | 29.6 GB |
| Feature files | ~2 GB | ~2 GB |
| Prepared samples | 0 | ~2 GB |
| Tensor cache | 0 | ~100 MB |
| **Total Additional** | 0 | ~2.1 GB |

### Training Speed

| Operation | Current | Proposed |
|-----------|---------|----------|
| Batch encoding | 10-12 ms | 3-5 ms |
| Embedding lookup | Dict lookup | Matrix index |
| Temporal encoding | Per-batch | Per-batch (same) |
| Split filtering | None (data leak!) | Pre-filtered |

### Code Complexity

| Aspect | Current | Proposed |
|--------|---------|----------|
| DataLoader classes | 2 separate | 1 unified |
| Embedding systems | 2 separate | 1 shared cache |
| Split logic | Duplicated | Single source |
| Maintenance | Update 2 places | Update 1 place |

---

## Implementation Plan

### Phase 1: Create Unified Sample Index (1-2 hours)

```bash
# New file: src/training/prepare_samples.py
python -m src.training.prepare_samples

# Output:
# data/prepared/train_samples.parquet
# data/prepared/validation_samples.parquet
# data/prepared/test_samples.parquet
```

### Phase 2: Create Tensor Cache (30 min)

```bash
# New file: src/training/prepare_tensor_cache.py
python -m src.training.prepare_tensor_cache

# Output:
# data/prepared/tensor_cache/product_embeddings.npy
# data/prepared/tensor_cache/customer_embeddings.npy
# data/prepared/tensor_cache/store_embeddings.npy
# data/prepared/tensor_cache/vocab.json
```

### Phase 3: Create WorldModelDataset (2-3 hours)

```bash
# New file: src/training/dataset.py
# - WorldModelDataset
# - WorldModelDataLoader
# - WorldModelBatch
```

### Phase 4: Integration Tests (1 hour)

```bash
# New file: src/training/test_dataloader.py
python -m src.training.test_dataloader

# Validates:
# - Correct tensor shapes
# - No data leakage between splits
# - Masking works correctly
# - Batch timing benchmarks
```

### Phase 5: Training Script (2-3 hours)

```bash
# New file: src/training/train_world_model.py
python -m src.training.train_world_model \
    --batch-size 64 \
    --epochs 10 \
    --learning-rate 1e-4
```

---

## Directory Structure After Implementation

```
retail_sim/
├── src/
│   ├── data_preparation/      # Keep for reference, but deprecated for training
│   ├── tensor_preparation/    # Keep T1-T6 encoders, dataset_optimized
│   └── training/              # NEW: Unified training pipeline
│       ├── __init__.py
│       ├── prepare_samples.py      # Stage 1: Create split samples
│       ├── prepare_tensor_cache.py # Stage 2: Cache static embeddings
│       ├── dataset.py              # WorldModelDataset, DataLoader
│       ├── test_dataloader.py      # Validation tests
│       └── train_world_model.py    # Training script
│
├── data/
│   ├── features/              # From feature_engineering (existing)
│   └── prepared/              # Training-ready data
│       ├── train_samples.parquet
│       ├── validation_samples.parquet
│       ├── test_samples.parquet
│       └── tensor_cache/
│           ├── product_embeddings.npy
│           ├── customer_embeddings.npy
│           ├── store_embeddings.npy
│           └── vocab.json
```

---

## Questions for Review

### 1. Bucket Batching vs Dynamic Padding

**Question**: Should we bucket by sequence length or use dynamic padding?

**Option A: Bucket Batching (Recommended)**
```
Bucket 1: sequences 1-10 items   → pad to 10
Bucket 2: sequences 11-20 items  → pad to 20
Bucket 3: sequences 21-35 items  → pad to 35
Bucket 4: sequences 36-50 items  → pad to 50
```

| Aspect | Bucket Batching | Dynamic Padding |
|--------|-----------------|-----------------|
| Padding waste | Low (10-20%) | High (50-80%) |
| GPU efficiency | High | Lower |
| Implementation | More complex | Simple |
| Batch composition | Similar lengths | Random lengths |

**Recommendation**: Bucket batching. Retail baskets have high variance (1-50+ items), so dynamic padding wastes significant compute on padding tokens.

**Trade-off**: Bucket batching means samples aren't fully shuffled across lengths. For training stability, we shuffle within buckets and rotate bucket order each epoch.

---

### 2. Multi-GPU / Distributed Training

**Question**: Do you need distributed training support?

**Option A: Single GPU**
```python
dataloader = WorldModelDataLoader(dataset, batch_size=64)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
```

**Option B: Multi-GPU DataParallel (Simple)**
```python
model = nn.DataParallel(model)  # Wraps model
dataloader = WorldModelDataLoader(dataset, batch_size=64 * n_gpus)
# Same training loop, PyTorch handles distribution
```

**Option C: Multi-GPU DistributedDataParallel (Recommended for scale)**
```python
# Requires launch with torchrun
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = WorldModelDataLoader(dataset, batch_size=64, sampler=sampler)
model = DistributedDataParallel(model, device_ids=[local_rank])
```

| Aspect | Single GPU | DataParallel | DistributedDataParallel |
|--------|-----------|--------------|------------------------|
| Setup complexity | None | Low | Medium |
| Scaling efficiency | 1x | 0.7-0.8x per GPU | 0.9-0.95x per GPU |
| Memory per GPU | Full model | Full model | Full model |
| Communication | None | GPU→CPU→GPU | GPU↔GPU (NCCL) |
| Recommended for | Dev/debug | 2-4 GPUs | 4+ GPUs |

**Recommendation**:
- Start with single GPU for development
- Add DistributedDataParallel support for production training
- I'll implement both options with a `--distributed` flag

---

### 3. Streaming vs In-Memory Loading

**Question**: For 223M rows (10GB), should we support streaming/chunked loading?

**Option A: Full In-Memory (Current)**
```python
# Load all transactions at init
self.transactions = pd.read_csv('transactions.csv', dtype=TRANSACTION_DTYPES)
# Memory: 10GB, Load time: 5min, Access: O(1)
```

**Option B: Memory-Mapped File**
```python
# Convert CSV to memory-mapped format once
# Then access without loading into RAM
self.transactions = np.memmap('transactions.npy', mode='r')
# Memory: ~100MB, Load time: instant, Access: O(1)
```

**Option C: Chunked/Streaming**
```python
# Load chunks on demand
def __getitem__(self, idx):
    chunk_id = idx // chunk_size
    if chunk_id != self.current_chunk_id:
        self.current_chunk = load_chunk(chunk_id)
    # Memory: 1-2GB, Load time: per-chunk, Access: O(chunk_load)
```

| Aspect | In-Memory | Memory-Mapped | Streaming |
|--------|-----------|---------------|-----------|
| RAM usage | 10GB | ~100MB | 1-2GB |
| Initial load | 5 min | Instant | Per-chunk |
| Random access | Fast | Fast | Slow (cache miss) |
| Implementation | Simple | Medium | Complex |
| Data format | CSV/Parquet | Custom binary | CSV/Parquet |

**Recommendation**:
- **10GB is acceptable** for most systems (64GB+ RAM common for ML)
- If memory constrained: Memory-mapped option (one-time conversion to binary format)
- Streaming not recommended - random access patterns in shuffled training cause thrashing

**My suggestion**: Keep in-memory for now, add memory-mapped option later if needed.

---

### 4. Price Features Status

**Question**: Are price feature files ready, or should we use placeholders?

**Current State**:
```python
# In dataset_optimized.py - currently using random placeholder
price_features = np.random.randn(batch_size, self.max_seq_len, 64).astype(np.float32) * 0.1
```

**Expected Source**: `data/features/price_features.parquet`

**T4 Price Context Components (64d)**:
| Component | Dimension | Source |
|-----------|-----------|--------|
| Fourier price encoding | 24d | actual_price |
| Log-price features | 16d | actual_price, base_price |
| Relative position | 16d | category_avg_price |
| Price velocity | 8d | prior_price vs current |

**Options**:

**Option A: Placeholder (Current)**
- Use random/zero vectors for T4
- Model learns without price signal
- Can add later without architecture change

**Option B: Generate from transactions**
```python
# SPEND column exists in transactions
# Can derive: actual_price = SPEND / QUANTITY
# Missing: base_price, category_avg, prior_price
```

**Option C: Full price features**
- Requires running feature_engineering Layer 2
- Produces proper price_features.parquet

**Question for you**:
1. Do you have `data/features/price_features.parquet`?
2. If not, should I generate basic price features from SPEND/QUANTITY?
3. Or use placeholder for now and add later?

---

### 5. Auxiliary Tasks Configuration

**Question**: Should trip prediction (T6) be mandatory or optional?

**T6 Trip Context provides 4 auxiliary prediction targets**:

| Task | Classes | Description |
|------|---------|-------------|
| mission_type | 4 | Top Up, Full Shop, Small Shop, Emergency |
| mission_focus | 5 | Fresh, Grocery, Mixed, Nonfood, General |
| price_sensitivity | 3 | LA (low), MM (medium), UM (high) |
| basket_size | 3 | S, M, L |

**Option A: Mandatory Auxiliary Tasks**
```python
# Always compute auxiliary losses
loss = mlm_loss + 0.1 * mission_type_loss + 0.1 * mission_focus_loss + ...
```

**Pros**:
- Regularization effect
- Better representations
- Multi-task learning benefits

**Cons**:
- More complexity
- Need to tune loss weights
- Slower training

**Option B: Optional (Configurable)**
```python
# Config-driven
auxiliary_tasks:
  mission_type: {weight: 0.1, enabled: true}
  mission_focus: {weight: 0.1, enabled: true}
  price_sensitivity: {weight: 0.05, enabled: false}
  basket_size: {weight: 0.05, enabled: false}
```

**Pros**:
- Flexibility to experiment
- Can disable for faster iteration
- A/B test task combinations

**Option C: Two-Stage Training**
```python
# Stage 1: MLM only (fast pretraining)
# Stage 2: MLM + auxiliary tasks (fine-tuning)
```

**Recommendation**: Option B (Configurable) with sensible defaults:
- `mission_type`: enabled, weight=0.1
- `mission_focus`: enabled, weight=0.1
- `price_sensitivity`: disabled by default
- `basket_size`: disabled by default

This matches common practice in multi-task transformers (BERT, T5).

---

### 6. Validation Frequency

**Question**: Validate every N steps or every epoch?

**Dataset Scale**:
```
Training baskets: ~24M (weeks 1-80)
Validation baskets: ~4M (weeks 81-95)
Test baskets: ~3M (weeks 96-117)

At batch_size=64:
- Training steps/epoch: ~375,000
- Validation steps: ~62,500
```

**Option A: Every Epoch**
```python
for epoch in range(num_epochs):
    train_one_epoch()
    validate()  # Full validation set
    save_checkpoint()
```

| Pros | Cons |
|------|------|
| Simple | Long wait between validations |
| Full validation | May miss early divergence |
| Standard practice | Wastes compute if model diverged |

**Option B: Every N Steps**
```python
for step, batch in enumerate(train_loader):
    train_step(batch)
    if step % validate_every == 0:
        validate(subset=True)  # Sample of validation
```

| Pros | Cons |
|------|------|
| Early divergence detection | More overhead |
| Frequent feedback | Partial validation may be noisy |
| Common in large-scale training | More checkpoints |

**Option C: Hybrid (Recommended)**
```python
validate_every_n_steps = 10000      # Quick validation on subset
full_validate_every_n_epochs = 1    # Full validation each epoch
early_stopping_patience = 3          # Stop if no improvement

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        train_step(batch)

        if step % validate_every_n_steps == 0:
            quick_val_loss = validate(max_batches=100)  # ~6400 samples
            log(f"Step {step}: val_loss={quick_val_loss}")

    # Full validation at epoch end
    full_val_loss = validate(max_batches=None)

    if not improved(full_val_loss):
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break
```

**Recommendation**: Hybrid approach with:
- Quick validation every 10,000 steps (100 batches, ~1 minute)
- Full validation every epoch
- Early stopping with patience=3 epochs
- Checkpoint best model by validation loss

---

### Summary of Recommendations

| Question | Recommendation |
|----------|----------------|
| 1. Batching | Bucket batching by sequence length |
| 2. Multi-GPU | Start single GPU, add DDP support |
| 3. Memory | In-memory (10GB acceptable) |
| 4. Price features | Need your input on availability |
| 5. Auxiliary tasks | Configurable, mission_type/focus enabled by default |
| 6. Validation | Hybrid: every 10k steps + every epoch |

---

## Implementation Status

### Completed Implementation (2024-11-28)

The unified training pipeline has been implemented in `src/training/`:

#### 1. Module Structure
```
src/training/
├── __init__.py                 # Module exports
├── prepare_samples.py          # Stage 1: Enhance temporal metadata
├── prepare_tensor_cache.py     # Stage 2: Create tensor cache
└── dataset.py                  # WorldModelDataset + DataLoaders
```

#### 2. Final Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Batching | Bucket batching | Minimize padding waste for variable-length baskets |
| Memory | In-memory (10GB) | Acceptable for ML workstations, simplest implementation |
| Price features | Loaded from `price_features.parquet` | 64d Fourier-encoded features exist |
| Auxiliary tasks | All 4 enabled by default | Better representations via multi-task learning |
| Validation | Hybrid approach | Quick every 10k steps + full every epoch |

#### 3. Data Flow

```
STAGE 1: prepare_samples.py
─────────────────────────────
data/prepared/temporal_metadata.parquet (existing, 47M baskets)
         │
         ▼
raw_data/transactions.csv (extract SHOP_WEEKDAY, SHOP_HOUR, SHOP_DATE)
         │
         ▼
data/prepared/
├── train_samples.parquet      (32M samples, weeks 1-80)
├── validation_samples.parquet (6M samples, weeks 81-95)
└── test_samples.parquet       (9M samples, weeks 96-117)

STAGE 2: prepare_tensor_cache.py
────────────────────────────────
data/features/
├── product_embeddings.pkl      (GraphSage, 4997 x 256d)
├── customer_history_embeddings.pkl (99999 x 160d)
├── customer_embeddings.parquet (99999 x 160d static)
├── store_features.parquet      (761 x 96d)
└── price_features.parquet      (104M rows x 64d)
         │
         ▼
data/tensor_cache/
├── product_embeddings.npy      (4998 x 256, with PAD row)
├── customer_history_embeddings.npy (99999 x 160)
├── customer_static_embeddings.npy  (99999 x 160)
├── store_embeddings.npy        (761 x 96)
├── price_features_indexed.parquet (with product_idx, store_idx, week_idx)
└── vocab.json                  (ID → index mappings)

STAGE 3: WorldModelDataset
──────────────────────────
Loads samples + tensor cache, encodes on-the-fly:
- T1: Customer context [B, 192] (segment 64 + history 96 + affinity 32)
- T2: Product embeddings [B, S, 256] (GraphSage + positional)
- T3: Temporal context [B, 64]
- T4: Price features [B, S, 64] (Fourier-encoded)
- T5: Store context [B, 96]
- T6: Trip context [B, 48]
```

#### 4. Key Classes

**WorldModelDataset** (`src/training/dataset.py:89`)
- Loads split samples from parquet
- Memory-maps tensor cache for efficient lookup
- Vectorized batch encoding for all T1-T6 tensors
- BERT-style masking for MLM training
- All 4 auxiliary task labels

**WorldModelDataLoader** (`src/training/dataset.py:467`)
- Bucket batching by history length
- Configurable batch size and shuffle
- Training mask application

**EvaluationDataLoader** (`src/training/dataset.py:518`)
- Groups batches by date/hour/week
- Enables fine-grained evaluation metrics
- No masking during evaluation

**TensorCache** (`src/training/prepare_tensor_cache.py:186`)
- O(1) embedding lookup by ID or index
- Lazy loading of price features
- Batch retrieval methods

#### 5. Usage

**Step 1: Data Preparation (Run Once)**

```bash
# Enhance temporal metadata with day/hour columns
python -m src.training.prepare_samples --project-root /path/to/retail_sim

# Create tensor cache from feature files
python -m src.training.prepare_tensor_cache --project-root /path/to/retail_sim
```

Or programmatically:

```python
from pathlib import Path
from src.training import enhance_temporal_metadata, prepare_tensor_cache

project_root = Path('/path/to/retail_sim')

# Stage 1: Create train/validation/test split files with day/hour columns
enhance_temporal_metadata(project_root)

# Stage 2: Convert feature files to numpy tensor cache
prepare_tensor_cache(project_root)
```

**Step 2: Create Dataset and DataLoader**

```python
from pathlib import Path
from src.training import WorldModelDataset, WorldModelDataLoader

project_root = Path('/path/to/retail_sim')

# Create training dataset
train_dataset = WorldModelDataset(
    project_root,
    split='train',           # 'train', 'validation', or 'test'
    max_seq_len=50,          # Maximum product sequence length
    mask_prob=0.15,          # BERT-style masking probability
    enable_auxiliary_tasks=True,  # Enable all 4 auxiliary prediction tasks
)

# Create dataloader with bucket batching
train_loader = WorldModelDataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    apply_masking=True,      # Apply MLM masking during training
    bucket_batching=True,    # Group similar sequence lengths
)

print(f"Dataset size: {len(train_dataset):,} samples")
print(f"DataLoader batches: {len(train_loader):,}")
```

**Step 3: Training Loop**

```python
for batch in train_loader:
    # Dense context tensors
    customer_context = batch.customer_context      # [B, 192] T1
    temporal_context = batch.temporal_context      # [B, 64]  T3
    store_context = batch.store_context            # [B, 96]  T5
    trip_context = batch.trip_context              # [B, 48]  T6

    # Sequence tensors
    product_embeddings = batch.product_embeddings  # [B, S, 256] T2
    price_features = batch.price_features          # [B, S, 64]  T4
    attention_mask = batch.attention_mask          # [B, S]

    # Concatenated features (convenience methods)
    dense_context = batch.get_dense_context()      # [B, 400]
    seq_features = batch.get_sequence_features()   # [B, S, 320]

    # MLM targets (for training)
    masked_positions = batch.masked_positions      # [B, max_masks]
    masked_targets = batch.masked_targets          # [B, max_masks]

    # Auxiliary task labels
    mission_type = batch.auxiliary_labels['mission_type']        # [B]
    mission_focus = batch.auxiliary_labels['mission_focus']      # [B]
    price_sensitivity = batch.auxiliary_labels['price_sensitivity']  # [B]
    basket_size = batch.auxiliary_labels['basket_size']          # [B]

    # Forward pass, compute loss, backprop...
```

**Step 4: Evaluation with Day/Hour Grouping**

```python
from src.training import WorldModelDataset, EvaluationDataLoader

# Create validation dataset
val_dataset = WorldModelDataset(
    project_root,
    split='validation',
    load_transactions=True,
)

# Group evaluation by date for fine-grained metrics
eval_loader = EvaluationDataLoader(
    val_dataset,
    batch_size=64,
    group_by='date',  # 'date', 'hour', or 'week'
)

# Evaluate per date
date_metrics = {}
for date_key, batch in eval_loader:
    # batch has no masking applied
    # Compute predictions and metrics
    predictions = model(batch)
    date_metrics[date_key] = compute_metrics(predictions, batch)

# Aggregate metrics by date
print(f"Evaluated {len(date_metrics)} dates")
```

**Step 5: Using TensorCache Directly (Optional)**

```python
from src.training.prepare_tensor_cache import TensorCache
from pathlib import Path

# Load pre-computed embeddings
cache = TensorCache(Path('/path/to/retail_sim/data/tensor_cache'))

# Single lookups
product_emb = cache.get_product_embedding('PRD0900001')  # [256]
cust_hist, cust_static = cache.get_customer_embedding('CUST0000000001')  # [160], [160]
store_emb = cache.get_store_embedding('STORE00001')  # [96]

# Batch lookups (efficient)
product_indices = np.array([1, 2, 3, 4, 5])
product_embs = cache.get_product_embeddings_batch(product_indices)  # [5, 256]

# ID to index conversion
prod_idx = cache.product_id_to_idx('PRD0900001')  # Returns integer index
cust_idx = cache.customer_id_to_idx('CUST0000000001')
store_idx = cache.store_id_to_idx('STORE00001')
```

#### 6. Tests

Integration tests in `tests/test_training/test_dataset.py`:
- Split integrity (no temporal leakage)
- Tensor shape verification
- Masking correctness
- Bucket batching efficiency
- Evaluation grouping

Run tests:
```bash
pytest tests/test_training/ -v
```

### Remaining Work

1. **Run tensor cache preparation**: Execute `prepare_tensor_cache.py` to create cached embeddings
2. **Run sample enhancement**: Execute `prepare_samples.py` to add day/hour columns
3. **Training script**: Create `train_world_model.py` with training loop, validation, checkpointing
4. **Evaluation utilities**: Add metric computation for day/hour level evaluation
