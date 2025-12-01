"""
Section 4.7: Data Preparation for Train/Val/Test
=================================================
Prepares temporally-correct datasets from processed features
to training-ready samples with proper split boundaries.

Pipeline Stages:
1. stage1_temporal_metadata.py - Split assignments and flags
2. stage2_customer_histories.py - Per-split history extraction
3. stage3_training_samples.py - Bucketed by history length
4. stage4_tensor_cache.py - Static embedding caching

Usage:
    python -m src.data_preparation.run_data_preparation

    # Or run individual stages:
    python -m src.data_preparation.stage1_temporal_metadata
    python -m src.data_preparation.stage4_tensor_cache

Temporal Split Boundaries:
    - Training: Weeks 1-80 (72%)
    - Validation: Weeks 81-95 (13%)
    - Test: Weeks 96-117 (15%)

Note: SHOP_WEEK in raw data is YYYYWW format (e.g., 200607-200819).
      Stage 1 normalizes this to sequential weeks 1-117.

Output Directory Structure:
    data/prepared/
    ├── temporal_metadata.parquet      # Basket-level split assignments
    ├── customer_histories.parquet     # Per-basket history sequences
    ├── samples/
    │   ├── train_bucket_1_samples.parquet
    │   ├── train_bucket_2_samples.parquet
    │   ├── train_bucket_3_samples.parquet
    │   ├── train_bucket_4_samples.parquet
    │   ├── train_bucket_5_samples.parquet
    │   ├── validation_samples.parquet
    │   └── test_samples.parquet
    └── tensor_cache/
        ├── vocab.json                 # ID -> index mappings for all entities
        ├── product_embeddings.pt      # (N_products, 128) tensor
        ├── segment_embeddings.pt      # (N_segments, 128) for cold-start
        ├── store_embeddings.pt        # (N_stores, 128) tensor
        └── category_embeddings.pt     # (N_categories, 128) tensor
"""

from .stage1_temporal_metadata import TemporalMetadataCreator
from .stage2_customer_histories import CustomerHistoryExtractor
from .stage3_training_samples import TrainingSampleCreator
from .stage4_tensor_cache import TensorCacheBuilder
from .dataloader import (
    RetailSimDataset,
    BucketBatchSampler,
    create_dataloader,
    create_dataloaders
)

__all__ = [
    # Pipeline stages
    'TemporalMetadataCreator',
    'CustomerHistoryExtractor',
    'TrainingSampleCreator',
    'TensorCacheBuilder',
    # DataLoader components
    'RetailSimDataset',
    'BucketBatchSampler',
    'create_dataloader',
    'create_dataloaders',
]
