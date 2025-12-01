"""
Tensor Preparation Module
=========================
Section 4 of RetailSim: Tensor Preparation Specification

Tensors:
- T1: Customer Context [192d]
- T2: Product Sequence [256d/item]
- T3: Temporal Context [64d]
- T4: Price Context [64d/item]
- T5: Store Context [96d]
- T6: Trip Context [48d]

Two implementations available:
1. Original (dataset.py) - Dict-based lookups, Python loops
2. Optimized (dataset_optimized.py) - Vectorized, category dtypes

Memory comparison (full 307M row dataset):
- Original: ~350 GB
- Optimized: ~12-15 GB (96% reduction)

Speed comparison (per batch):
- Original: ~100ms
- Optimized: ~1ms (100x faster)

Quick Start (Optimized - Recommended):
    from src.tensor_preparation import (
        RetailSimDatasetOptimized,
        RetailSimDataLoaderOptimized,
        TRANSACTION_DTYPES
    )

    dataset = RetailSimDatasetOptimized(project_root, use_sampled=True)
    dataloader = RetailSimDataLoaderOptimized(dataset, batch_size=32)

    for batch in dataloader:
        dense = batch.get_dense_context()      # [B, 400]
        seq = batch.get_sequence_features()    # [B, S, 320]

See docs/section4_tensor_preparation.md for full documentation.
"""

from .t1_customer_context import CustomerContextEncoder
from .t2_product_sequence import ProductSequenceEncoder, ProductSequenceBatch
from .t3_temporal_context import TemporalContextEncoder
from .t4_price_context import PriceContextEncoder, PriceContextBatch
from .t5_store_context import StoreContextEncoder
from .t6_trip_context import TripContextEncoder

# Original implementation (for backwards compatibility)
from .dataset import RetailSimDataset, RetailSimDataLoader, RetailSimBatch

# Optimized implementation (recommended)
from .dataset_optimized import (
    RetailSimDatasetOptimized,
    RetailSimDataLoaderOptimized,
    VectorizedTensorEncoder,
    TRANSACTION_DTYPES
)

__all__ = [
    # Encoders
    'CustomerContextEncoder',
    'ProductSequenceEncoder',
    'ProductSequenceBatch',
    'TemporalContextEncoder',
    'PriceContextEncoder',
    'PriceContextBatch',
    'StoreContextEncoder',
    'TripContextEncoder',
    # Original dataset (backwards compatible)
    'RetailSimDataset',
    'RetailSimDataLoader',
    'RetailSimBatch',
    # Optimized dataset (recommended)
    'RetailSimDatasetOptimized',
    'RetailSimDataLoaderOptimized',
    'VectorizedTensorEncoder',
    'TRANSACTION_DTYPES',
]
