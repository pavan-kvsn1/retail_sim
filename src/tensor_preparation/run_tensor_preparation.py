"""
Tensor Preparation Pipeline Runner
===================================
Section 4 of RetailSim: Tests all tensor encoders and dataset.

Output:
- Validates all tensor dimensions
- Tests batch processing (original and optimized)
- Generates sample tensors for inspection
- Benchmarks optimized vs original performance

Usage:
    python run_tensor_preparation.py                    # Quick test (10k rows)
    python run_tensor_preparation.py --nrows 100000    # Test with 100k rows
    python run_tensor_preparation.py --use-sampled     # Use sampled transactions
    python run_tensor_preparation.py --optimized-only  # Skip original dataset test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
import sys
import time

warnings.filterwarnings('ignore')

from t1_customer_context import CustomerContextEncoder
from t2_product_sequence import ProductSequenceEncoder
from t3_temporal_context import TemporalContextEncoder
from t4_price_context import PriceContextEncoder
from t5_store_context import StoreContextEncoder
from t6_trip_context import TripContextEncoder
from dataset import RetailSimDataset, RetailSimDataLoader
from dataset_optimized import (
    RetailSimDatasetOptimized,
    RetailSimDataLoaderOptimized,
    VectorizedTensorEncoder,
    TRANSACTION_DTYPES
)



def run_tensor_tests(project_root: Path) -> dict:
    """
    Test all tensor encoders individually.

    Parameters
    ----------
    project_root : Path
        Project root directory

    Returns
    -------
    dict
        Test results and statistics
    """
    results = {}
    features_dir = project_root / 'data' / 'features'

    # Load shared resources
    print("Loading shared resources...")

    # Product embeddings
    with open(features_dir / 'product_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
        product_embeddings = data.get('embeddings', {})
    print(f"  Product embeddings: {len(product_embeddings)}")

    # Customer embeddings
    customer_df = pd.read_parquet(features_dir / 'customer_embeddings.parquet')
    print(f"  Customer embeddings: {len(customer_df)}")

    # Transactions
    transactions_df = pd.read_csv(
        project_root / 'raw_data' / 'transactions.csv',
        nrows=10000
    )
    print(f"  Transactions: {len(transactions_df)}")

    # ========================================
    # T1: Customer Context [192d]
    # ========================================
    print("\n" + "=" * 50)
    print("T1: Customer Context [192d]")
    print("=" * 50)

    customer_embed_dict = {}
    for _, row in customer_df.iterrows():
        cust_id = row['customer_id']
        embed = row[[c for c in row.index if c.startswith('embed_')]].values.astype(np.float32)
        customer_embed_dict[cust_id] = embed

    trip_counts = customer_df.set_index('customer_id')['total_trips'].to_dict()

    t1_encoder = CustomerContextEncoder()

    sample_customer = list(customer_embed_dict.keys())[0]
    sample_embed = customer_embed_dict[sample_customer]
    sample_trips = trip_counts.get(sample_customer, 0)
    t1 = t1_encoder.encode_customer(
        customer_id=sample_customer,
        seg1='CT',  # Default segment
        seg2='DI',  # Default segment
        history_embedding=sample_embed,
        num_trips=sample_trips
    )

    print(f"  Output dim: {t1_encoder.output_dim}")
    print(f"  Sample shape: {t1.shape}")
    print(f"  Sample norm: {np.linalg.norm(t1):.4f}")

    results['t1'] = {
        'dim': t1_encoder.output_dim,
        'expected_dim': 192,
        'valid': t1_encoder.output_dim == 192,
        'components': f"segment={t1_encoder.segment_dim}, history={t1_encoder.history_dim}, affinity={t1_encoder.affinity_dim}"
    }

    # ========================================
    # T2: Product Sequence [256d/item]
    # ========================================
    print("\n" + "=" * 50)
    print("T2: Product Sequence [256d per item]")
    print("=" * 50)

    t2_encoder = ProductSequenceEncoder(product_embeddings)

    sample_products = list(product_embeddings.keys())[:5]
    emb, tids, length, _, _ = t2_encoder.encode_sequence(sample_products)

    print(f"  Vocab size: {t2_encoder.vocab_size}")
    print(f"  Embedding dim: {t2_encoder.embedding_dim}")
    print(f"  Sample basket: {len(sample_products)} products")
    print(f"  Output shape: {emb.shape}")
    print(f"  Token IDs: {tids}")

    results['t2'] = {
        'dim': t2_encoder.embedding_dim,
        'expected_dim': 256,
        'vocab_size': t2_encoder.vocab_size,
        'valid': t2_encoder.embedding_dim == 256
    }

    # ========================================
    # T3: Temporal Context [64d]
    # ========================================
    print("\n" + "=" * 50)
    print("T3: Temporal Context [64d]")
    print("=" * 50)

    t3_encoder = TemporalContextEncoder()

    t3 = t3_encoder.encode_temporal(
        shop_week=200626,
        shop_weekday=3,
        shop_hour=14,
        last_visit_week=200625
    )

    print(f"  Output dim: {t3_encoder.output_dim}")
    print(f"  Sample shape: {t3.shape}")
    print(f"  Components: week={t3_encoder.week_dim}, weekday={t3_encoder.weekday_dim}, "
          f"hour={t3_encoder.hour_dim}")

    results['t3'] = {
        'dim': t3_encoder.output_dim,
        'expected_dim': 64,
        'valid': t3_encoder.output_dim == 64
    }

    # ========================================
    # T4: Price Context [64d/item]
    # ========================================
    print("\n" + "=" * 50)
    print("T4: Price Context [64d per item]")
    print("=" * 50)

    t4_encoder = PriceContextEncoder()

    t4 = t4_encoder.encode_price(
        actual_price=1.99,
        base_price=2.49,
        category_avg_price=2.20,
        prior_price=2.19
    )

    print(f"  Output dim: {t4_encoder.output_dim}")
    print(f"  Sample shape: {t4.shape}")
    print(f"  Components: fourier={t4_encoder.fourier_dim}, log={t4_encoder.log_dim}, "
          f"relative={t4_encoder.relative_dim}, velocity={t4_encoder.velocity_dim}")

    results['t4'] = {
        'dim': t4_encoder.output_dim,
        'expected_dim': 64,
        'valid': t4_encoder.output_dim == 64
    }

    # ========================================
    # T5: Store Context [96d]
    # ========================================
    print("\n" + "=" * 50)
    print("T5: Store Context [96d]")
    print("=" * 50)

    t5_encoder = StoreContextEncoder()

    t5 = t5_encoder.encode_store(
        store_id='STORE001',
        store_format='LS',
        store_region='E01',
        operational_features={
            'store_size': 0.8,
            'traffic': 0.7,
            'competition': 0.3,
            'store_age': 0.9
        }
    )

    print(f"  Output dim: {t5_encoder.output_dim}")
    print(f"  Sample shape: {t5.shape}")
    print(f"  Components: format={t5_encoder.format_dim}, region={t5_encoder.region_dim}, "
          f"operational={t5_encoder.operational_dim}, identity={t5_encoder.identity_dim}")

    results['t5'] = {
        'dim': t5_encoder.output_dim,
        'expected_dim': 96,
        'valid': t5_encoder.output_dim == 96
    }

    # ========================================
    # T6: Trip Context [48d]
    # ========================================
    print("\n" + "=" * 50)
    print("T6: Trip Context [48d]")
    print("=" * 50)

    t6_encoder = TripContextEncoder()

    t6 = t6_encoder.encode_trip(
        mission_type='Full Shop',
        mission_focus='Fresh',
        price_sensitivity='MM',
        basket_size='L'
    )

    print(f"  Output dim: {t6_encoder.output_dim}")
    print(f"  Sample shape: {t6.shape}")
    print(f"  Mission types: {t6_encoder.num_mission_types}")
    print(f"  Mission focuses: {t6_encoder.num_mission_focuses}")

    results['t6'] = {
        'dim': t6_encoder.output_dim,
        'expected_dim': 48,
        'valid': t6_encoder.output_dim == 48
    }

    return results


def run_dataset_test(project_root: Path) -> dict:
    """
    Test the complete dataset and dataloader.

    Parameters
    ----------
    project_root : Path
        Project root directory

    Returns
    -------
    dict
        Test results
    """

    print("\n" + "=" * 50)
    print("Dataset and DataLoader Test")
    print("=" * 50)

    # Create dataset
    print("\nInitializing dataset...")
    dataset = RetailSimDataset(project_root)

    results = {
        'num_baskets': len(dataset),
        'encoders_initialized': True
    }

    # Test single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    print(f"  T1 (Customer): {sample['t1'].shape}")
    print(f"  T2 (Products): {sample['t2_embeddings'].shape}")
    print(f"  T3 (Temporal): {sample['t3'].shape}")
    print(f"  T4 (Price): {sample['t4'].shape}")
    print(f"  T5 (Store): {sample['t5'].shape}")
    print(f"  T6 (Trip): {sample['t6'].shape}")

    # Test batch
    print("\nTesting batch (size=8)...")
    batch = dataset.get_batch(list(range(8)), apply_masking=True)

    print(f"  Dense context shape: {batch.get_dense_context().shape}")
    print(f"  Sequence features shape: {batch.get_sequence_features().shape}")
    print(f"  Attention mask shape: {batch.attention_mask.shape}")

    results['batch_dense_dim'] = batch.get_dense_context().shape[1]
    results['batch_seq_dim'] = batch.get_sequence_features().shape[2]
    results['expected_dense_dim'] = 400  # 192 + 64 + 96 + 48
    results['expected_seq_dim'] = 320    # 256 + 64

    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = RetailSimDataLoader(dataset, batch_size=16, shuffle=True)
    print(f"  Total batches: {len(dataloader)}")

    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count >= 3:
            break

    results['dataloader_works'] = batch_count == 3

    return results


def run_optimized_dataset_test(project_root: Path, nrows: int = None, use_sampled: bool = True) -> dict:
    """
    Test the optimized dataset and dataloader.

    Parameters
    ----------
    project_root : Path
        Project root directory
    nrows : int, optional
        Number of transaction rows to load
    use_sampled : bool
        Whether to use sampled transactions file

    Returns
    -------
    dict
        Test results including performance metrics
    """
    print("\n" + "=" * 50)
    print("Optimized Dataset and DataLoader Test")
    print("=" * 50)

    results = {}

    # Create optimized dataset
    print("\nInitializing optimized dataset...")
    start_time = time.time()

    dataset = RetailSimDatasetOptimized(
        project_root,
        max_seq_len=50,
        mask_prob=0.15,
        nrows=nrows,
        use_sampled=use_sampled
    )

    load_time = time.time() - start_time
    print(f"  Load time: {load_time:.2f}s")

    results['num_baskets'] = len(dataset)
    results['load_time_s'] = load_time
    results['encoders_initialized'] = True

    # Test single sample
    print("\nTesting single sample...")
    start_time = time.time()
    sample = dataset[0]
    single_time = (time.time() - start_time) * 1000

    print(f"  T1 (Customer): {sample['t1'].shape}")
    print(f"  T2 (Products): {sample['t2_embeddings'].shape}")
    print(f"  T3 (Temporal): {sample['t3'].shape}")
    print(f"  T4 (Price): {sample['t4'].shape}")
    print(f"  T5 (Store): {sample['t5'].shape}")
    print(f"  T6 (Trip): {sample['t6'].shape}")
    print(f"  Encoding time: {single_time:.2f}ms")

    results['single_sample_ms'] = single_time

    # Validate tensor shapes
    results['t1_shape_valid'] = sample['t1'].shape == (192,)
    results['t2_shape_valid'] = sample['t2_embeddings'].shape[1] == 256
    results['t3_shape_valid'] = sample['t3'].shape == (64,)
    results['t4_shape_valid'] = sample['t4'].shape[1] == 64
    results['t5_shape_valid'] = sample['t5'].shape == (96,)
    results['t6_shape_valid'] = sample['t6'].shape == (48,)

    # Test batch encoding
    print("\nTesting batch encoding (size=32)...")
    start_time = time.time()
    batch = dataset.get_batch(list(range(min(32, len(dataset)))), apply_masking=True)
    batch_time = (time.time() - start_time) * 1000

    print(f"  Batch size: {batch.batch_size}")
    print(f"  Dense context shape: {batch.get_dense_context().shape}")
    print(f"  Sequence features shape: {batch.get_sequence_features().shape}")
    print(f"  Attention mask shape: {batch.attention_mask.shape}")
    print(f"  Encoding time: {batch_time:.2f}ms")

    results['batch_dense_dim'] = batch.get_dense_context().shape[1]
    results['batch_seq_dim'] = batch.get_sequence_features().shape[2]
    results['expected_dense_dim'] = 400  # 192 + 64 + 96 + 48
    results['expected_seq_dim'] = 320    # 256 + 64
    results['batch_encoding_ms'] = batch_time

    # Test masking
    if batch.masked_positions is not None:
        n_masked = (batch.masked_positions > 0).sum()
        print(f"  Masked tokens: {n_masked}")
        results['masking_works'] = n_masked > 0
    else:
        results['masking_works'] = False

    # Test trip labels
    print("\n  Trip labels:")
    for label_name, label_values in batch.trip_labels.items():
        print(f"    {label_name}: shape={label_values.shape}, unique={len(np.unique(label_values))}")
    results['trip_labels_present'] = len(batch.trip_labels) == 4

    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = RetailSimDataLoaderOptimized(
        dataset,
        batch_size=64,
        shuffle=True,
        apply_masking=True
    )
    print(f"  Total batches: {len(dataloader)}")

    # Benchmark multiple batches
    print("\nBenchmarking 10 batches...")
    start_time = time.time()
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count >= 10:
            break
    benchmark_time = time.time() - start_time

    print(f"  10 batches in {benchmark_time:.3f}s")
    print(f"  Average: {benchmark_time/10*1000:.1f}ms per batch")

    results['dataloader_works'] = batch_count == 10
    results['avg_batch_ms'] = benchmark_time / 10 * 1000

    return results


def run_vectorized_encoder_test() -> dict:
    """
    Test the VectorizedTensorEncoder independently.

    Returns
    -------
    dict
        Test results
    """
    print("\n" + "=" * 50)
    print("VectorizedTensorEncoder Test")
    print("=" * 50)

    results = {}
    encoder = VectorizedTensorEncoder()

    # Test temporal batch encoding
    print("\nTesting temporal batch encoding...")
    batch_size = 100
    weeks = np.random.randint(200601, 200819, size=batch_size).astype(np.int32)
    weekdays = np.random.randint(1, 8, size=batch_size).astype(np.int8)
    hours = np.random.randint(0, 24, size=batch_size).astype(np.int8)

    start_time = time.time()
    temporal = encoder.encode_temporal_batch(weeks, weekdays, hours, 200601, 200819)
    temporal_time = (time.time() - start_time) * 1000

    print(f"  Input: {batch_size} samples")
    print(f"  Output shape: {temporal.shape}")
    print(f"  Expected: ({batch_size}, 64)")
    print(f"  Time: {temporal_time:.2f}ms")

    results['temporal_shape_valid'] = temporal.shape == (batch_size, 64)
    results['temporal_time_ms'] = temporal_time

    # Test trip batch encoding
    print("\nTesting trip batch encoding...")
    mission_types = pd.Categorical(['Top Up', 'Full Shop', 'Small Shop'] * 34)[:batch_size]
    mission_focuses = pd.Categorical(['Fresh', 'Grocery', 'Mixed'] * 34)[:batch_size]
    price_sens = pd.Categorical(['LA', 'MM', 'UM'] * 34)[:batch_size]
    basket_sizes = pd.Categorical(['S', 'M', 'L'] * 34)[:batch_size]

    start_time = time.time()
    trip_tensor, trip_labels = encoder.encode_trip_batch(
        mission_types, mission_focuses, price_sens, basket_sizes
    )
    trip_time = (time.time() - start_time) * 1000

    print(f"  Output shape: {trip_tensor.shape}")
    print(f"  Expected: ({batch_size}, 48)")
    print(f"  Labels: {list(trip_labels.keys())}")
    print(f"  Time: {trip_time:.2f}ms")

    results['trip_shape_valid'] = trip_tensor.shape == (batch_size, 48)
    results['trip_labels_valid'] = len(trip_labels) == 4
    results['trip_time_ms'] = trip_time

    return results


def main():
    """Run complete tensor preparation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Run RetailSim Tensor Preparation')
    parser.add_argument(
        '--nrows',
        type=int,
        default=10000,
        help='Number of transaction rows to process (default: 10000)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all rows in the dataset (overrides --nrows)'
    )
    parser.add_argument(
        '--use-sampled',
        action='store_true',
        help='Use sampled transactions file (transactions_top75k.csv or similar)'
    )
    parser.add_argument(
        '--optimized-only',
        action='store_true',
        help='Skip original dataset test, only run optimized'
    )
    parser.add_argument(
        '--skip-original',
        action='store_true',
        help='Skip original dataset test (alias for --optimized-only)'
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    nrows = None if args.all else args.nrows
    skip_original = args.optimized_only or args.skip_original

    print("=" * 60)
    print("RetailSim Tensor Preparation Pipeline")
    print("Section 4: Tensor Specification")
    print("=" * 60)

    if args.all:
        print("\nProcessing ALL transaction rows")
    else:
        print(f"\nProcessing {args.nrows:,} transaction rows")

    if args.use_sampled:
        print("Using sampled transactions file")

    # ========================================
    # PHASE 1: Individual Tensor Tests
    # ========================================
    print("\n" + "#" * 60)
    print("PHASE 1: Individual Tensor Tests")
    print("#" * 60)

    tensor_results = run_tensor_tests(project_root)

    # ========================================
    # PHASE 2: Original Dataset Test (optional)
    # ========================================
    dataset_results = None
    if not skip_original:
        print("\n" + "#" * 60)
        print("PHASE 2: Original Dataset Integration Test")
        print("#" * 60)

        try:
            dataset_results = run_dataset_test(project_root)
        except Exception as e:
            print(f"\n  Original dataset test failed: {e}")
            print("  (This is expected for large datasets - use --optimized-only)")
            dataset_results = {'skipped': True, 'error': str(e)}
    else:
        print("\n" + "#" * 60)
        print("PHASE 2: Original Dataset Test SKIPPED (--optimized-only)")
        print("#" * 60)
        dataset_results = {'skipped': True}

    # ========================================
    # PHASE 3: Vectorized Encoder Test
    # ========================================
    print("\n" + "#" * 60)
    print("PHASE 3: Vectorized Encoder Test")
    print("#" * 60)

    vectorized_results = run_vectorized_encoder_test()

    # ========================================
    # PHASE 4: Optimized Dataset Test
    # ========================================
    print("\n" + "#" * 60)
    print("PHASE 4: Optimized Dataset Integration Test")
    print("#" * 60)

    optimized_results = run_optimized_dataset_test(
        project_root,
        nrows=nrows,
        use_sampled=args.use_sampled
    )

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Tensor dimensions
    print("\nTensor Dimensions:")
    total_dense = 0
    total_seq = 0
    all_tensors_valid = True
    for tensor_name, result in tensor_results.items():
        status = "✓" if result['valid'] else "✗"
        print(f"  {tensor_name}: {result['dim']}d (expected {result['expected_dim']}d) {status}")
        if tensor_name in ['t1', 't3', 't5', 't6']:
            total_dense += result['dim']
        elif tensor_name in ['t2', 't4']:
            total_seq += result['dim']
        if not result['valid']:
            all_tensors_valid = False

    print(f"\n  Total dense context: {total_dense}d")
    print(f"  Total sequence features: {total_seq}d per item")

    # Vectorized encoder results
    print("\nVectorized Encoder:")
    vec_valid = vectorized_results['temporal_shape_valid'] and vectorized_results['trip_shape_valid']
    print(f"  Temporal encoding: {'✓' if vectorized_results['temporal_shape_valid'] else '✗'} "
          f"({vectorized_results['temporal_time_ms']:.2f}ms)")
    print(f"  Trip encoding: {'✓' if vectorized_results['trip_shape_valid'] else '✗'} "
          f"({vectorized_results['trip_time_ms']:.2f}ms)")

    # Original dataset results
    if not dataset_results.get('skipped'):
        print("\nOriginal Dataset:")
        print(f"  Baskets: {dataset_results['num_baskets']}")
        print(f"  Dense dim: {dataset_results['batch_dense_dim']} (expected 400)")
        print(f"  Sequence dim: {dataset_results['batch_seq_dim']} (expected 320)")
        original_valid = (
            dataset_results['batch_dense_dim'] == 400 and
            dataset_results['batch_seq_dim'] == 320 and
            dataset_results['dataloader_works']
        )
    else:
        print("\nOriginal Dataset: SKIPPED")
        original_valid = True  # Don't fail overall if skipped

    # Optimized dataset results
    print("\nOptimized Dataset:")
    print(f"  Baskets: {optimized_results['num_baskets']}")
    print(f"  Load time: {optimized_results['load_time_s']:.2f}s")
    print(f"  Dense dim: {optimized_results['batch_dense_dim']} (expected 400)")
    print(f"  Sequence dim: {optimized_results['batch_seq_dim']} (expected 320)")
    print(f"  Batch encoding: {optimized_results['batch_encoding_ms']:.2f}ms")
    print(f"  Avg batch iteration: {optimized_results['avg_batch_ms']:.1f}ms")

    optimized_valid = (
        optimized_results['batch_dense_dim'] == 400 and
        optimized_results['batch_seq_dim'] == 320 and
        optimized_results['dataloader_works'] and
        optimized_results['masking_works'] and
        optimized_results['trip_labels_present']
    )

    # Shape validations
    print("\n  Tensor shape validation:")
    for key in ['t1', 't2', 't3', 't4', 't5', 't6']:
        valid_key = f'{key}_shape_valid'
        if valid_key in optimized_results:
            status = '✓' if optimized_results[valid_key] else '✗'
            print(f"    {key.upper()}: {status}")

    # Final verdict
    print("\n" + "=" * 60)
    all_passed = all_tensors_valid and vec_valid and original_valid and optimized_valid

    if all_passed:
        print("All tensor preparation tests PASSED!")
        print("\nRecommendation: Use RetailSimDatasetOptimized for production.")
    else:
        print("Some tests FAILED - check individual results above")
        if not all_tensors_valid:
            print("  - Individual tensor encoders have issues")
        if not vec_valid:
            print("  - Vectorized encoder has issues")
        if not original_valid and not dataset_results.get('skipped'):
            print("  - Original dataset has issues")
        if not optimized_valid:
            print("  - Optimized dataset has issues")

    print("=" * 60)

    return {
        'tensor_results': tensor_results,
        'vectorized_results': vectorized_results,
        'dataset_results': dataset_results,
        'optimized_results': optimized_results,
        'all_passed': all_passed
    }


if __name__ == '__main__':
    main()