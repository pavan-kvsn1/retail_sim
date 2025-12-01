"""
Evaluation Script for Section 4: Tensor Preparation
====================================================
Evaluates quality of tensor encoders and dataset construction.

Metrics:
- Tensor dimension correctness
- NaN/Inf detection
- Batch consistency
- Coverage statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_tensor_dimensions() -> Dict[str, Any]:
    """
    Evaluate that all tensor dimensions match specification.
    """
    from src.tensor_preparation import (
        CustomerContextEncoder,
        TemporalContextEncoder,
        StoreContextEncoder,
        TripContextEncoder,
        PriceContextEncoder,
    )

    metrics = {}

    # T1: Customer Context [192d]
    t1_encoder = CustomerContextEncoder()
    metrics['t1_expected'] = 192
    metrics['t1_actual'] = t1_encoder.output_dim
    metrics['t1_correct'] = t1_encoder.output_dim == 192

    # T3: Temporal Context [64d]
    t3_encoder = TemporalContextEncoder()
    metrics['t3_expected'] = 64
    metrics['t3_actual'] = t3_encoder.output_dim
    metrics['t3_correct'] = t3_encoder.output_dim == 64

    # T4: Price Context [64d/item]
    t4_encoder = PriceContextEncoder()
    metrics['t4_expected'] = 64
    metrics['t4_actual'] = t4_encoder.output_dim
    metrics['t4_correct'] = t4_encoder.output_dim == 64

    # T5: Store Context [96d]
    t5_encoder = StoreContextEncoder()
    metrics['t5_expected'] = 96
    metrics['t5_actual'] = t5_encoder.output_dim
    metrics['t5_correct'] = t5_encoder.output_dim == 96

    # T6: Trip Context [48d]
    t6_encoder = TripContextEncoder()
    metrics['t6_expected'] = 48
    metrics['t6_actual'] = t6_encoder.output_dim
    metrics['t6_correct'] = t6_encoder.output_dim == 48

    # Total dimensions
    metrics['dense_context_expected'] = 400  # T1 + T3 + T5 + T6
    metrics['dense_context_actual'] = (
        t1_encoder.output_dim + t3_encoder.output_dim +
        t5_encoder.output_dim + t6_encoder.output_dim
    )
    metrics['dense_context_correct'] = metrics['dense_context_actual'] == 400

    # Quality score
    correct_count = sum([
        metrics['t1_correct'],
        metrics['t3_correct'],
        metrics['t4_correct'],
        metrics['t5_correct'],
        metrics['t6_correct'],
        metrics['dense_context_correct'],
    ])
    metrics['quality_score'] = (correct_count / 6) * 100

    return metrics


def evaluate_t1_encoder(sample_size: int = 100) -> Dict[str, Any]:
    """
    Evaluate T1 Customer Context encoder quality.
    """
    from src.tensor_preparation import CustomerContextEncoder

    metrics = {}
    encoder = CustomerContextEncoder()

    # Generate sample encodings
    seg1_codes = ['CT', 'YA', 'FM', 'MA', 'UC']
    seg2_codes = ['DI', 'VH', 'MV', 'OL', 'UN']

    encodings = []
    for i in range(sample_size):
        t1 = encoder.encode_customer(
            customer_id=f'C{i}',
            seg1=np.random.choice(seg1_codes),
            seg2=np.random.choice(seg2_codes),
            history_embedding=np.random.randn(160) if i > 10 else None,
            num_trips=np.random.randint(0, 50)
        )
        encodings.append(t1)

    all_encodings = np.array(encodings)

    # Check for NaN/Inf
    metrics['nan_count'] = int(np.isnan(all_encodings).sum())
    metrics['inf_count'] = int(np.isinf(all_encodings).sum())

    # Distribution statistics
    metrics['mean'] = float(all_encodings.mean())
    metrics['std'] = float(all_encodings.std())
    metrics['min'] = float(all_encodings.min())
    metrics['max'] = float(all_encodings.max())

    # Variance per dimension
    dim_vars = all_encodings.var(axis=0)
    metrics['zero_variance_dims'] = int((dim_vars < 1e-10).sum())
    metrics['avg_dim_variance'] = float(dim_vars.mean())

    # Quality score
    quality_score = 100
    if metrics['nan_count'] > 0:
        quality_score -= 40
    if metrics['inf_count'] > 0:
        quality_score -= 40
    if metrics['zero_variance_dims'] > encoder.output_dim * 0.1:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_t2_encoder(embeddings_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Evaluate T2 Product Sequence encoder quality.
    """
    from src.tensor_preparation import ProductSequenceEncoder

    metrics = {}

    # Load or create embeddings
    if embeddings_path and embeddings_path.exists():
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            embeddings = data.get('embeddings', data)
    else:
        # Create mock embeddings
        embeddings = {f'P{i}': np.random.randn(256) for i in range(100)}

    encoder = ProductSequenceEncoder(embeddings)

    metrics['num_products'] = len(embeddings)
    metrics['embedding_dim'] = encoder.embedding_dim
    metrics['vocab_size'] = encoder.vocab_size

    # Test sequence encoding
    products = list(embeddings.keys())[:20]
    emb, tids, length, mask_pos, mask_tgt = encoder.encode_sequence(
        products, add_eos=True, apply_masking=True
    )

    metrics['sequence_length'] = int(length)
    metrics['mask_positions'] = len(mask_pos) if mask_pos is not None else 0

    # Check special tokens
    metrics['has_eos'] = bool(tids[-1] == encoder.EOS_TOKEN_ID)
    metrics['pad_token_id'] = encoder.PAD_TOKEN_ID
    metrics['mask_token_id'] = encoder.MASK_TOKEN_ID
    metrics['eos_token_id'] = encoder.EOS_TOKEN_ID

    # Test batch encoding
    baskets = [products[:5], products[5:15], products[15:18]]
    batch = encoder.encode_batch(baskets, apply_masking=False)

    metrics['batch_shape'] = list(batch.embeddings.shape)
    metrics['attention_mask_valid'] = bool(batch.attention_mask.sum() > 0)

    # Quality score
    quality_score = 100
    if not metrics['has_eos']:
        quality_score -= 20
    if encoder.embedding_dim != 256:
        quality_score -= 30
    if not metrics['attention_mask_valid']:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_t3_encoder(sample_size: int = 100) -> Dict[str, Any]:
    """
    Evaluate T3 Temporal Context encoder quality.
    """
    from src.tensor_preparation import TemporalContextEncoder

    metrics = {}
    encoder = TemporalContextEncoder()

    # Generate sample encodings
    encodings = []
    for i in range(sample_size):
        week = 200601 + (i % 52)
        t3 = encoder.encode_temporal(
            shop_week=week,
            shop_weekday=np.random.randint(0, 7),
            shop_hour=np.random.randint(6, 22),
            last_visit_week=week - np.random.randint(1, 10) if i > 20 else None
        )
        encodings.append(t3)

    all_encodings = np.array(encodings)

    # Check for NaN/Inf
    metrics['nan_count'] = int(np.isnan(all_encodings).sum())
    metrics['inf_count'] = int(np.isinf(all_encodings).sum())

    # Distribution statistics
    metrics['mean'] = float(all_encodings.mean())
    metrics['std'] = float(all_encodings.std())

    # Variance per dimension
    dim_vars = all_encodings.var(axis=0)
    metrics['zero_variance_dims'] = int((dim_vars < 1e-10).sum())

    # Quality score
    quality_score = 100
    if metrics['nan_count'] > 0:
        quality_score -= 40
    if metrics['inf_count'] > 0:
        quality_score -= 40
    if metrics['zero_variance_dims'] > encoder.output_dim * 0.1:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_t4_encoder(sample_size: int = 100) -> Dict[str, Any]:
    """
    Evaluate T4 Price Context encoder quality.
    """
    from src.tensor_preparation import PriceContextEncoder

    metrics = {}
    encoder = PriceContextEncoder()

    # Generate sample encodings
    encodings = []
    for i in range(sample_size):
        base_price = np.random.uniform(0.5, 20.0)
        discount = np.random.uniform(0, 0.3)
        t4 = encoder.encode_price(
            actual_price=base_price * (1 - discount),
            base_price=base_price,
            category_avg_price=base_price * np.random.uniform(0.8, 1.2),
            prior_price=base_price * np.random.uniform(0.95, 1.05) if i > 20 else None
        )
        encodings.append(t4)

    all_encodings = np.array(encodings)

    # Check for NaN/Inf
    metrics['nan_count'] = int(np.isnan(all_encodings).sum())
    metrics['inf_count'] = int(np.isinf(all_encodings).sum())

    # Fourier features should be bounded
    metrics['min_value'] = float(all_encodings.min())
    metrics['max_value'] = float(all_encodings.max())
    metrics['bounded'] = bool(metrics['min_value'] >= -10 and metrics['max_value'] <= 10)

    # Variance per dimension
    dim_vars = all_encodings.var(axis=0)
    metrics['zero_variance_dims'] = int((dim_vars < 1e-10).sum())

    # Quality score
    quality_score = 100
    if metrics['nan_count'] > 0:
        quality_score -= 40
    if metrics['inf_count'] > 0:
        quality_score -= 40
    if not metrics['bounded']:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_t5_encoder(sample_size: int = 100) -> Dict[str, Any]:
    """
    Evaluate T5 Store Context encoder quality.
    """
    from src.tensor_preparation import StoreContextEncoder

    metrics = {}
    encoder = StoreContextEncoder()

    # Generate sample encodings
    formats = ['LS', 'MS', 'CS', 'DC']
    regions = ['E01', 'E02', 'N01', 'S01', 'W01']

    encodings = []
    for i in range(sample_size):
        t5 = encoder.encode_store(
            store_id=f'STORE{i:03d}',
            store_format=np.random.choice(formats),
            store_region=np.random.choice(regions),
            operational_features={
                'store_size': np.random.uniform(0, 1),
                'traffic': np.random.uniform(0, 1),
                'competition': np.random.uniform(0, 1),
                'store_age': np.random.uniform(0, 1)
            } if i > 20 else None
        )
        encodings.append(t5)

    all_encodings = np.array(encodings)

    # Check for NaN/Inf
    metrics['nan_count'] = int(np.isnan(all_encodings).sum())
    metrics['inf_count'] = int(np.isinf(all_encodings).sum())

    # Distribution statistics
    metrics['mean'] = float(all_encodings.mean())
    metrics['std'] = float(all_encodings.std())

    # Variance per dimension
    dim_vars = all_encodings.var(axis=0)
    metrics['zero_variance_dims'] = int((dim_vars < 1e-10).sum())

    # Quality score
    quality_score = 100
    if metrics['nan_count'] > 0:
        quality_score -= 40
    if metrics['inf_count'] > 0:
        quality_score -= 40
    if metrics['zero_variance_dims'] > encoder.output_dim * 0.1:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_t6_encoder(sample_size: int = 100) -> Dict[str, Any]:
    """
    Evaluate T6 Trip Context encoder quality.
    """
    from src.tensor_preparation import TripContextEncoder

    metrics = {}
    encoder = TripContextEncoder()

    # Generate sample encodings
    mission_types = ['Full Shop', 'Top Up', 'Specific Need', 'Browse']
    mission_focuses = ['Fresh', 'Grocery', 'Mixed', 'Non-Food']
    sensitivities = ['LA', 'MM', 'UM']
    sizes = ['S', 'M', 'L']

    encodings = []
    for i in range(sample_size):
        t6 = encoder.encode_trip(
            mission_type=np.random.choice(mission_types),
            mission_focus=np.random.choice(mission_focuses),
            price_sensitivity=np.random.choice(sensitivities),
            basket_size=np.random.choice(sizes)
        )
        encodings.append(t6)

    all_encodings = np.array(encodings)

    # Check for NaN/Inf
    metrics['nan_count'] = int(np.isnan(all_encodings).sum())
    metrics['inf_count'] = int(np.isinf(all_encodings).sum())

    # Distribution statistics
    metrics['mean'] = float(all_encodings.mean())
    metrics['std'] = float(all_encodings.std())

    # Variance per dimension
    dim_vars = all_encodings.var(axis=0)
    metrics['zero_variance_dims'] = int((dim_vars < 1e-10).sum())

    # Quality score
    quality_score = 100
    if metrics['nan_count'] > 0:
        quality_score -= 40
    if metrics['inf_count'] > 0:
        quality_score -= 40
    if metrics['zero_variance_dims'] > encoder.output_dim * 0.1:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_dataset(project_root: Path) -> Dict[str, Any]:
    """
    Evaluate dataset construction quality.
    """
    metrics = {}

    # Check required files
    features_dir = project_root / 'data' / 'features'
    processed_dir = project_root / 'data' / 'processed'

    required_files = {
        'product_embeddings': features_dir / 'product_embeddings.pkl',
        'customer_embeddings': features_dir / 'customer_embeddings.parquet',
        'price_features': features_dir / 'price_features.parquet',
        'store_features': features_dir / 'store_features.parquet',
        'mission_patterns': processed_dir / 'customer_mission_patterns.parquet',
    }

    metrics['files_present'] = {}
    for name, path in required_files.items():
        metrics['files_present'][name] = path.exists()

    metrics['all_files_present'] = all(metrics['files_present'].values())

    # Quality score based on file availability
    present_count = sum(metrics['files_present'].values())
    metrics['quality_score'] = (present_count / len(required_files)) * 100

    return metrics


def run_evaluation(project_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Run complete tensor preparation evaluation.
    """
    results = {}

    print("=" * 60)
    print("Tensor Preparation Evaluation")
    print("=" * 60)

    # Dimension checks
    print("\n--- Dimension Verification ---")
    results['dimensions'] = evaluate_tensor_dimensions()
    print(f"  T1 (Customer): {results['dimensions']['t1_actual']}d (expected 192d) {'✓' if results['dimensions']['t1_correct'] else '✗'}")
    print(f"  T3 (Temporal): {results['dimensions']['t3_actual']}d (expected 64d) {'✓' if results['dimensions']['t3_correct'] else '✗'}")
    print(f"  T4 (Price): {results['dimensions']['t4_actual']}d (expected 64d) {'✓' if results['dimensions']['t4_correct'] else '✗'}")
    print(f"  T5 (Store): {results['dimensions']['t5_actual']}d (expected 96d) {'✓' if results['dimensions']['t5_correct'] else '✗'}")
    print(f"  T6 (Trip): {results['dimensions']['t6_actual']}d (expected 48d) {'✓' if results['dimensions']['t6_correct'] else '✗'}")
    print(f"  Dense Context: {results['dimensions']['dense_context_actual']}d (expected 400d) {'✓' if results['dimensions']['dense_context_correct'] else '✗'}")
    print(f"  Quality Score: {results['dimensions']['quality_score']:.0f}/100")

    # T1 Encoder
    print("\n--- T1: Customer Context Encoder ---")
    results['t1_encoder'] = evaluate_t1_encoder()
    print(f"  NaN values: {results['t1_encoder']['nan_count']}")
    print(f"  Inf values: {results['t1_encoder']['inf_count']}")
    print(f"  Zero variance dims: {results['t1_encoder']['zero_variance_dims']}")
    print(f"  Quality Score: {results['t1_encoder']['quality_score']:.0f}/100")

    # T2 Encoder
    print("\n--- T2: Product Sequence Encoder ---")
    embeddings_path = project_root / 'data' / 'features' / 'product_embeddings.pkl'
    results['t2_encoder'] = evaluate_t2_encoder(embeddings_path if embeddings_path.exists() else None)
    print(f"  Vocab size: {results['t2_encoder']['vocab_size']}")
    print(f"  Embedding dim: {results['t2_encoder']['embedding_dim']}")
    print(f"  Has EOS token: {results['t2_encoder']['has_eos']}")
    print(f"  Quality Score: {results['t2_encoder']['quality_score']:.0f}/100")

    # T3 Encoder
    print("\n--- T3: Temporal Context Encoder ---")
    results['t3_encoder'] = evaluate_t3_encoder()
    print(f"  NaN values: {results['t3_encoder']['nan_count']}")
    print(f"  Zero variance dims: {results['t3_encoder']['zero_variance_dims']}")
    print(f"  Quality Score: {results['t3_encoder']['quality_score']:.0f}/100")

    # T4 Encoder
    print("\n--- T4: Price Context Encoder ---")
    results['t4_encoder'] = evaluate_t4_encoder()
    print(f"  NaN values: {results['t4_encoder']['nan_count']}")
    print(f"  Values bounded: {results['t4_encoder']['bounded']}")
    print(f"  Quality Score: {results['t4_encoder']['quality_score']:.0f}/100")

    # T5 Encoder
    print("\n--- T5: Store Context Encoder ---")
    results['t5_encoder'] = evaluate_t5_encoder()
    print(f"  NaN values: {results['t5_encoder']['nan_count']}")
    print(f"  Zero variance dims: {results['t5_encoder']['zero_variance_dims']}")
    print(f"  Quality Score: {results['t5_encoder']['quality_score']:.0f}/100")

    # T6 Encoder
    print("\n--- T6: Trip Context Encoder ---")
    results['t6_encoder'] = evaluate_t6_encoder()
    print(f"  NaN values: {results['t6_encoder']['nan_count']}")
    print(f"  Zero variance dims: {results['t6_encoder']['zero_variance_dims']}")
    print(f"  Quality Score: {results['t6_encoder']['quality_score']:.0f}/100")

    # Dataset
    print("\n--- Dataset Construction ---")
    results['dataset'] = evaluate_dataset(project_root)
    print(f"  Files present: {sum(results['dataset']['files_present'].values())}/{len(results['dataset']['files_present'])}")
    for name, present in results['dataset']['files_present'].items():
        status = '✓' if present else '✗'
        print(f"    {name}: {status}")
    print(f"  Quality Score: {results['dataset']['quality_score']:.0f}/100")

    # Overall score
    scores = [r['quality_score'] for r in results.values() if 'quality_score' in r]
    overall_score = np.mean(scores) if scores else 0

    print("\n" + "=" * 60)
    print(f"Overall Quality Score: {overall_score:.1f}/100")
    print("=" * 60)

    results['overall'] = {
        'quality_score': overall_score,
        'components_evaluated': len(scores),
    }

    return results


def main():
    """Run evaluation and save results."""
    project_root = Path(__file__).parent.parent
    results = run_evaluation(project_root)

    # Save results
    output_path = project_root / 'evals' / 'tensor_preparation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
