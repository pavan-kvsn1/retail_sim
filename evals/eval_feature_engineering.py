"""
Evaluation Script for Section 3: Feature Engineering
====================================================
Evaluates quality of generated features and embeddings.

Metrics:
- Embedding quality (variance, coverage)
- Feature distributions
- Consistency checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, Any
from scipy.stats import entropy


def evaluate_pseudo_brands(brands_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate pseudo-brand inference quality.
    """
    metrics = {}

    # Coverage
    metrics['total_products'] = len(brands_df)
    metrics['unique_brands'] = brands_df['pseudo_brand_id'].nunique()
    metrics['brands_per_product'] = metrics['unique_brands'] / max(metrics['total_products'], 1)

    # Brand size distribution
    brand_sizes = brands_df.groupby('pseudo_brand_id').size()
    metrics['avg_brand_size'] = float(brand_sizes.mean())
    metrics['max_brand_size'] = int(brand_sizes.max())
    metrics['min_brand_size'] = int(brand_sizes.min())
    metrics['singleton_brands'] = int((brand_sizes == 1).sum())
    metrics['singleton_pct'] = float(metrics['singleton_brands'] / max(len(brand_sizes), 1))

    # Price tier distribution
    if 'price_tier' in brands_df.columns:
        tier_dist = brands_df['price_tier'].value_counts(normalize=True).to_dict()
        metrics['tier_distribution'] = tier_dist

    # Quality score
    quality_score = 100
    if metrics['singleton_pct'] > 0.5:
        quality_score -= 20  # Too many single-product brands
    if metrics['unique_brands'] < 10:
        quality_score -= 20  # Too few brands

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_price_features(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate Fourier price encoding quality.
    """
    metrics = {}

    # Coverage
    metrics['total_records'] = len(features_df)

    # Feature columns
    feature_cols = [c for c in features_df.columns
                   if c.startswith(('fourier_', 'log_', 'relative_', 'velocity_'))]
    metrics['num_features'] = len(feature_cols)

    # Check for NaN/Inf
    nan_count = features_df[feature_cols].isna().sum().sum()
    inf_count = np.isinf(features_df[feature_cols].values).sum()
    metrics['nan_values'] = int(nan_count)
    metrics['inf_values'] = int(inf_count)

    # Feature statistics
    feature_means = features_df[feature_cols].mean()
    feature_stds = features_df[feature_cols].std()

    metrics['avg_feature_mean'] = float(feature_means.mean())
    metrics['avg_feature_std'] = float(feature_stds.mean())
    metrics['zero_variance_features'] = int((feature_stds < 1e-10).sum())

    # Feature bounds (Fourier should be bounded)
    fourier_cols = [c for c in feature_cols if c.startswith('fourier_')]
    if fourier_cols:
        max_vals = features_df[fourier_cols].max().max()
        min_vals = features_df[fourier_cols].min().min()
        metrics['fourier_max'] = float(max_vals)
        metrics['fourier_min'] = float(min_vals)

    # Quality score
    quality_score = 100
    if metrics['nan_values'] > 0:
        quality_score -= 30
    if metrics['inf_values'] > 0:
        quality_score -= 30
    if metrics['zero_variance_features'] > metrics['num_features'] * 0.1:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_product_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Evaluate GraphSAGE product embedding quality.
    """
    metrics = {}

    # Coverage
    metrics['num_products'] = len(embeddings)

    if metrics['num_products'] == 0:
        metrics['quality_score'] = 0
        return metrics

    # Embedding dimension
    sample_embed = next(iter(embeddings.values()))
    metrics['embedding_dim'] = int(sample_embed.shape[0])

    # Collect all embeddings
    all_embeds = np.array(list(embeddings.values()))

    # Check for NaN/Inf
    metrics['nan_embeddings'] = int(np.isnan(all_embeds).any(axis=1).sum())
    metrics['inf_embeddings'] = int(np.isinf(all_embeds).any(axis=1).sum())

    # Embedding statistics
    norms = np.linalg.norm(all_embeds, axis=1)
    metrics['avg_norm'] = float(norms.mean())
    metrics['std_norm'] = float(norms.std())
    metrics['min_norm'] = float(norms.min())
    metrics['max_norm'] = float(norms.max())
    metrics['zero_norm_count'] = int((norms < 1e-10).sum())

    # Variance per dimension
    dim_vars = all_embeds.var(axis=0)
    metrics['avg_dim_variance'] = float(dim_vars.mean())
    metrics['zero_variance_dims'] = int((dim_vars < 1e-10).sum())

    # Embedding diversity (average pairwise distance)
    if len(all_embeds) > 1:
        sample_size = min(100, len(all_embeds))
        sample_idx = np.random.choice(len(all_embeds), sample_size, replace=False)
        sample = all_embeds[sample_idx]
        dists = []
        for i in range(min(50, sample_size)):
            for j in range(i+1, min(50, sample_size)):
                dists.append(np.linalg.norm(sample[i] - sample[j]))
        metrics['avg_pairwise_distance'] = float(np.mean(dists)) if dists else 0

    # Quality score
    quality_score = 100
    if metrics['nan_embeddings'] > 0:
        quality_score -= 30
    if metrics['zero_norm_count'] > 0:
        quality_score -= 20
    if metrics['zero_variance_dims'] > metrics['embedding_dim'] * 0.1:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_customer_embeddings(embeddings_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate customer history embedding quality.
    """
    metrics = {}

    # Coverage
    metrics['num_customers'] = len(embeddings_df)

    if metrics['num_customers'] == 0:
        metrics['quality_score'] = 0
        return metrics

    # Embedding columns
    embed_cols = [c for c in embeddings_df.columns if c.startswith('embed_')]
    metrics['embedding_dim'] = len(embed_cols)

    # Get embeddings as array
    all_embeds = embeddings_df[embed_cols].values

    # Check for NaN/Inf
    metrics['nan_embeddings'] = int(np.isnan(all_embeds).any(axis=1).sum())
    metrics['inf_embeddings'] = int(np.isinf(all_embeds).any(axis=1).sum())

    # Embedding statistics
    norms = np.linalg.norm(all_embeds, axis=1)
    metrics['avg_norm'] = float(norms.mean())
    metrics['std_norm'] = float(norms.std())

    # Variance per dimension
    dim_vars = all_embeds.var(axis=0)
    metrics['avg_dim_variance'] = float(dim_vars.mean())
    metrics['zero_variance_dims'] = int((dim_vars < 1e-10).sum())

    # Trip count analysis
    if 'total_trips' in embeddings_df.columns:
        metrics['avg_trips'] = float(embeddings_df['total_trips'].mean())
        metrics['cold_start_customers'] = int((embeddings_df['total_trips'] < 5).sum())

    # Quality score
    quality_score = 100
    if metrics['nan_embeddings'] > 0:
        quality_score -= 30
    if metrics['zero_variance_dims'] > metrics['embedding_dim'] * 0.1:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_store_features(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate store context feature quality.
    """
    metrics = {}

    # Coverage
    metrics['num_stores'] = len(features_df)

    if metrics['num_stores'] == 0:
        metrics['quality_score'] = 0
        return metrics

    # Feature columns
    feature_cols = [c for c in features_df.columns
                   if c.startswith(('identity_', 'format_', 'region_', 'operational_'))]
    metrics['num_features'] = len(feature_cols)

    # Check for NaN/Inf
    nan_count = features_df[feature_cols].isna().sum().sum()
    metrics['nan_values'] = int(nan_count)

    # Feature statistics
    all_features = features_df[feature_cols].values
    metrics['avg_feature_mean'] = float(all_features.mean())
    metrics['avg_feature_std'] = float(all_features.std())

    # Store format distribution
    if 'format' in features_df.columns:
        format_dist = features_df['format'].value_counts().to_dict()
        metrics['format_distribution'] = format_dist

    # Quality score
    quality_score = 100
    if metrics['nan_values'] > 0:
        quality_score -= 30
    if metrics['num_features'] != 96:
        quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def run_evaluation(project_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Run complete feature engineering evaluation.
    """
    results = {}
    features_dir = project_root / 'data' / 'features'

    print("=" * 60)
    print("Feature Engineering Evaluation")
    print("=" * 60)

    # Layer 1: Pseudo-Brands
    print("\n--- Layer 1: Pseudo-Brands ---")
    brands_path = features_dir / 'pseudo_brands.parquet'
    if brands_path.exists():
        brands_df = pd.read_parquet(brands_path)
        results['pseudo_brands'] = evaluate_pseudo_brands(brands_df)
        print(f"  Products: {results['pseudo_brands']['total_products']:,}")
        print(f"  Brands: {results['pseudo_brands']['unique_brands']:,}")
        print(f"  Quality Score: {results['pseudo_brands']['quality_score']}/100")
    else:
        print("  [MISSING] pseudo_brands.parquet")
        results['pseudo_brands'] = {'quality_score': 0, 'error': 'file not found'}

    # Layer 2: Price Features
    print("\n--- Layer 2: Price Features ---")
    price_path = features_dir / 'price_features.parquet'
    if price_path.exists():
        price_df = pd.read_parquet(price_path)
        results['price_features'] = evaluate_price_features(price_df)
        print(f"  Records: {results['price_features']['total_records']:,}")
        print(f"  Features: {results['price_features']['num_features']}")
        print(f"  Quality Score: {results['price_features']['quality_score']}/100")
    else:
        print("  [MISSING] price_features.parquet")
        results['price_features'] = {'quality_score': 0, 'error': 'file not found'}

    # Layer 3: Product Embeddings
    print("\n--- Layer 3: Product Embeddings ---")
    embed_path = features_dir / 'product_embeddings.pkl'
    if embed_path.exists():
        with open(embed_path, 'rb') as f:
            data = pickle.load(f)
            embeddings = data.get('embeddings', data)
        results['product_embeddings'] = evaluate_product_embeddings(embeddings)
        print(f"  Products: {results['product_embeddings']['num_products']:,}")
        print(f"  Dimension: {results['product_embeddings'].get('embedding_dim', 'N/A')}")
        print(f"  Quality Score: {results['product_embeddings']['quality_score']}/100")
    else:
        print("  [MISSING] product_embeddings.pkl")
        results['product_embeddings'] = {'quality_score': 0, 'error': 'file not found'}

    # Layer 4: Customer Embeddings
    print("\n--- Layer 4: Customer Embeddings ---")
    cust_path = features_dir / 'customer_embeddings.parquet'
    if cust_path.exists():
        cust_df = pd.read_parquet(cust_path)
        results['customer_embeddings'] = evaluate_customer_embeddings(cust_df)
        print(f"  Customers: {results['customer_embeddings']['num_customers']:,}")
        print(f"  Dimension: {results['customer_embeddings'].get('embedding_dim', 'N/A')}")
        print(f"  Quality Score: {results['customer_embeddings']['quality_score']}/100")
    else:
        print("  [MISSING] customer_embeddings.parquet")
        results['customer_embeddings'] = {'quality_score': 0, 'error': 'file not found'}

    # Layer 5: Store Features
    print("\n--- Layer 5: Store Features ---")
    store_path = features_dir / 'store_features.parquet'
    if store_path.exists():
        store_df = pd.read_parquet(store_path)
        results['store_features'] = evaluate_store_features(store_df)
        print(f"  Stores: {results['store_features']['num_stores']:,}")
        print(f"  Features: {results['store_features'].get('num_features', 'N/A')}")
        print(f"  Quality Score: {results['store_features']['quality_score']}/100")
    else:
        print("  [MISSING] store_features.parquet")
        results['store_features'] = {'quality_score': 0, 'error': 'file not found'}

    # Overall score
    scores = [r['quality_score'] for r in results.values() if 'quality_score' in r]
    overall_score = np.mean(scores) if scores else 0

    print("\n" + "=" * 60)
    print(f"Overall Quality Score: {overall_score:.1f}/100")
    print("=" * 60)

    results['overall'] = {
        'quality_score': overall_score,
        'layers_evaluated': len(scores),
        'all_files_present': all('error' not in r for r in results.values())
    }

    return results


def main():
    """Run evaluation and save results."""
    project_root = Path(__file__).parent.parent
    results = run_evaluation(project_root)

    # Save results
    output_path = project_root / 'evals' / 'feature_engineering_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
