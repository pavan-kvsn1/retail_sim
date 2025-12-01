"""
Feature Engineering Pipeline Runner
====================================
Runs all 5 layers of feature engineering sequentially.

Requires: Data Pipeline (Section 2) outputs to exist.

Usage:
    python run_feature_engineering.py --nrows 10000
    python run_feature_engineering.py --all  # Process all rows

Output files (in data/features/):
    - pseudo_brands.parquet
    - price_features.parquet
    - product_embeddings.pkl
    - customer_history_embeddings.pkl
    - store_features.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time

from layer1_pseudo_brand import PseudoBrandInference
from layer2_fourier_price import FourierPriceEncoder
from layer3_graph_embeddings import GraphSAGEEncoder
from layer4_customer_history import CustomerHistoryEncoder
from layer5_store_context import StoreContextEncoder


def run_feature_engineering(nrows: int = 10000, process_all: bool = False):
    """
    Run complete feature engineering pipeline.

    Parameters
    ----------
    nrows : int
        Number of transaction rows to process (ignored if process_all=True)
    process_all : bool
        If True, process all rows in the dataset
    """
    print("=" * 70)
    print("RetailSim Feature Engineering - Section 3")
    print("=" * 70)
    
    if process_all:
        print("\nProcessing ALL transaction rows")
        nrows = None
    else:
        print(f"\nProcessing {nrows:,} transaction rows")

    # Paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    processed_dir = project_root / 'data' / 'processed'
    features_dir = project_root / 'data' / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Check that data pipeline outputs exist
    required_files = [
        processed_dir / 'prices_derived.parquet',
        processed_dir / 'product_graph.pkl',
        processed_dir / 'customer_store_affinity.parquet',
        processed_dir / 'customer_mission_patterns.parquet'
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("\nERROR: Missing data pipeline outputs:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run the data pipeline first:")
        print("  python src/data_pipeline/run_pipeline.py --nrows", nrows)
        return None

    # Load data
    print("\n" + "=" * 70)
    print("Loading data...")
    print("=" * 70)

    # Load transactions
    transactions_df = pd.read_csv(
        raw_data_path,
        nrows=nrows,
        usecols=[
            'PROD_CODE', 'PROD_CODE_10', 'PROD_CODE_20', 'PROD_CODE_30', 'PROD_CODE_40',
            'STORE_CODE', 'STORE_FORMAT', 'STORE_REGION',
            'SHOP_WEEK', 'SHOP_HOUR', 'SHOP_WEEKDAY',
            'SPEND', 'QUANTITY', 'CUST_CODE', 'BASKET_ID',
            'BASKET_TYPE', 'BASKET_DOMINANT_MISSION',
            'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE'
        ]
    )
    print(f"  - Transactions: {len(transactions_df):,}")

    # Load pipeline outputs
    prices_df = pd.read_parquet(processed_dir / 'prices_derived.parquet')
    print(f"  - Prices: {len(prices_df):,}")

    with open(processed_dir / 'product_graph.pkl', 'rb') as f:
        product_graph = pickle.load(f)
    print(f"  - Graph: {product_graph.number_of_nodes()} nodes, {product_graph.number_of_edges()} edges")

    customer_affinity = pd.read_parquet(processed_dir / 'customer_store_affinity.parquet')
    print(f"  - Customer affinity: {len(customer_affinity):,}")

    mission_patterns = pd.read_parquet(processed_dir / 'customer_mission_patterns.parquet')
    print(f"  - Mission patterns: {len(mission_patterns):,}")

    # Layer 1: Pseudo-Brand Inference
    print("\n" + "=" * 70)
    layer1_start = time.time()
    inferencer = PseudoBrandInference()
    pseudo_brands = inferencer.run(transactions_df, prices_df, product_graph)
    pseudo_brands.to_parquet(features_dir / 'pseudo_brands.parquet', index=False)
    print(f"\nLayer 1 completed in {time.time() - layer1_start:.1f}s")

    # Layer 2: Fourier Price Encoding
    print("\n" + "=" * 70)
    layer2_start = time.time()
    encoder2 = FourierPriceEncoder()
    price_features = encoder2.run(prices_df)
    price_features.to_parquet(features_dir / 'price_features.parquet', index=False)
    print(f"\nLayer 2 completed in {time.time() - layer2_start:.1f}s")

    # Layer 3: Graph Embeddings
    print("\n" + "=" * 70)
    layer3_start = time.time()
    encoder3 = GraphSAGEEncoder()
    product_embeddings = encoder3.run(product_graph, pseudo_brands)
    encoder3.save(str(features_dir / 'product_embeddings.pkl'))
    print(f"\nLayer 3 completed in {time.time() - layer3_start:.1f}s")

    # Layer 4: Customer History Encoding
    print("\n" + "=" * 70)
    layer4_start = time.time()
    encoder4 = CustomerHistoryEncoder()
    customer_embeddings = encoder4.run(transactions_df, product_embeddings, mission_patterns)
    encoder4.save(str(features_dir / 'customer_history_embeddings.pkl'))

    # Also save as parquet for tensor preparation
    trip_counts = transactions_df.groupby('CUST_CODE')['BASKET_ID'].nunique().to_dict()
    customer_rows = []
    for cust_id, embed in customer_embeddings.items():
        row = {'customer_id': cust_id, 'total_trips': trip_counts.get(cust_id, 0)}
        for i, val in enumerate(embed):
            row[f'embed_{i}'] = val
        customer_rows.append(row)
    customer_df = pd.DataFrame(customer_rows)
    customer_df.to_parquet(features_dir / 'customer_embeddings.parquet', index=False)
    print(f"\nLayer 4 completed in {time.time() - layer4_start:.1f}s")

    # Layer 5: Store Context Features
    print("\n" + "=" * 70)
    layer5_start = time.time()
    encoder5 = StoreContextEncoder()
    store_features = encoder5.run(transactions_df, customer_affinity)
    store_features.to_parquet(features_dir / 'store_features.parquet', index=False)
    print(f"\nLayer 5 completed in {time.time() - layer5_start:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {time.time() - total_start:.1f}s")
    print(f"\nOutput files:")
    print(f"  - {features_dir / 'pseudo_brands.parquet'}")
    print(f"  - {features_dir / 'price_features.parquet'}")
    print(f"  - {features_dir / 'product_embeddings.pkl'}")
    print(f"  - {features_dir / 'customer_history_embeddings.pkl'}")
    print(f"  - {features_dir / 'customer_embeddings.parquet'}")
    print(f"  - {features_dir / 'store_features.parquet'}")

    # Output summary
    print(f"\nFeature Summary:")
    print(f"  - Pseudo-brands: {len(pseudo_brands):,} products, {pseudo_brands['pseudo_brand_id'].nunique():,} brands")
    print(f"  - Price features: {len(price_features):,} observations, 64d")
    print(f"  - Product embeddings: {len(product_embeddings):,} products, 256d")
    print(f"  - Customer embeddings: {len(customer_embeddings):,} customers, 160d")
    print(f"  - Store features: {len(store_features):,} stores, 96d")

    return {
        'pseudo_brands': pseudo_brands,
        'price_features': price_features,
        'product_embeddings': product_embeddings,
        'customer_embeddings': customer_embeddings,
        'store_features': store_features
    }


def main():
    parser = argparse.ArgumentParser(description='Run RetailSim Feature Engineering')
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
    args = parser.parse_args()

    run_feature_engineering(nrows=args.nrows, process_all=args.all)


if __name__ == '__main__':
    main()
