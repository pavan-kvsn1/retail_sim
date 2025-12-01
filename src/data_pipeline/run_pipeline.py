"""
Data Pipeline Runner
====================
Runs all 4 stages of the data pipeline sequentially.

Usage:
    python run_pipeline.py --nrows 10000
    python run_pipeline.py --all  # Process all rows

Output files (in data/processed/):
    - prices_derived.parquet
    - product_graph.pkl
    - customer_store_affinity.parquet
    - customer_mission_patterns.parquet
"""

import argparse
import pandas as pd
from pathlib import Path
import time

from stage1_price_derivation import PriceDerivationPipeline, load_transactions_sample as load_stage1
from stage2_product_graph import ProductGraphBuilder, load_transactions_sample as load_stage2
from stage3_customer_store_affinity import CustomerStoreAffinityPipeline, load_transactions_sample as load_stage3
from stage4_mission_patterns import MissionPatternPipeline, load_transactions_sample as load_stage4


def run_full_pipeline(nrows: int = 10000, process_all: bool = False):
    """
    Run complete data pipeline on sample data.

    Parameters
    ----------
    nrows : int
        Number of transaction rows to process (ignored if process_all=True)
    process_all : bool
        If True, process all rows in the dataset
    """
    print("=" * 70)
    print("RetailSim Data Pipeline - Section 2")
    print("=" * 70)
    
    if process_all:
        print("\nProcessing ALL transaction rows")
        nrows = None
    else:
        print(f"\nProcessing {nrows:,} transaction rows")

    # Paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Load full sample once (for efficiency)
    print("\n" + "=" * 70)
    print("Loading transaction data...")
    print("=" * 70)

    all_columns = [
        'PROD_CODE', 'PROD_CODE_10', 'PROD_CODE_20', 'PROD_CODE_30', 'PROD_CODE_40',
        'STORE_CODE', 'STORE_REGION', 'SHOP_WEEK',
        'SPEND', 'QUANTITY', 'CUST_CODE', 'BASKET_ID',
        'BASKET_TYPE', 'BASKET_DOMINANT_MISSION',
        'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE'
    ]

    transactions_df = pd.read_csv(
        raw_data_path,
        nrows=nrows,
        usecols=all_columns
    )

    print(f"Loaded {len(transactions_df):,} transactions")
    print(f"  - Products: {transactions_df['PROD_CODE'].nunique():,}")
    print(f"  - Stores: {transactions_df['STORE_CODE'].nunique():,}")
    print(f"  - Customers: {transactions_df['CUST_CODE'].nunique():,}")
    print(f"  - Baskets: {transactions_df['BASKET_ID'].nunique():,}")
    print(f"  - Weeks: {transactions_df['SHOP_WEEK'].nunique():,}")

    # Stage 1: Price Derivation
    print("\n" + "=" * 70)
    stage1_start = time.time()
    pipeline1 = PriceDerivationPipeline()
    prices_df = pipeline1.run(transactions_df)
    prices_df.to_parquet(output_dir / 'prices_derived.parquet', index=False)
    print(f"\nStage 1 completed in {time.time() - stage1_start:.1f}s")

    # Stage 2: Product Graph
    print("\n" + "=" * 70)
    stage2_start = time.time()
    builder = ProductGraphBuilder(
        min_copurchase_count=5,  # Lower for sample data
        top_k_complements=10
    )
    graph = builder.run(transactions_df)
    builder.save(str(output_dir / 'product_graph.pkl'))
    print(f"\nStage 2 completed in {time.time() - stage2_start:.1f}s")

    # Stage 3: Customer-Store Affinity
    print("\n" + "=" * 70)
    stage3_start = time.time()
    pipeline3 = CustomerStoreAffinityPipeline()
    affinity_df = pipeline3.run(transactions_df)
    affinity_df.to_parquet(output_dir / 'customer_store_affinity.parquet', index=False)
    print(f"\nStage 3 completed in {time.time() - stage3_start:.1f}s")

    # Stage 4: Mission Patterns
    print("\n" + "=" * 70)
    stage4_start = time.time()
    pipeline4 = MissionPatternPipeline(min_trips=2)  # Lower for sample
    patterns_df = pipeline4.run(transactions_df)
    patterns_df.to_parquet(output_dir / 'customer_mission_patterns.parquet', index=False)
    print(f"\nStage 4 completed in {time.time() - stage4_start:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {time.time() - total_start:.1f}s")
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'prices_derived.parquet'}")
    print(f"  - {output_dir / 'product_graph.pkl'}")
    print(f"  - {output_dir / 'customer_store_affinity.parquet'}")
    print(f"  - {output_dir / 'customer_mission_patterns.parquet'}")

    # Output summary
    print(f"\nData Summary:")
    print(f"  - Prices: {len(prices_df):,} records")
    print(f"  - Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    print(f"  - Customer Affinity: {len(affinity_df):,} customers")
    print(f"  - Mission Patterns: {len(patterns_df):,} customers")

    return {
        'prices': prices_df,
        'graph': graph,
        'affinity': affinity_df,
        'patterns': patterns_df
    }


def main():
    parser = argparse.ArgumentParser(description='Run RetailSim Data Pipeline')
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

    run_full_pipeline(nrows=args.nrows, process_all=args.all)


if __name__ == '__main__':
    main()
