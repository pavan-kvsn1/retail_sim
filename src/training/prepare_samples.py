"""
Stage 1: Enhance Temporal Metadata for World Model Training.

Leverages existing data/prepared/temporal_metadata.parquet and adds:
- shop_weekday: Day of week (1-7, Monday=1)
- shop_hour: Hour of transaction (0-23)
- shop_date: Actual date (YYYYMMDD format)

These columns enable day/hour level evaluation granularity for validation/test sets.

Output:
- data/prepared/train_samples.parquet
- data/prepared/validation_samples.parquet
- data/prepared/test_samples.parquet
- data/prepared/samples_metadata.json

Reference: RetailSim_Data_Pipeline_and_World_Model_Design.md Section 4.7
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def enhance_temporal_metadata(
    project_root: Path,
    output_dir: Optional[Path] = None,
    chunk_size: int = 10_000_000,
) -> Dict[str, Path]:
    """
    Enhance existing temporal_metadata with day/hour columns for evaluation.

    Args:
        project_root: Path to retail_sim project root
        output_dir: Output directory (default: project_root/data/prepared)
        chunk_size: Rows per chunk when reading transactions

    Returns:
        Dict mapping split names to output file paths
    """
    project_root = Path(project_root)
    temporal_metadata_path = project_root / 'data' / 'prepared' / 'temporal_metadata.parquet'
    transactions_path = project_root / 'raw_data' / 'transactions.csv'
    time_path = project_root / 'raw_data' / 'time.csv'

    if output_dir is None:
        output_dir = project_root / 'data' / 'prepared'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing temporal metadata
    logger.info(f"Loading temporal metadata from {temporal_metadata_path}")
    tm = pd.read_parquet(temporal_metadata_path)
    logger.info(f"Loaded {len(tm):,} baskets")

    # Load time reference for week-to-date mapping
    logger.info(f"Loading time reference from {time_path}")
    time_df = pd.read_csv(time_path)
    time_df['date_from'] = pd.to_datetime(time_df['date_from'], format='%Y%m%d')
    week_to_start_date = dict(zip(time_df['shop_week'], time_df['date_from']))

    # Extract basket-level temporal info from transactions
    logger.info("Extracting basket temporal info from transactions...")

    basket_temporal = []
    chunk_num = 0

    for chunk in pd.read_csv(
        transactions_path,
        usecols=['BASKET_ID', 'SHOP_WEEK', 'SHOP_WEEKDAY', 'SHOP_HOUR', 'SHOP_DATE'],
        chunksize=chunk_size,
        dtype={
            'BASKET_ID': 'int64',
            'SHOP_WEEK': 'int32',
            'SHOP_WEEKDAY': 'int8',
            'SHOP_HOUR': 'int8',
            'SHOP_DATE': 'int32',
        }
    ):
        chunk_num += 1
        logger.info(f"Processing chunk {chunk_num} ({len(chunk):,} rows)")

        # Get first row per basket (all rows in a basket have same temporal info)
        basket_info = chunk.groupby('BASKET_ID').first().reset_index()
        basket_info.columns = ['basket_id', 'shop_week_raw', 'shop_weekday', 'shop_hour', 'shop_date']
        basket_temporal.append(basket_info)

    # Combine and deduplicate
    logger.info("Combining basket temporal info...")
    all_basket_temporal = pd.concat(basket_temporal, ignore_index=True)
    all_basket_temporal = all_basket_temporal.drop_duplicates(subset='basket_id')
    logger.info(f"Extracted temporal info for {len(all_basket_temporal):,} baskets")

    # Merge with temporal metadata
    logger.info("Merging temporal info with metadata...")
    enhanced = tm.merge(
        all_basket_temporal[['basket_id', 'shop_weekday', 'shop_hour', 'shop_date']],
        on='basket_id',
        how='left'
    )

    # Check for missing values
    missing_temporal = enhanced['shop_weekday'].isna().sum()
    if missing_temporal > 0:
        logger.warning(f"{missing_temporal:,} baskets missing temporal info")

    # Convert shop_date from YYYYMMDD int to proper date string
    enhanced['shop_date_str'] = enhanced['shop_date'].apply(
        lambda x: f"{int(x) // 10000}-{(int(x) % 10000) // 100:02d}-{int(x) % 100:02d}" if pd.notna(x) else None
    )

    # Ensure proper types (handle NaN values by converting to nullable integer)
    # Use IntegerArray for proper NaN handling
    from pandas.arrays import IntegerArray
    for col, np_dtype, pd_dtype in [('shop_weekday', np.int8, pd.Int8Dtype()),
                                     ('shop_hour', np.int8, pd.Int8Dtype()),
                                     ('shop_date', np.int32, pd.Int32Dtype())]:
        values = enhanced[col].values
        mask = pd.isna(values)
        # Fill NaN with 0 for conversion, then apply mask
        values_filled = np.where(mask, 0, values).astype(np_dtype)
        enhanced[col] = IntegerArray(values_filled, mask=mask)

    # Log statistics
    logger.info(f"\nEnhanced metadata shape: {enhanced.shape}")
    logger.info(f"Columns: {list(enhanced.columns)}")

    # Split statistics
    split_stats = enhanced.groupby('split').agg({
        'basket_id': 'count',
        'customer_id': 'nunique',
        'week': ['min', 'max'],
        'is_cold_start': 'sum'
    })
    logger.info(f"\nSplit Statistics:\n{split_stats}")

    # Hour distribution for validation/test (for evaluation)
    for split_name in ['validation', 'test']:
        split_data = enhanced[enhanced['split'] == split_name]
        hour_dist = split_data['shop_hour'].value_counts().sort_index()
        logger.info(f"\n{split_name} hour distribution (sample):")
        logger.info(hour_dist.head(10).to_string())

    # Save split files
    output_files = {}

    output_columns = [
        'basket_id', 'customer_id', 'store_id',
        'week', 'week_original',  # week is normalized (1-117), week_original is YYYYWW
        'shop_weekday', 'shop_hour', 'shop_date', 'shop_date_str',
        'num_products', 'total_spend', 'total_quantity',
        'history_length', 'bucket',
        'prior_basket_count', 'is_cold_start', 'is_novel_product'
    ]

    for split_name in ['train', 'validation', 'test']:
        split_data = enhanced[enhanced['split'] == split_name].copy()
        split_output = split_data[output_columns]

        output_path = output_dir / f'{split_name}_samples.parquet'
        split_output.to_parquet(output_path, index=False)
        output_files[split_name] = output_path

        logger.info(f"Saved {split_name}: {len(split_output):,} samples to {output_path}")

    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'source': str(temporal_metadata_path),
        'total_baskets': len(enhanced),
        'splits': {
            split: {
                'n_samples': int((enhanced['split'] == split).sum()),
                'n_customers': int(enhanced[enhanced['split'] == split]['customer_id'].nunique()),
                'week_range': [
                    int(enhanced[enhanced['split'] == split]['week'].min()),
                    int(enhanced[enhanced['split'] == split]['week'].max())
                ],
                'week_original_range': [
                    int(enhanced[enhanced['split'] == split]['week_original'].min()),
                    int(enhanced[enhanced['split'] == split]['week_original'].max())
                ],
                'cold_start_count': int(enhanced[(enhanced['split'] == split) & enhanced['is_cold_start']].shape[0])
            }
            for split in ['train', 'validation', 'test']
        },
        'bucket_boundaries': {
            '1': '1-25 weeks history',
            '2': '26-50 weeks history',
            '3': '51-75 weeks history',
            '4': '76-100 weeks history',
            '5': '101+ weeks history'
        },
        'temporal_columns': {
            'shop_weekday': '1-7 (Monday=1, Sunday=7)',
            'shop_hour': '0-23',
            'shop_date': 'YYYYMMDD integer',
            'shop_date_str': 'YYYY-MM-DD string'
        }
    }

    metadata_path = output_dir / 'samples_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return output_files


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare temporal split samples')
    parser.add_argument(
        '--project-root',
        type=str,
        default='/Users/hazymoji/Documents/DataDev/ML Projects/retail_sim',
        help='Path to project root'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10_000_000,
        help='Rows per chunk for memory efficiency'
    )

    args = parser.parse_args()

    output_files = enhance_temporal_metadata(
        project_root=Path(args.project_root),
        chunk_size=args.chunk_size
    )

    print("\nOutput files created:")
    for split, path in output_files.items():
        print(f"  {split}: {path}")
