"""
Stage 3: Training Sample Creation
=================================
Creates training samples bucketed by history length for efficient batching.

Input: customer_histories.parquet + temporal_metadata.parquet
Output:
  - train_samples_bucket_{1-5}.parquet
  - val_samples.parquet
  - test_samples.parquet

Bucketing Strategy (by history weeks):
  - Bucket 1: weeks 1-25 (0-24 weeks history)
  - Bucket 2: weeks 26-50 (25-49 weeks history)
  - Bucket 3: weeks 51-75 (50-74 weeks history)
  - Bucket 4: weeks 76-100 (75-99 weeks history)
  - Bucket 5: weeks 101-117 (100-116 weeks history)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import warnings

warnings.filterwarnings('ignore')


class TrainingSampleCreator:
    """
    Creates training-ready samples with proper structure.

    Samples include:
    - Target basket (products to predict)
    - Customer history (prior baskets)
    - Context (store, time features)
    - Metadata (cold-start flag, split, etc.)
    """

    # Bucket boundaries (by history_length in weeks)
    BUCKET_BOUNDARIES = [0, 25, 50, 75, 100, 117]

    def __init__(
        self,
        mask_ratio: float = 0.15,
        min_basket_size: int = 2
    ):
        """
        Parameters
        ----------
        mask_ratio : float
            Ratio of products to mask for MEM (Masked Event Modeling)
        min_basket_size : int
            Minimum products in target basket (filter smaller)
        """
        self.mask_ratio = mask_ratio
        self.min_basket_size = min_basket_size

    def run(
        self,
        history_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Create training samples for all splits.

        Parameters
        ----------
        history_df : pd.DataFrame
            Customer histories from Stage 2
        metadata_df : pd.DataFrame
            Temporal metadata from Stage 1
        transactions_df : pd.DataFrame
            Raw transactions for target basket extraction

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with keys like 'train_bucket_1', 'validation', 'test'
        """
        print("Stage 3: Training Sample Creation")
        print("=" * 60)

        # Step 1: Build target baskets
        print("\nStep 1: Building target baskets...")
        target_baskets = self._build_target_baskets(transactions_df)
        print(f"  - Built {len(target_baskets):,} target baskets")

        # Step 2: Merge history with targets
        print("\nStep 2: Merging histories with targets...")
        samples_df = self._merge_history_targets(
            history_df, metadata_df, target_baskets
        )
        print(f"  - Created {len(samples_df):,} samples")

        # Step 3: Filter by minimum basket size
        print("\nStep 3: Filtering by minimum basket size...")
        samples_df = samples_df[
            samples_df['target_products'].apply(len) >= self.min_basket_size
        ]
        print(f"  - Remaining: {len(samples_df):,} samples")

        # Step 4: Split into buckets
        print("\nStep 4: Splitting into buckets...")
        split_samples = self._split_into_buckets(samples_df)
        self._print_split_stats(split_samples)

        print("\n" + "=" * 60)
        print("Training Sample Creation Complete!")

        return split_samples

    def _build_target_baskets(
        self,
        transactions_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Build mapping from basket_id to list of products (target)."""
        target_baskets = {}

        grouped = transactions_df.groupby('BASKET_ID')['PROD_CODE'].apply(list)

        for basket_id, products in grouped.items():
            target_baskets[basket_id] = list(set(products))  # Unique products

        return target_baskets

    def _merge_history_targets(
        self,
        history_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        target_baskets: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Merge customer histories with target baskets."""
        samples = []

        # Merge history with full metadata
        merged = history_df.merge(
            metadata_df[['basket_id', 'store_id', 'total_spend', 'num_products']],
            on='basket_id',
            how='left'
        )

        for _, row in merged.iterrows():
            basket_id = row['basket_id']
            target_products = target_baskets.get(basket_id, [])

            if not target_products:
                continue

            # Parse history products if stored as JSON string
            history_products = row['history_products']
            if isinstance(history_products, str):
                history_products = json.loads(history_products)

            samples.append({
                'basket_id': basket_id,
                'customer_id': row['customer_id'],
                'store_id': row['store_id'],
                'week': row['week'],
                'split': row['split'],
                'bucket': row['bucket'],
                'is_cold_start': row['is_cold_start'],
                'num_prior_baskets': row['num_prior_baskets'],
                'history_products': history_products,
                'target_products': target_products,
                'target_size': len(target_products),
                'total_spend': row['total_spend']
            })

        return pd.DataFrame(samples)

    def _split_into_buckets(
        self,
        samples_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Split samples into train buckets and val/test sets."""
        split_samples = {}

        # Training: split by bucket
        train_df = samples_df[samples_df['split'] == 'train']
        for bucket in range(1, 6):
            bucket_df = train_df[train_df['bucket'] == bucket]
            if len(bucket_df) > 0:
                split_samples[f'train_bucket_{bucket}'] = bucket_df.copy()

        # Validation and test: keep as single files
        val_df = samples_df[samples_df['split'] == 'validation']
        if len(val_df) > 0:
            split_samples['validation'] = val_df.copy()

        test_df = samples_df[samples_df['split'] == 'test']
        if len(test_df) > 0:
            split_samples['test'] = test_df.copy()

        return split_samples

    def _print_split_stats(self, split_samples: Dict[str, pd.DataFrame]) -> None:
        """Print statistics for each split."""
        print("\n  Sample counts per split:")

        total = 0
        for name, df in sorted(split_samples.items()):
            count = len(df)
            total += count

            # Additional stats
            avg_history = df['num_prior_baskets'].mean()
            avg_target = df['target_size'].mean()
            cold_start_pct = df['is_cold_start'].mean() * 100

            print(f"\n  {name}:")
            print(f"    - Samples: {count:,}")
            print(f"    - Avg history length: {avg_history:.1f} baskets")
            print(f"    - Avg target size: {avg_target:.1f} products")
            print(f"    - Cold-start: {cold_start_pct:.1f}%")

        print(f"\n  Total samples: {total:,}")

    def save(
        self,
        split_samples: Dict[str, pd.DataFrame],
        output_dir: Path
    ) -> None:
        """Save samples to parquet files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in split_samples.items():
            # Serialize list columns
            df_save = df.copy()
            df_save['history_products'] = df_save['history_products'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
            df_save['target_products'] = df_save['target_products'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )

            output_path = output_dir / f'{name}_samples.parquet'
            df_save.to_parquet(output_path, index=False)
            print(f"Saved {name}: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    """Run training sample creation."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    metadata_path = project_root / 'data' / 'prepared' / 'temporal_metadata.parquet'
    history_path = project_root / 'data' / 'prepared' / 'customer_histories.parquet'
    output_dir = project_root / 'data' / 'prepared' / 'samples'

    # Load transactions
    print(f"Loading transactions from {raw_data_path}...")
    transactions_df = pd.read_csv(
        raw_data_path,
        usecols=['BASKET_ID', 'PROD_CODE']
    )
    print(f"  - Loaded {len(transactions_df):,} transactions")

    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    metadata_df = pd.read_parquet(metadata_path)
    print(f"  - Loaded {len(metadata_df):,} baskets")

    # Load histories
    print(f"\nLoading histories from {history_path}...")
    history_df = pd.read_parquet(history_path)
    print(f"  - Loaded {len(history_df):,} histories")

    # Create samples
    creator = TrainingSampleCreator()
    split_samples = creator.run(history_df, metadata_df, transactions_df)

    # Save
    print("\nSaving samples...")
    creator.save(split_samples, output_dir)

    return split_samples


if __name__ == '__main__':
    main()
