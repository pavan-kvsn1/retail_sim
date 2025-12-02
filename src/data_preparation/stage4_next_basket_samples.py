"""
Stage 4: Next-Basket Prediction Samples
=======================================
Creates training samples for autoregressive next-basket prediction.

Input: customer_histories.parquet + temporal_metadata.parquet + transactions
Output:
  - train_next_basket.parquet
  - val_next_basket.parquet
  - test_next_basket.parquet

Each sample contains:
  - input_basket_id: Basket at time t (full, unmasked)
  - target_basket_id: Basket at time t+1 (to predict)
  - customer_history: All baskets before time t
  - temporal context for t+1 (when prediction is made)

This enables proper world model training where we predict FUTURE behavior.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import warnings

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings('ignore')


class NextBasketSampleCreator:
    """
    Creates (basket_t, basket_t+1) pairs for next-basket prediction.

    For a world model / RL environment, we need to predict what a customer
    will buy NEXT, given their history and current context.
    """

    def __init__(
        self,
        min_basket_size: int = 2,
        min_history_baskets: int = 1,
        max_time_gap_weeks: int = 12,
    ):
        """
        Parameters
        ----------
        min_basket_size : int
            Minimum products in target basket
        min_history_baskets : int
            Minimum prior baskets required (0 = allow cold start)
        max_time_gap_weeks : int
            Maximum weeks between basket_t and basket_t+1 (filter outliers)
        """
        self.min_basket_size = min_basket_size
        self.min_history_baskets = min_history_baskets
        self.max_time_gap_weeks = max_time_gap_weeks

    def run(
        self,
        transactions_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Create next-basket prediction samples.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions
        metadata_df : pd.DataFrame
            Temporal metadata with splits

        Returns
        -------
        Dict[str, pd.DataFrame]
            {'train': df, 'validation': df, 'test': df}
        """
        print("Stage 4: Next-Basket Prediction Samples")
        print("=" * 60)

        # Step 1: Build basket -> products mapping
        print("\nStep 1: Building basket-product mappings...")
        basket_products = self._build_basket_products(transactions_df)
        print(f"  - Mapped {len(basket_products):,} baskets")

        # Step 2: Build customer timelines (sorted by time)
        print("\nStep 2: Building customer timelines...")
        customer_timelines = self._build_customer_timelines(metadata_df)
        print(f"  - Built timelines for {len(customer_timelines):,} customers")

        # Step 3: Create consecutive basket pairs
        print("\nStep 3: Creating consecutive basket pairs...")
        pairs = self._create_basket_pairs(customer_timelines, metadata_df, basket_products)
        print(f"  - Created {len(pairs):,} basket pairs")

        # Step 4: Split by temporal boundaries
        print("\nStep 4: Splitting by temporal boundaries...")
        split_samples = self._split_samples(pairs, metadata_df)
        self._print_stats(split_samples)

        print("\n" + "=" * 60)
        print("Next-Basket Sample Creation Complete!")

        return split_samples

    def _build_basket_products(self, df: pd.DataFrame) -> Dict[int, List[str]]:
        """Build mapping from basket_id to list of products."""
        return df.groupby('BASKET_ID')['PROD_CODE'].apply(list).to_dict()

    def _build_customer_timelines(
        self,
        metadata_df: pd.DataFrame
    ) -> Dict[str, List[Dict]]:
        """
        Build sorted timeline of baskets per customer.

        Returns dict: customer_id -> [
            {'basket_id': x, 'week': w, 'store_id': s, 'weekday': d, 'hour': h, ...},
            ...
        ]
        """
        # Determine which columns to keep
        cols_to_keep = ['basket_id', 'week', 'store_id', 'split', 'bucket']
        for optional_col in ['shop_weekday', 'shop_hour', 'shop_date']:
            if optional_col in metadata_df.columns:
                cols_to_keep.append(optional_col)

        # Sort by customer and week (shop_date may not exist)
        sort_cols = ['customer_id', 'week']
        if 'shop_date' in metadata_df.columns:
            sort_cols.append('shop_date')

        # Use groupby + apply for efficient processing
        # This is MUCH faster than iterrows for 30M+ rows
        sorted_df = metadata_df[['customer_id'] + cols_to_keep].sort_values(sort_cols)

        def group_to_timeline(group):
            return group[cols_to_keep].to_dict('records')

        customer_timelines = sorted_df.groupby('customer_id', sort=False).apply(
            group_to_timeline
        ).to_dict()

        return customer_timelines

    def _create_basket_pairs(
        self,
        customer_timelines: Dict[str, List[Dict]],
        metadata_df: pd.DataFrame,
        basket_products: Dict[int, List[str]]
    ) -> List[Dict]:
        """
        Create (basket_t, basket_t+1) pairs for each customer.

        For customer with baskets [b1, b2, b3, b4]:
        - Pair 1: input=b1, target=b2, history=[]
        - Pair 2: input=b2, target=b3, history=[b1]
        - Pair 3: input=b3, target=b4, history=[b1, b2]
        """
        pairs = []
        skipped_small_target = 0
        skipped_time_gap = 0
        skipped_no_products = 0

        for customer_id, timeline in tqdm(customer_timelines.items(), desc="Creating pairs"):
            if len(timeline) < 2:
                continue  # Need at least 2 baskets for a pair

            for i in range(len(timeline) - 1):
                input_basket = timeline[i]
                target_basket = timeline[i + 1]

                input_bid = input_basket['basket_id']
                target_bid = target_basket['basket_id']

                # Get products
                input_products = basket_products.get(input_bid, [])
                target_products = basket_products.get(target_bid, [])

                # Filter: minimum target basket size
                if len(target_products) < self.min_basket_size:
                    skipped_small_target += 1
                    continue

                # Filter: products must exist
                if not input_products or not target_products:
                    skipped_no_products += 1
                    continue

                # Filter: time gap not too large
                week_gap = target_basket['week'] - input_basket['week']
                if week_gap > self.max_time_gap_weeks:
                    skipped_time_gap += 1
                    continue

                # Count history baskets (don't store full products to save memory)
                # We only need the count - actual products can be loaded during training
                num_history = i  # Number of baskets before current

                # Filter: minimum history (optional)
                if num_history < self.min_history_baskets:
                    continue

                pairs.append({
                    'customer_id': customer_id,
                    # Input basket (t)
                    'input_basket_id': input_bid,
                    'input_week': input_basket['week'],
                    'input_store_id': input_basket['store_id'],
                    'input_products': input_products,
                    'input_weekday': input_basket.get('shop_weekday', 1),
                    'input_hour': input_basket.get('shop_hour', 12),
                    # Target basket (t+1)
                    'target_basket_id': target_bid,
                    'target_week': target_basket['week'],
                    'target_store_id': target_basket['store_id'],
                    'target_products': target_products,
                    'target_weekday': target_basket.get('shop_weekday', 1),
                    'target_hour': target_basket.get('shop_hour', 12),
                    # History metadata (don't store full products to save memory)
                    'num_history_baskets': num_history,
                    # Metadata
                    'week_gap': week_gap,
                    'target_size': len(target_products),
                    'input_size': len(input_products),
                    # Use target basket's split (predicting into that time period)
                    'split': target_basket['split'],
                    'bucket': target_basket.get('bucket', 1),
                })

        print(f"  - Skipped {skipped_small_target:,} (target too small)")
        print(f"  - Skipped {skipped_time_gap:,} (time gap > {self.max_time_gap_weeks} weeks)")
        print(f"  - Skipped {skipped_no_products:,} (missing products)")

        return pairs

    def _split_samples(
        self,
        pairs: List[Dict],
        metadata_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Split samples by the target basket's temporal split."""
        split_samples = {
            'train': [],
            'validation': [],
            'test': []
        }

        for pair in pairs:
            split = pair['split']
            if split in split_samples:
                split_samples[split].append(pair)

        return {
            k: pd.DataFrame(v) if v else pd.DataFrame()
            for k, v in split_samples.items()
        }

    def _print_stats(self, split_samples: Dict[str, pd.DataFrame]) -> None:
        """Print statistics for each split."""
        print("\n  Sample statistics:")

        total = 0
        for split_name, df in split_samples.items():
            if len(df) == 0:
                continue

            count = len(df)
            total += count

            avg_history = df['num_history_baskets'].mean()
            avg_target = df['target_size'].mean()
            avg_input = df['input_size'].mean()
            avg_gap = df['week_gap'].mean()

            print(f"\n  {split_name}:")
            print(f"    - Samples: {count:,}")
            print(f"    - Avg history baskets: {avg_history:.1f}")
            print(f"    - Avg input basket size: {avg_input:.1f} products")
            print(f"    - Avg target basket size: {avg_target:.1f} products")
            print(f"    - Avg week gap (t to t+1): {avg_gap:.1f} weeks")

        print(f"\n  Total samples: {total:,}")

    def save(
        self,
        split_samples: Dict[str, pd.DataFrame],
        output_dir: Path
    ) -> None:
        """Save samples to parquet files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, df in split_samples.items():
            if len(df) == 0:
                continue

            # Serialize list columns
            df_save = df.copy()
            for col in ['input_products', 'target_products', 'history_basket_ids', 'history_products']:
                if col in df_save.columns:
                    df_save[col] = df_save[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, list) else x
                    )

            output_path = output_dir / f'{split_name}_next_basket.parquet'
            df_save.to_parquet(output_path, index=False)
            print(f"Saved {split_name}: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    """Run next-basket sample creation."""
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    metadata_path = project_root / 'data' / 'prepared' / 'temporal_metadata.parquet'
    output_dir = project_root / 'data' / 'prepared'

    # Load transactions with optimized dtypes
    print(f"Loading transactions from {raw_data_path}...")
    transactions_df = pd.read_csv(
        raw_data_path,
        usecols=['BASKET_ID', 'PROD_CODE', 'CUST_CODE'],
        dtype={'PROD_CODE': 'category', 'CUST_CODE': 'category'}
    )
    print(f"  - Loaded {len(transactions_df):,} transactions")
    print(f"  - Memory usage: {transactions_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Load metadata with optimized dtypes
    print(f"\nLoading metadata from {metadata_path}...")
    metadata_df = pd.read_parquet(metadata_path)
    # Convert string columns to category for memory efficiency
    for col in ['customer_id', 'store_id', 'split', 'bucket']:
        if col in metadata_df.columns and metadata_df[col].dtype == 'object':
            metadata_df[col] = metadata_df[col].astype('category')
    print(f"  - Loaded {len(metadata_df):,} baskets")
    print(f"  - Memory usage: {metadata_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Create samples
    creator = NextBasketSampleCreator(
        min_basket_size=2,
        min_history_baskets=0,  # Allow cold start
        max_time_gap_weeks=12,
    )
    split_samples = creator.run(transactions_df, metadata_df)

    # Save
    print("\nSaving samples...")
    creator.save(split_samples, output_dir)

    return split_samples


if __name__ == '__main__':
    main()
