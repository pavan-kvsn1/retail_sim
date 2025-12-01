"""
Stage 1: Temporal Metadata Creation
====================================
Creates basket-level metadata with temporal split assignments.

Input: Raw transactions
Output: temporal_metadata.parquet

Split Boundaries:
- Training: Weeks 1-80 (72%)
- Validation: Weeks 81-95 (13%)
- Test: Weeks 96-117 (15%)

Flags:
- is_cold_start: Customer has <5 baskets in prior history
- is_novel_product: Basket contains products not seen in training
- bucket: History length bucket (1-5)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class TemporalMetadataCreator:
    """
    Creates temporal metadata for train/val/test splits.

    Assigns each basket to a split based on week number and computes
    metadata flags for cold-start evaluation and bucketing.
    """

    # Split boundaries (from design doc)
    TRAIN_END_WEEK = 80
    VAL_END_WEEK = 95
    # Test: 96-117

    # History length buckets
    BUCKET_BOUNDARIES = [0, 25, 50, 75, 100, 117]

    # Cold-start threshold
    COLD_START_THRESHOLD = 5

    def __init__(
        self,
        train_end_week: int = 80,
        val_end_week: int = 95,
        cold_start_threshold: int = 5
    ):
        """
        Parameters
        ----------
        train_end_week : int
            Last week of training set (default: 80)
        val_end_week : int
            Last week of validation set (default: 95)
        cold_start_threshold : int
            Minimum baskets to not be considered cold-start (default: 5)
        """
        self.train_end_week = train_end_week
        self.val_end_week = val_end_week
        self.cold_start_threshold = cold_start_threshold
        self.week_mapping = None  # Will store original -> sequential mapping

    def run(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal metadata for all baskets.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions with columns: BASKET_ID, CUST_CODE, STORE_CODE,
            SHOP_WEEK, PROD_CODE, SPEND, QUANTITY

        Returns
        -------
        pd.DataFrame
            Basket-level metadata with split assignments and flags
        """
        print("Stage 1: Temporal Metadata Creation")
        print("=" * 60)

        # Step 1: Aggregate to basket level
        print("\nStep 1: Aggregating to basket level...")
        basket_df = self._aggregate_to_basket_level(transactions_df)
        print(f"  - Total baskets: {len(basket_df):,}")

        # Step 2: Assign splits
        print("\nStep 2: Assigning temporal splits...")
        basket_df = self._assign_splits(basket_df)
        self._print_split_stats(basket_df)

        # Step 3: Compute history lengths and buckets
        print("\nStep 3: Computing history lengths...")
        basket_df = self._compute_history_info(basket_df)
        self._print_bucket_stats(basket_df)

        # Step 4: Flag cold-start customers
        print("\nStep 4: Flagging cold-start customers...")
        basket_df = self._flag_cold_start(basket_df, transactions_df)
        self._print_cold_start_stats(basket_df)

        # Step 5: Flag novel products
        print("\nStep 5: Flagging novel products...")
        basket_df = self._flag_novel_products(basket_df, transactions_df)
        self._print_novel_product_stats(basket_df)

        print("\n" + "=" * 60)
        print("Temporal Metadata Complete!")
        print(f"  - Output columns: {list(basket_df.columns)}")

        return basket_df

    def _aggregate_to_basket_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transactions to basket level."""
        # First, normalize SHOP_WEEK to sequential week numbers (1-N)
        # SHOP_WEEK is in YYYYWW format (e.g., 200607 = year 2006, week 7)
        df = df.copy()
        unique_weeks = sorted(df['SHOP_WEEK'].unique())
        week_to_seq = {w: i + 1 for i, w in enumerate(unique_weeks)}
        df['SHOP_WEEK_SEQ'] = df['SHOP_WEEK'].map(week_to_seq)

        print(f"  - Week range: {unique_weeks[0]} to {unique_weeks[-1]} ({len(unique_weeks)} weeks)")
        print(f"  - Mapped to sequential weeks: 1 to {len(unique_weeks)}")

        # Store mapping for later use
        self.week_mapping = week_to_seq
        self.seq_to_week = {v: k for k, v in week_to_seq.items()}

        basket_agg = df.groupby('BASKET_ID').agg({
            'CUST_CODE': 'first',
            'STORE_CODE': 'first',
            'SHOP_WEEK_SEQ': 'first',
            'SHOP_WEEK': 'first',  # Keep original for reference
            'PROD_CODE': 'nunique',  # Number of unique products
            'SPEND': 'sum',
            'QUANTITY': 'sum'
        }).reset_index()

        basket_agg = basket_agg.rename(columns={
            'BASKET_ID': 'basket_id',
            'CUST_CODE': 'customer_id',
            'STORE_CODE': 'store_id',
            'SHOP_WEEK_SEQ': 'week',
            'SHOP_WEEK': 'week_original',
            'PROD_CODE': 'num_products',
            'SPEND': 'total_spend',
            'QUANTITY': 'total_quantity'
        })

        return basket_agg

    def _assign_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign each basket to train/validation/test based on week."""
        df = df.copy()

        conditions = [
            df['week'] <= self.train_end_week,
            (df['week'] > self.train_end_week) & (df['week'] <= self.val_end_week),
            df['week'] > self.val_end_week
        ]
        choices = ['train', 'validation', 'test']

        df['split'] = np.select(conditions, choices, default='unknown')

        return df

    def _print_split_stats(self, df: pd.DataFrame) -> None:
        """Print split statistics."""
        split_counts = df['split'].value_counts()
        total = len(df)

        for split in ['train', 'validation', 'test']:
            count = split_counts.get(split, 0)
            pct = count / total * 100
            print(f"  - {split}: {count:,} baskets ({pct:.1f}%)")

    def _compute_history_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute history length and assign buckets."""
        df = df.copy()

        # History length = week - 1 (all prior weeks available)
        df['history_length'] = df['week'] - 1

        # Assign to buckets based on history length
        df['bucket'] = pd.cut(
            df['history_length'],
            bins=self.BUCKET_BOUNDARIES,
            labels=[1, 2, 3, 4, 5],
            include_lowest=True
        ).astype(int)

        return df

    def _print_bucket_stats(self, df: pd.DataFrame) -> None:
        """Print bucket distribution."""
        print("  Bucket distribution:")
        for split in ['train', 'validation', 'test']:
            split_df = df[df['split'] == split]
            bucket_counts = split_df['bucket'].value_counts().sort_index()
            print(f"    {split}:")
            for bucket, count in bucket_counts.items():
                pct = count / len(split_df) * 100
                print(f"      Bucket {bucket}: {count:,} ({pct:.1f}%)")

    def _flag_cold_start(
        self,
        basket_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Flag baskets where customer has < threshold prior baskets."""
        basket_df = basket_df.copy()

        # Sort by customer and week
        basket_df = basket_df.sort_values(['customer_id', 'week'])

        # Compute cumulative basket count per customer (prior to current basket)
        basket_df['prior_basket_count'] = basket_df.groupby('customer_id').cumcount()

        # Flag cold-start
        basket_df['is_cold_start'] = basket_df['prior_basket_count'] < self.cold_start_threshold

        return basket_df

    def _print_cold_start_stats(self, df: pd.DataFrame) -> None:
        """Print cold-start statistics."""
        for split in ['train', 'validation', 'test']:
            split_df = df[df['split'] == split]
            cold_count = split_df['is_cold_start'].sum()
            pct = cold_count / len(split_df) * 100
            print(f"  - {split}: {cold_count:,} cold-start baskets ({pct:.1f}%)")

    def _flag_novel_products(
        self,
        basket_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Flag baskets containing products not seen in training set."""
        basket_df = basket_df.copy()

        # Get original week values that correspond to training weeks
        # Find all original weeks that map to sequential weeks <= train_end_week
        train_original_weeks = set()
        if self.week_mapping is not None:
            for orig_week, seq_week in self.week_mapping.items():
                if seq_week <= self.train_end_week:
                    train_original_weeks.add(orig_week)

        # Get products in training set
        train_weeks = transactions_df['SHOP_WEEK'].isin(train_original_weeks)
        training_products = set(transactions_df.loc[train_weeks, 'PROD_CODE'].unique())

        print(f"  - Training products: {len(training_products):,}")

        # Get products per basket
        basket_products = transactions_df.groupby('BASKET_ID')['PROD_CODE'].apply(set).to_dict()

        # Flag baskets with novel products
        def has_novel_product(basket_id):
            products = basket_products.get(basket_id, set())
            return len(products - training_products) > 0

        basket_df['is_novel_product'] = basket_df['basket_id'].apply(has_novel_product)

        return basket_df

    def _print_novel_product_stats(self, df: pd.DataFrame) -> None:
        """Print novel product statistics."""
        for split in ['train', 'validation', 'test']:
            split_df = df[df['split'] == split]
            novel_count = split_df['is_novel_product'].sum()
            pct = novel_count / len(split_df) * 100 if len(split_df) > 0 else 0
            print(f"  - {split}: {novel_count:,} baskets with novel products ({pct:.1f}%)")

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save temporal metadata to parquet."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False)
        print(f"\nSaved to: {output_path}")
        print(f"  - Size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    """Run temporal metadata creation."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    output_path = project_root / 'data' / 'prepared' / 'temporal_metadata.parquet'

    # Load transactions
    print(f"Loading transactions from {raw_data_path}...")
    transactions_df = pd.read_csv(
        raw_data_path,
        usecols=['BASKET_ID', 'CUST_CODE', 'STORE_CODE', 'SHOP_WEEK',
                 'PROD_CODE', 'SPEND', 'QUANTITY']
    )
    print(f"  - Loaded {len(transactions_df):,} transactions")

    # Create metadata
    creator = TemporalMetadataCreator()
    metadata_df = creator.run(transactions_df)

    # Save
    creator.save(metadata_df, output_path)

    return metadata_df


if __name__ == '__main__':
    main()
