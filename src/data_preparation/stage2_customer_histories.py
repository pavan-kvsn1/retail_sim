"""
Stage 2: Customer History Extraction
====================================
Extracts per-customer shopping histories respecting temporal boundaries.

Input: transactions + temporal_metadata.parquet
Output: customer_histories.parquet

For each customer-basket pair in val/test:
- Extract all prior baskets (within split boundary)
- Build chronological history sequences
- Truncate to max_history_length
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class CustomerHistoryExtractor:
    """
    Extracts customer shopping histories for each split.

    Ensures temporal correctness: histories only contain data
    available at prediction time.
    """

    # Maximum history length (baskets)
    MAX_HISTORY_BASKETS = 50

    # Maximum products per basket in history
    MAX_PRODUCTS_PER_BASKET = 30

    def __init__(
        self,
        max_history_baskets: int = 50,
        max_products_per_basket: int = 30
    ):
        """
        Parameters
        ----------
        max_history_baskets : int
            Maximum number of prior baskets to include in history
        max_products_per_basket : int
            Maximum products per basket (truncate larger baskets)
        """
        self.max_history_baskets = max_history_baskets
        self.max_products_per_basket = max_products_per_basket

    def run(
        self,
        transactions_df: pd.DataFrame,
        metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract customer histories for all baskets.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions with BASKET_ID, CUST_CODE, SHOP_WEEK, PROD_CODE
        metadata_df : pd.DataFrame
            Temporal metadata from Stage 1

        Returns
        -------
        pd.DataFrame
            Customer histories with basket_id, history (list of prior baskets)
        """
        print("Stage 2: Customer History Extraction")
        print("=" * 60)

        # Step 1: Build basket -> products mapping
        print("\nStep 1: Building basket-product mappings...")
        basket_products = self._build_basket_products(transactions_df)
        print(f"  - Mapped {len(basket_products):,} baskets")

        # Step 2: Build customer timeline
        print("\nStep 2: Building customer timelines...")
        customer_baskets = self._build_customer_timelines(metadata_df)
        print(f"  - Built timelines for {len(customer_baskets):,} customers")

        # Step 3: Extract histories for each basket
        print("\nStep 3: Extracting histories...")
        histories = self._extract_histories(
            metadata_df, customer_baskets, basket_products
        )
        print(f"  - Extracted {len(histories):,} histories")

        # Step 4: Create output dataframe
        print("\nStep 4: Creating output dataframe...")
        history_df = self._create_history_dataframe(histories, metadata_df)
        self._print_history_stats(history_df)

        print("\n" + "=" * 60)
        print("Customer History Extraction Complete!")

        return history_df

    def _build_basket_products(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Build mapping from basket_id to list of products."""
        basket_products = {}

        # Group by basket and get products
        grouped = df.groupby('BASKET_ID')['PROD_CODE'].apply(list)

        for basket_id, products in grouped.items():
            # Truncate if too many products
            if len(products) > self.max_products_per_basket:
                products = products[:self.max_products_per_basket]
            basket_products[basket_id] = products

        return basket_products

    def _build_customer_timelines(
        self,
        metadata_df: pd.DataFrame
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        Build sorted timeline of baskets per customer.

        Returns dict: customer_id -> [(week, basket_id), ...]
        """
        customer_baskets = defaultdict(list)

        # Sort by customer and week
        sorted_df = metadata_df.sort_values(['customer_id', 'week'])

        for _, row in sorted_df.iterrows():
            customer_baskets[row['customer_id']].append(
                (row['week'], row['basket_id'])
            )

        return dict(customer_baskets)

    def _extract_histories(
        self,
        metadata_df: pd.DataFrame,
        customer_baskets: Dict[str, List[Tuple[int, str]]],
        basket_products: Dict[str, List[str]]
    ) -> List[Dict]:
        """
        Extract history for each basket.

        History = all prior baskets from same customer, up to max_history_baskets.
        """
        histories = []

        # Create lookup for basket position in customer timeline
        basket_positions = {}
        for customer_id, timeline in customer_baskets.items():
            for idx, (week, basket_id) in enumerate(timeline):
                basket_positions[basket_id] = (customer_id, idx)

        for _, row in metadata_df.iterrows():
            basket_id = row['basket_id']
            customer_id = row['customer_id']

            # Get customer's timeline
            timeline = customer_baskets.get(customer_id, [])

            # Get position of current basket
            if basket_id in basket_positions:
                _, current_idx = basket_positions[basket_id]
            else:
                current_idx = 0

            # Get prior baskets (everything before current index)
            prior_baskets = timeline[:current_idx]

            # Take most recent baskets (truncate from front if too many)
            if len(prior_baskets) > self.max_history_baskets:
                prior_baskets = prior_baskets[-self.max_history_baskets:]

            # Extract products for each prior basket
            history_products = []
            history_weeks = []

            for week, hist_basket_id in prior_baskets:
                products = basket_products.get(hist_basket_id, [])
                if products:
                    history_products.append(products)
                    history_weeks.append(week)

            histories.append({
                'basket_id': basket_id,
                'customer_id': customer_id,
                'num_prior_baskets': len(history_products),
                'history_basket_ids': [b[1] for b in prior_baskets],
                'history_weeks': history_weeks,
                'history_products': history_products
            })

        return histories

    def _create_history_dataframe(
        self,
        histories: List[Dict],
        metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create output dataframe with histories."""
        history_df = pd.DataFrame(histories)

        # Merge with metadata to get split info
        history_df = history_df.merge(
            metadata_df[['basket_id', 'split', 'week', 'is_cold_start', 'bucket']],
            on='basket_id',
            how='left'
        )

        return history_df

    def _print_history_stats(self, df: pd.DataFrame) -> None:
        """Print history statistics."""
        print("\n  History length statistics:")

        for split in ['train', 'validation', 'test']:
            split_df = df[df['split'] == split]
            if len(split_df) == 0:
                continue

            lengths = split_df['num_prior_baskets']
            print(f"\n  {split}:")
            print(f"    - Baskets: {len(split_df):,}")
            print(f"    - Mean history length: {lengths.mean():.1f} baskets")
            print(f"    - Median: {lengths.median():.0f}")
            print(f"    - Max: {lengths.max():.0f}")
            print(f"    - Zero history: {(lengths == 0).sum():,} ({(lengths == 0).mean()*100:.1f}%)")

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save customer histories to parquet."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert lists to strings for parquet compatibility
        df_save = df.copy()
        df_save['history_basket_ids'] = df_save['history_basket_ids'].apply(
            lambda x: ','.join(map(str, x)) if x else ''
        )
        df_save['history_weeks'] = df_save['history_weeks'].apply(
            lambda x: ','.join(map(str, x)) if x else ''
        )
        # Products are nested lists - serialize as JSON
        import json
        df_save['history_products'] = df_save['history_products'].apply(
            lambda x: json.dumps(x) if x else '[]'
        )

        df_save.to_parquet(output_path, index=False)
        print(f"\nSaved to: {output_path}")
        print(f"  - Size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    """Run customer history extraction."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    metadata_path = project_root / 'data' / 'prepared' / 'temporal_metadata.parquet'
    output_path = project_root / 'data' / 'prepared' / 'customer_histories.parquet'

    # Load transactions
    print(f"Loading transactions from {raw_data_path}...")
    transactions_df = pd.read_csv(
        raw_data_path,
        usecols=['BASKET_ID', 'CUST_CODE', 'SHOP_WEEK', 'PROD_CODE']
    )
    print(f"  - Loaded {len(transactions_df):,} transactions")

    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    metadata_df = pd.read_parquet(metadata_path)
    print(f"  - Loaded {len(metadata_df):,} baskets")

    # Extract histories
    extractor = CustomerHistoryExtractor()
    history_df = extractor.run(transactions_df, metadata_df)

    # Save
    extractor.save(history_df, output_path)

    return history_df


if __name__ == '__main__':
    main()
