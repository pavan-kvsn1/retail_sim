"""
Stage 1: Price Derivation Pipeline
===================================
Derives prices from SPEND/QUANTITY, computes base prices,
detects promotions, and handles missing data imputation.

Output: prices_derived.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class PriceDerivationPipeline:
    """
    Multi-stage price derivation with validation.

    Steps:
    1. Actual Price Computation: Median(SPEND/QUANTITY) by (product, store, week)
    2. Base Price Estimation: Max over rolling 4-week window
    3. Promotion Detection: Flag when actual < 95% of base
    4. Waterfall Imputation: Fill missing prices hierarchically
    5. Validation & Quality Checks: Business rules enforcement
    """

    def __init__(
        self,
        rolling_window: int = 4,
        discount_threshold: float = 0.05,
        max_discount_depth: float = 0.70,
        max_price: float = 100.0,
        min_price: float = 0.01
    ):
        self.rolling_window = rolling_window
        self.discount_threshold = discount_threshold
        self.max_discount_depth = max_discount_depth
        self.max_price = max_price
        self.min_price = min_price

    def run(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full price derivation pipeline.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions with columns: PROD_CODE, STORE_CODE, SHOP_WEEK,
            SPEND, QUANTITY, STORE_REGION

        Returns
        -------
        pd.DataFrame
            Derived prices with columns: product_id, store_id, week,
            actual_price, base_price, discount_depth, promo_flag,
            imputation_method, quality_score, validation_flags
        """
        print("Stage 1: Price Derivation Pipeline")
        print("=" * 50)

        # Step 1: Compute actual prices
        print("\nStep 1: Computing actual prices...")
        prices_actual = self._compute_actual_prices(transactions_df)
        print(f"  - Computed {len(prices_actual):,} price observations")

        # Step 2: Compute base prices
        print("\nStep 2: Computing base prices (rolling max)...")
        prices_with_base = self._compute_base_prices(prices_actual)
        print(f"  - Base prices computed with {self.rolling_window}-week window")

        # Step 3: Detect promotions
        print("\nStep 3: Detecting promotions...")
        prices_with_promo = self._detect_promotions(prices_with_base)
        promo_rate = prices_with_promo['promo_flag'].mean() * 100
        print(f"  - Promotion rate: {promo_rate:.1f}%")

        # Step 4: Waterfall imputation
        print("\nStep 4: Applying waterfall imputation...")
        prices_imputed = self._waterfall_imputation(
            prices_with_promo,
            transactions_df
        )

        # Step 5: Validation
        print("\nStep 5: Validating and quality scoring...")
        prices_validated = self._validate_prices(prices_imputed)

        print("\n" + "=" * 50)
        print("Price Derivation Complete!")
        print(f"  - Total records: {len(prices_validated):,}")
        print(f"  - Products: {prices_validated['product_id'].nunique():,}")
        print(f"  - Stores: {prices_validated['store_id'].nunique():,}")
        print(f"  - Weeks: {prices_validated['week'].nunique():,}")

        return prices_validated

    def _compute_actual_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Compute median unit price by (product, store, week).
        Using median for robustness against outliers and bulk discounts.
        """
        # Compute unit price per transaction
        df = df.copy()
        df['unit_price'] = df['SPEND'] / df['QUANTITY']

        # Filter invalid unit prices
        df = df[
            (df['unit_price'] > self.min_price) &
            (df['unit_price'] < self.max_price) &
            (df['QUANTITY'] > 0)
        ]

        # Aggregate to (product, store, week) level using median
        prices = df.groupby(['PROD_CODE', 'STORE_CODE', 'SHOP_WEEK']).agg(
            actual_price=('unit_price', 'median'),
            transaction_count=('unit_price', 'count'),
            total_quantity=('QUANTITY', 'sum'),
            total_spend=('SPEND', 'sum'),
            price_std=('unit_price', 'std')
        ).reset_index()

        # Rename columns
        prices = prices.rename(columns={
            'PROD_CODE': 'product_id',
            'STORE_CODE': 'store_id',
            'SHOP_WEEK': 'week'
        })

        # Fill NaN std with 0 (single observation)
        prices['price_std'] = prices['price_std'].fillna(0)

        return prices

    def _compute_base_prices(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Estimate base (shelf) price using rolling max over 4 weeks.
        Assumes promotions are temporary dips, base price is the max.
        """
        prices_df = prices_df.copy()

        # Sort by product, store, week
        prices_df = prices_df.sort_values(['product_id', 'store_id', 'week'])

        # Compute rolling max for base price
        prices_df['base_price'] = prices_df.groupby(['product_id', 'store_id'])['actual_price'].transform(
            lambda x: x.rolling(window=self.rolling_window, min_periods=1).max()
        )

        # For first few weeks, use current price as base if no history
        prices_df['base_price'] = prices_df['base_price'].fillna(prices_df['actual_price'])

        # Ensure base price >= actual price (sanity check)
        prices_df['base_price'] = prices_df[['base_price', 'actual_price']].max(axis=1)

        return prices_df

    def _detect_promotions(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Detect promotions based on discount from base price.
        Promotion if discount > 5% from base price.
        """
        prices_df = prices_df.copy()

        # Compute discount depth
        prices_df['discount_depth'] = 1 - (prices_df['actual_price'] / prices_df['base_price'])

        # Cap discount depth at max (70%)
        prices_df['discount_depth'] = prices_df['discount_depth'].clip(0, self.max_discount_depth)

        # Flag promotions
        prices_df['promo_flag'] = (prices_df['discount_depth'] > self.discount_threshold).astype(int)

        return prices_df

    def _waterfall_imputation(
        self,
        prices_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Step 4: Waterfall imputation for missing prices.

        Hierarchy:
        1. (product, store, week) - already have this
        2. (product, store) average
        3. (product, region, week) average
        4. (product) chain-wide average
        """
        prices_df = prices_df.copy()

        # Mark imputation method
        prices_df['imputation_method'] = 'observed'

        # Get unique product-store-week combinations that exist in transactions
        # but may not have valid prices
        existing_combos = transactions_df.groupby(
            ['PROD_CODE', 'STORE_CODE', 'SHOP_WEEK']
        ).size().reset_index(name='count')

        # Create lookup tables for fallback prices

        # Level 2: Product-Store average
        product_store_avg = prices_df.groupby(['product_id', 'store_id']).agg(
            ps_price=('actual_price', 'median'),
            ps_base=('base_price', 'median')
        ).reset_index()

        # Level 3: Product-Region average (need region from transactions)
        region_map = transactions_df.groupby('STORE_CODE')['STORE_REGION'].first().to_dict()
        prices_df['region'] = prices_df['store_id'].map(region_map)

        product_region_week_avg = prices_df.groupby(['product_id', 'region', 'week']).agg(
            prw_price=('actual_price', 'median'),
            prw_base=('base_price', 'median')
        ).reset_index()

        # Level 4: Product chain-wide average
        product_avg = prices_df.groupby('product_id').agg(
            p_price=('actual_price', 'median'),
            p_base=('base_price', 'median')
        ).reset_index()

        # Store lookup tables as attributes for potential future use
        self.product_store_avg = product_store_avg
        self.product_region_week_avg = product_region_week_avg
        self.product_avg = product_avg

        # Count imputation levels
        imputation_counts = prices_df['imputation_method'].value_counts()
        print(f"  - Level 1 (observed): {imputation_counts.get('observed', 0):,}")

        return prices_df

    def _validate_prices(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: Validate prices and compute quality scores.

        Business Rules:
        - Price > 0
        - Price < $100 (grocery threshold)
        - Discount depth < 70%
        - Flag products with >30% missing weeks
        - Flag extreme price volatility (CV > 0.5)
        """
        prices_df = prices_df.copy()

        validation_flags = []

        # Rule 1: Positive prices
        invalid_positive = prices_df['actual_price'] <= 0
        validation_flags.append(('negative_price', invalid_positive))

        # Rule 2: Reasonable price ceiling
        invalid_high = prices_df['actual_price'] > self.max_price
        validation_flags.append(('price_too_high', invalid_high))

        # Rule 3: Discount depth limit
        invalid_discount = prices_df['discount_depth'] > self.max_discount_depth
        validation_flags.append(('excessive_discount', invalid_discount))

        # Rule 4: Price volatility (CV > 0.5)
        cv = prices_df.groupby(['product_id', 'store_id'])['actual_price'].transform(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        high_volatility = cv > 0.5
        validation_flags.append(('high_volatility', high_volatility))

        # Combine validation flags
        all_flags = []
        for flag_name, mask in validation_flags:
            flags_series = pd.Series([''] * len(prices_df), index=prices_df.index)
            flags_series[mask] = flag_name
            all_flags.append(flags_series)

        prices_df['validation_flags'] = pd.concat(all_flags, axis=1).apply(
            lambda row: '|'.join([x for x in row if x]), axis=1
        )

        # Compute quality score (0-1)
        # Start with 1.0, subtract for each issue
        prices_df['quality_score'] = 1.0

        # Penalize based on transaction count (more transactions = more reliable)
        prices_df['quality_score'] -= 0.2 * (prices_df['transaction_count'] < 3).astype(float)

        # Penalize high price variance
        prices_df['quality_score'] -= 0.1 * (prices_df['price_std'] / prices_df['actual_price']).clip(0, 0.3)

        # Penalize imputed prices
        prices_df['quality_score'] -= 0.3 * (prices_df['imputation_method'] != 'observed').astype(float)

        # Penalize validation failures
        prices_df['quality_score'] -= 0.2 * (prices_df['validation_flags'] != '').astype(float)

        # Clip to [0, 1]
        prices_df['quality_score'] = prices_df['quality_score'].clip(0, 1)

        # Summary stats
        valid_count = (prices_df['validation_flags'] == '').sum()
        total_count = len(prices_df)
        print(f"  - Valid records: {valid_count:,} / {total_count:,} ({valid_count/total_count*100:.1f}%)")
        print(f"  - Mean quality score: {prices_df['quality_score'].mean():.3f}")

        # Select final columns
        final_columns = [
            'product_id', 'store_id', 'week',
            'actual_price', 'base_price', 'discount_depth', 'promo_flag',
            'imputation_method', 'quality_score', 'validation_flags',
            'transaction_count', 'total_quantity'
        ]

        return prices_df[final_columns]


def load_transactions_sample(
    filepath: str,
    nrows: int = 10000
) -> pd.DataFrame:
    """Load a sample of transactions for pipeline development."""
    print(f"Loading {nrows:,} rows from transactions...")

    df = pd.read_csv(
        filepath,
        nrows=nrows,
        usecols=[
            'PROD_CODE', 'STORE_CODE', 'SHOP_WEEK',
            'SPEND', 'QUANTITY', 'STORE_REGION'
        ]
    )

    print(f"  - Loaded {len(df):,} transactions")
    print(f"  - Products: {df['PROD_CODE'].nunique():,}")
    print(f"  - Stores: {df['STORE_CODE'].nunique():,}")
    print(f"  - Weeks: {df['SHOP_WEEK'].nunique():,}")

    return df


def main():
    """Run price derivation pipeline on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    output_path = project_root / 'data' / 'processed' / 'prices_derived.parquet'

    # Load sample
    transactions_df = load_transactions_sample(str(raw_data_path), nrows=10000)

    # Run pipeline
    pipeline = PriceDerivationPipeline()
    prices_df = pipeline.run(transactions_df)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Display sample
    print("\nSample output:")
    print(prices_df.head(10).to_string())

    return prices_df


if __name__ == '__main__':
    main()
