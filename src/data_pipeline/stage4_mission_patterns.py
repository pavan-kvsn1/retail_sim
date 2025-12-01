"""
Stage 4: Historical Mission Pattern Extraction
===============================================
Extracts customer mission patterns from transaction history:
1. Mission Type Distribution (Top-up, Full Shop, etc.)
2. Mission Focus Distribution (Fresh, Grocery, Mixed)
3. Price Sensitivity Tendency
4. Basket Size Tendency

Output: customer_mission_patterns.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import entropy
import json
import warnings

warnings.filterwarnings('ignore')


class MissionPatternPipeline:
    """
    Extracts historical mission patterns from customer behavior.

    Patterns:
    - Mission Type: Top-up vs Full Shop distribution
    - Mission Focus: Fresh, Grocery, Mixed distribution
    - Price Sensitivity: Average sensitivity tendency
    - Basket Size: Size tendency and variability
    """

    # Encoding mappings for categorical variables
    PRICE_SENSITIVITY_MAP = {
        'LA': 0.0,   # Low sensitivity (buys at regular price)
        'MM': 0.5,   # Medium
        'UM': 1.0    # High sensitivity (price-driven)
    }

    BASKET_SIZE_MAP = {
        'S': 0.33,   # Small
        'M': 0.67,   # Medium
        'L': 1.0     # Large
    }

    def __init__(self, min_trips: int = 3):
        """
        Parameters
        ----------
        min_trips : int
            Minimum number of trips to compute reliable patterns
        """
        self.min_trips = min_trips

    def run(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute mission pattern extraction.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions with columns: CUST_CODE, BASKET_ID,
            BASKET_TYPE, BASKET_DOMINANT_MISSION, BASKET_PRICE_SENSITIVITY,
            BASKET_SIZE

        Returns
        -------
        pd.DataFrame
            Customer mission pattern features
        """
        print("Stage 4: Historical Mission Pattern Extraction")
        print("=" * 50)

        # Filter to customers with valid IDs
        df = transactions_df[transactions_df['CUST_CODE'].notna()].copy()
        print(f"\nCustomers with valid IDs: {df['CUST_CODE'].nunique():,}")

        # Aggregate to basket level (one row per basket)
        print("\nAggregating to basket level...")
        baskets_df = self._aggregate_to_baskets(df)
        print(f"  - Total baskets: {len(baskets_df):,}")

        # Filter customers with minimum trips
        basket_counts = baskets_df.groupby('CUST_CODE').size()
        valid_customers = basket_counts[basket_counts >= self.min_trips].index
        baskets_df = baskets_df[baskets_df['CUST_CODE'].isin(valid_customers)]
        print(f"  - Customers with >= {self.min_trips} trips: {len(valid_customers):,}")

        # Step 1: Mission Type Distribution
        print("\nStep 1: Computing mission type distributions...")
        mission_type = self._compute_mission_type_distribution(baskets_df)

        # Step 2: Mission Focus Distribution
        print("\nStep 2: Computing mission focus distributions...")
        mission_focus = self._compute_mission_focus_distribution(baskets_df)

        # Step 3: Price Sensitivity Tendency
        print("\nStep 3: Computing price sensitivity tendencies...")
        price_sensitivity = self._compute_price_sensitivity(baskets_df)

        # Step 4: Basket Size Tendency
        print("\nStep 4: Computing basket size tendencies...")
        basket_size = self._compute_basket_size_tendency(baskets_df)

        # Merge all features
        print("\nMerging all mission pattern features...")
        patterns_df = self._merge_features(
            mission_type,
            mission_focus,
            price_sensitivity,
            basket_size
        )

        print("\n" + "=" * 50)
        print("Mission Pattern Extraction Complete!")
        print(f"  - Total customers: {len(patterns_df):,}")
        print(f"  - Features: {len(patterns_df.columns)}")

        return patterns_df

    def _aggregate_to_baskets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transactions to basket level.
        Each basket should have one mission type, focus, sensitivity, size.
        """
        baskets = df.groupby(['CUST_CODE', 'BASKET_ID']).agg({
            'BASKET_TYPE': 'first',
            'BASKET_DOMINANT_MISSION': 'first',
            'BASKET_PRICE_SENSITIVITY': 'first',
            'BASKET_SIZE': 'first',
            'SHOP_WEEK': 'first',
            'QUANTITY': 'sum',
            'SPEND': 'sum'
        }).reset_index()

        return baskets

    def _compute_mission_type_distribution(
        self,
        baskets_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute mission type (Top-up, Full Shop, etc.) distribution per customer.
        """
        # Get mission type counts per customer
        mission_counts = baskets_df.groupby(
            ['CUST_CODE', 'BASKET_TYPE']
        ).size().unstack(fill_value=0)

        # Normalize to probabilities
        mission_probs = mission_counts.div(mission_counts.sum(axis=1), axis=0)

        # Get dominant mission type
        dominant_type = mission_counts.idxmax(axis=1)

        # Compute entropy (variability)
        mission_entropy = mission_probs.apply(
            lambda x: entropy(x[x > 0]) if (x > 0).any() else 0,
            axis=1
        )

        # Create result dataframe
        result = pd.DataFrame({
            'customer_id': mission_counts.index,
            'dominant_mission_type': dominant_type.values,
            'mission_type_entropy': mission_entropy.values
        })

        # Add distribution as JSON string
        result['mission_type_dist'] = mission_probs.apply(
            lambda x: json.dumps(x.to_dict()), axis=1
        ).values

        # Add individual type probabilities
        for col in mission_probs.columns:
            safe_col = col.replace(' ', '_').lower() if col else 'unknown'
            result[f'p_mission_{safe_col}'] = mission_probs[col].values

        return result

    def _compute_mission_focus_distribution(
        self,
        baskets_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute mission focus (Fresh, Grocery, Mixed) distribution per customer.
        """
        # Get focus counts per customer
        focus_counts = baskets_df.groupby(
            ['CUST_CODE', 'BASKET_DOMINANT_MISSION']
        ).size().unstack(fill_value=0)

        # Normalize to probabilities
        focus_probs = focus_counts.div(focus_counts.sum(axis=1), axis=0)

        # Get dominant focus
        dominant_focus = focus_counts.idxmax(axis=1)

        # Compute entropy
        focus_entropy = focus_probs.apply(
            lambda x: entropy(x[x > 0]) if (x > 0).any() else 0,
            axis=1
        )

        # Create result dataframe
        result = pd.DataFrame({
            'customer_id': focus_counts.index,
            'dominant_focus': dominant_focus.values,
            'focus_entropy': focus_entropy.values
        })

        # Add distribution as JSON string
        result['mission_focus_dist'] = focus_probs.apply(
            lambda x: json.dumps(x.to_dict()), axis=1
        ).values

        # Add individual focus probabilities
        for col in focus_probs.columns:
            safe_col = col.replace(' ', '_').lower() if col else 'unknown'
            result[f'p_focus_{safe_col}'] = focus_probs[col].values

        return result

    def _compute_price_sensitivity(
        self,
        baskets_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute price sensitivity tendency per customer.

        Maps LA=0 (low), MM=0.5 (medium), UM=1.0 (high sensitivity)
        """
        # Map sensitivity to numeric
        baskets_df = baskets_df.copy()
        baskets_df['sensitivity_numeric'] = baskets_df['BASKET_PRICE_SENSITIVITY'].map(
            self.PRICE_SENSITIVITY_MAP
        )

        # Compute mean and std per customer
        sensitivity_stats = baskets_df.groupby('CUST_CODE').agg(
            mean_price_sensitivity=('sensitivity_numeric', 'mean'),
            sensitivity_volatility=('sensitivity_numeric', 'std')
        ).reset_index()

        sensitivity_stats.columns = ['customer_id', 'mean_price_sensitivity', 'sensitivity_volatility']

        # Fill NaN volatility (single observation) with 0
        sensitivity_stats['sensitivity_volatility'] = sensitivity_stats['sensitivity_volatility'].fillna(0)

        # Get mode (most common sensitivity level)
        mode_sensitivity = baskets_df.groupby('CUST_CODE')['BASKET_PRICE_SENSITIVITY'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'MM'
        ).reset_index()
        mode_sensitivity.columns = ['customer_id', 'typical_sensitivity']

        result = sensitivity_stats.merge(mode_sensitivity, on='customer_id')

        return result

    def _compute_basket_size_tendency(
        self,
        baskets_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute basket size tendency per customer.

        Maps S=0.33, M=0.67, L=1.0
        """
        # Map size to numeric
        baskets_df = baskets_df.copy()
        baskets_df['size_numeric'] = baskets_df['BASKET_SIZE'].map(self.BASKET_SIZE_MAP)

        # Compute mean and variance per customer
        size_stats = baskets_df.groupby('CUST_CODE').agg(
            mean_basket_size=('size_numeric', 'mean'),
            basket_size_variance=('size_numeric', 'var'),
            total_trips=('BASKET_ID', 'nunique'),
            avg_items_per_trip=('QUANTITY', 'mean'),
            avg_spend_per_trip=('SPEND', 'mean')
        ).reset_index()

        size_stats.columns = [
            'customer_id', 'mean_basket_size', 'basket_size_variance',
            'total_trips', 'avg_items_per_trip', 'avg_spend_per_trip'
        ]

        # Fill NaN variance with 0
        size_stats['basket_size_variance'] = size_stats['basket_size_variance'].fillna(0)

        # Get size distribution
        size_counts = baskets_df.groupby(
            ['CUST_CODE', 'BASKET_SIZE']
        ).size().unstack(fill_value=0)

        size_probs = size_counts.div(size_counts.sum(axis=1), axis=0)

        # Add size probabilities
        for col in ['S', 'M', 'L']:
            if col in size_probs.columns:
                size_stats[f'p_size_{col.lower()}'] = size_stats['customer_id'].map(
                    dict(zip(size_probs.index, size_probs[col]))
                ).fillna(0)
            else:
                size_stats[f'p_size_{col.lower()}'] = 0

        return size_stats

    def _compute_mission_consistency(
        self,
        mission_type: pd.DataFrame,
        mission_focus: pd.DataFrame,
        price_sensitivity: pd.DataFrame,
        basket_size: pd.DataFrame
    ) -> pd.Series:
        """
        Compute overall mission consistency score.

        High consistency = customer has predictable patterns
        Low consistency = customer behavior is highly variable
        """
        # Combine entropies (lower entropy = more consistent)
        type_entropy = mission_type.set_index('customer_id')['mission_type_entropy']
        focus_entropy = mission_focus.set_index('customer_id')['focus_entropy']
        sensitivity_vol = price_sensitivity.set_index('customer_id')['sensitivity_volatility']
        size_var = basket_size.set_index('customer_id')['basket_size_variance']

        # Normalize each component to [0, 1]
        type_norm = type_entropy / (type_entropy.max() + 1e-6)
        focus_norm = focus_entropy / (focus_entropy.max() + 1e-6)
        sens_norm = sensitivity_vol / (sensitivity_vol.max() + 1e-6)
        size_norm = size_var / (size_var.max() + 1e-6)

        # Consistency = 1 - average normalized variability
        consistency = 1 - (type_norm + focus_norm + sens_norm + size_norm) / 4

        return consistency

    def _merge_features(
        self,
        mission_type: pd.DataFrame,
        mission_focus: pd.DataFrame,
        price_sensitivity: pd.DataFrame,
        basket_size: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all mission pattern features into single output."""
        # Compute consistency
        consistency = self._compute_mission_consistency(
            mission_type, mission_focus, price_sensitivity, basket_size
        )

        # Start with mission type
        result = mission_type.copy()

        # Merge mission focus
        focus_cols = ['customer_id', 'dominant_focus', 'focus_entropy', 'mission_focus_dist']
        focus_cols += [c for c in mission_focus.columns if c.startswith('p_focus_')]
        result = result.merge(mission_focus[focus_cols], on='customer_id', how='left')

        # Merge price sensitivity
        result = result.merge(price_sensitivity, on='customer_id', how='left')

        # Merge basket size
        result = result.merge(basket_size, on='customer_id', how='left')

        # Add consistency
        result['mission_consistency'] = result['customer_id'].map(consistency)

        # Fill NaN values
        result = result.fillna({
            'mission_type_entropy': 0,
            'focus_entropy': 0,
            'sensitivity_volatility': 0,
            'basket_size_variance': 0,
            'mission_consistency': 0.5
        })

        return result


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
            'CUST_CODE', 'BASKET_ID', 'BASKET_TYPE',
            'BASKET_DOMINANT_MISSION', 'BASKET_PRICE_SENSITIVITY',
            'BASKET_SIZE', 'SHOP_WEEK', 'QUANTITY', 'SPEND'
        ]
    )

    print(f"  - Loaded {len(df):,} transactions")
    print(f"  - Customers: {df['CUST_CODE'].nunique():,}")
    print(f"  - Baskets: {df['BASKET_ID'].nunique():,}")
    print(f"  - Mission types: {df['BASKET_TYPE'].unique()}")
    print(f"  - Mission focuses: {df['BASKET_DOMINANT_MISSION'].unique()}")

    return df


def main():
    """Run mission pattern extraction on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    output_path = project_root / 'data' / 'processed' / 'customer_mission_patterns.parquet'

    # Load sample
    transactions_df = load_transactions_sample(str(raw_data_path), nrows=10000)

    # Run pipeline
    pipeline = MissionPatternPipeline(min_trips=2)  # Lower threshold for sample
    patterns_df = pipeline.run(transactions_df)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patterns_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Display sample and statistics
    print("\nSample output (selected columns):")
    display_cols = [
        'customer_id', 'dominant_mission_type', 'dominant_focus',
        'mean_price_sensitivity', 'mean_basket_size', 'mission_consistency'
    ]
    display_cols = [c for c in display_cols if c in patterns_df.columns]
    print(patterns_df[display_cols].head(10).to_string())

    print("\nFeature statistics:")
    numeric_cols = patterns_df.select_dtypes(include=[np.number]).columns
    print(patterns_df[numeric_cols].describe().to_string())

    return patterns_df


if __name__ == '__main__':
    main()
