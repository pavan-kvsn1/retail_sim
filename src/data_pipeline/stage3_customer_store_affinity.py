"""
Stage 3: Customer-Store Affinity Computation
=============================================
Quantifies customer-store relationships through multiple metrics:
1. Primary Store Identification
2. Store Loyalty Score (Herfindahl Index)
3. Store Switching Rate
4. Regional Diversity

Output: customer_store_affinity.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class CustomerStoreAffinityPipeline:
    """
    Computes customer-store affinity metrics.

    Metrics:
    - Primary Store: Store with plurality of visits
    - Loyalty Score (HHI): Concentration of visits across stores
    - Switching Rate: Frequency of visiting new stores
    - Regional Diversity: Number of distinct regions shopped
    """

    def __init__(self, switching_lookback_weeks: int = 4):
        """
        Parameters
        ----------
        switching_lookback_weeks : int
            Number of weeks to look back for switching rate calculation
        """
        self.switching_lookback_weeks = switching_lookback_weeks

    def run(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute customer-store affinity computation.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions with columns: CUST_CODE, STORE_CODE,
            STORE_REGION, SHOP_WEEK

        Returns
        -------
        pd.DataFrame
            Customer affinity features
        """
        print("Stage 3: Customer-Store Affinity Computation")
        print("=" * 50)

        # Filter to customers with valid IDs
        df = transactions_df[transactions_df['CUST_CODE'].notna()].copy()
        print(f"\nCustomers with valid IDs: {df['CUST_CODE'].nunique():,}")

        # Step 1: Aggregate visits by customer-store
        print("\nStep 1: Aggregating customer-store visits...")
        customer_store_visits = self._aggregate_visits(df)

        # Step 2: Compute primary store
        print("\nStep 2: Computing primary stores...")
        primary_stores = self._compute_primary_store(customer_store_visits)
        print(f"  - Primary stores identified for {len(primary_stores):,} customers")

        # Step 3: Compute loyalty score (HHI)
        print("\nStep 3: Computing loyalty scores (Herfindahl Index)...")
        loyalty_scores = self._compute_loyalty_score(customer_store_visits)
        print(f"  - Mean loyalty score: {loyalty_scores['loyalty_score'].mean():.3f}")

        # Step 4: Compute switching rate
        print("\nStep 4: Computing switching rates...")
        switching_rates = self._compute_switching_rate(df)
        print(f"  - Mean switching rate: {switching_rates['switching_rate'].mean():.3f}")

        # Step 5: Compute regional diversity
        print("\nStep 5: Computing regional diversity...")
        regional_diversity = self._compute_regional_diversity(df)
        print(f"  - Mean regions visited: {regional_diversity['region_diversity'].mean():.1f}")

        # Step 6: Compute additional metrics
        print("\nStep 6: Computing additional metrics...")
        additional_metrics = self._compute_additional_metrics(customer_store_visits)

        # Merge all features
        print("\nMerging all affinity features...")
        affinity_df = self._merge_features(
            primary_stores,
            loyalty_scores,
            switching_rates,
            regional_diversity,
            additional_metrics
        )

        print("\n" + "=" * 50)
        print("Customer-Store Affinity Complete!")
        print(f"  - Total customers: {len(affinity_df):,}")
        print(f"  - Features: {len(affinity_df.columns)}")

        return affinity_df

    def _aggregate_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate visit counts by customer-store."""
        visits = df.groupby(['CUST_CODE', 'STORE_CODE']).agg(
            visit_count=('BASKET_ID', 'nunique'),
            total_spend=('SPEND', 'sum'),
            total_items=('QUANTITY', 'sum'),
            first_visit_week=('SHOP_WEEK', 'min'),
            last_visit_week=('SHOP_WEEK', 'max')
        ).reset_index()

        return visits

    def _compute_primary_store(
        self,
        customer_store_visits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Identify primary store (plurality of visits) for each customer.
        """
        # Get store with max visits for each customer
        idx = customer_store_visits.groupby('CUST_CODE')['visit_count'].idxmax()
        primary = customer_store_visits.loc[idx, ['CUST_CODE', 'STORE_CODE', 'visit_count']].copy()
        primary.columns = ['customer_id', 'primary_store', 'primary_store_visits']

        # Compute primary store share
        total_visits = customer_store_visits.groupby('CUST_CODE')['visit_count'].sum()
        primary['primary_store_share'] = (
            primary.set_index('customer_id')['primary_store_visits'] /
            total_visits
        ).values

        return primary

    def _compute_loyalty_score(
        self,
        customer_store_visits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute Herfindahl-Hirschman Index (HHI) for store loyalty.

        HHI = Σ (visit_share_s)² for all stores s
        - HHI = 1.0: Perfect loyalty (100% to one store)
        - HHI = 0.1: Low loyalty (spread across 10 stores)
        """
        # Compute visit shares
        total_visits = customer_store_visits.groupby('CUST_CODE')['visit_count'].transform('sum')
        customer_store_visits = customer_store_visits.copy()
        customer_store_visits['visit_share'] = customer_store_visits['visit_count'] / total_visits

        # Compute HHI
        customer_store_visits['visit_share_sq'] = customer_store_visits['visit_share'] ** 2

        hhi = customer_store_visits.groupby('CUST_CODE')['visit_share_sq'].sum().reset_index()
        hhi.columns = ['customer_id', 'loyalty_score']

        return hhi

    def _compute_switching_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute store switching rate (vectorized).

        Switching = visiting a store not visited in prior N weeks.
        Rate = # switching weeks / total weeks
        """
        # Sort by customer and week
        df = df.sort_values(['CUST_CODE', 'SHOP_WEEK'])

        # Get unique customer-week-store combinations
        customer_week_stores = df.groupby(['CUST_CODE', 'SHOP_WEEK'])['STORE_CODE'].apply(set).reset_index()
        customer_week_stores.columns = ['customer_id', 'week', 'stores']

        # Create a helper function to compute switching for a group
        def compute_customer_switching(group):
            if len(group) < 2:
                return pd.Series({
                    'switching_rate': 0.0,
                    'switch_count': 0,
                    'total_weeks': len(group)
                })
            
            # Sort by week
            group = group.sort_values('week').reset_index(drop=True)
            stores_list = group['stores'].values
            
            # Vectorized approach: build rolling window of prior stores
            switch_count = 0
            total_weeks = len(group) - 1
            
            for i in range(1, len(stores_list)):
                # Get stores from prior weeks (within lookback window)
                start_idx = max(0, i - self.switching_lookback_weeks)
                prior_stores = set()
                for j in range(start_idx, i):
                    prior_stores.update(stores_list[j])
                
                # Check if any current store is novel
                if stores_list[i] - prior_stores:
                    switch_count += 1
            
            switching_rate = switch_count / total_weeks if total_weeks > 0 else 0
            
            return pd.Series({
                'switching_rate': switching_rate,
                'switch_count': switch_count,
                'total_weeks': total_weeks
            })
        
        # Apply groupby operation (vectorized at the customer level)
        result = customer_week_stores.groupby('customer_id').apply(compute_customer_switching).reset_index()
        
        return result

    def _compute_regional_diversity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute number of distinct regions shopped by each customer.
        """
        regional = df.groupby('CUST_CODE').agg(
            region_diversity=('STORE_REGION', 'nunique'),
            regions_list=('STORE_REGION', lambda x: list(x.unique()))
        ).reset_index()

        regional.columns = ['customer_id', 'region_diversity', 'regions_visited']

        return regional

    def _compute_additional_metrics(
        self,
        customer_store_visits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute additional affinity metrics:
        - Total stores visited
        - Visit concentration (% to top 3 stores)
        - Tenure (weeks from first to last visit)
        """
        # Stores visited
        stores_visited = customer_store_visits.groupby('CUST_CODE')['STORE_CODE'].nunique().reset_index()
        stores_visited.columns = ['customer_id', 'total_stores_visited']

        # Top 3 store concentration
        def top3_concentration(group):
            total = group['visit_count'].sum()
            top3 = group.nlargest(3, 'visit_count')['visit_count'].sum()
            return top3 / total if total > 0 else 1.0

        top3_conc = customer_store_visits.groupby('CUST_CODE').apply(top3_concentration).reset_index()
        top3_conc.columns = ['customer_id', 'visit_concentration']

        # Tenure
        tenure = customer_store_visits.groupby('CUST_CODE').agg(
            first_week=('first_visit_week', 'min'),
            last_week=('last_visit_week', 'max')
        ).reset_index()
        tenure['tenure_weeks'] = tenure['last_week'] - tenure['first_week'] + 1
        tenure = tenure[['CUST_CODE', 'tenure_weeks']]
        tenure.columns = ['customer_id', 'tenure_weeks']

        # Total spend
        total_spend = customer_store_visits.groupby('CUST_CODE')['total_spend'].sum().reset_index()
        total_spend.columns = ['customer_id', 'total_spend']

        # Merge
        additional = stores_visited.merge(top3_conc, on='customer_id')
        additional = additional.merge(tenure, on='customer_id')
        additional = additional.merge(total_spend, on='customer_id')

        return additional

    def _merge_features(
        self,
        primary_stores: pd.DataFrame,
        loyalty_scores: pd.DataFrame,
        switching_rates: pd.DataFrame,
        regional_diversity: pd.DataFrame,
        additional_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all feature dataframes into single output."""
        # Start with primary stores
        result = primary_stores.copy()

        # Merge loyalty scores
        result = result.merge(loyalty_scores, on='customer_id', how='left')

        # Merge switching rates
        switching_subset = switching_rates[['customer_id', 'switching_rate', 'switch_count']]
        result = result.merge(switching_subset, on='customer_id', how='left')

        # Merge regional diversity
        regional_subset = regional_diversity[['customer_id', 'region_diversity']]
        result = result.merge(regional_subset, on='customer_id', how='left')

        # Merge additional metrics
        result = result.merge(additional_metrics, on='customer_id', how='left')

        # Fill NaN values
        result = result.fillna({
            'switching_rate': 0,
            'switch_count': 0,
            'region_diversity': 1,
            'loyalty_score': 1.0
        })

        # Reorder columns
        column_order = [
            'customer_id',
            'primary_store',
            'loyalty_score',
            'switching_rate',
            'region_diversity',
            'total_stores_visited',
            'visit_concentration',
            'primary_store_share',
            'tenure_weeks',
            'total_spend'
        ]

        # Only keep columns that exist
        column_order = [c for c in column_order if c in result.columns]
        result = result[column_order]

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
            'CUST_CODE', 'STORE_CODE', 'STORE_REGION',
            'SHOP_WEEK', 'BASKET_ID', 'SPEND', 'QUANTITY'
        ]
    )

    print(f"  - Loaded {len(df):,} transactions")
    print(f"  - Customers: {df['CUST_CODE'].nunique():,}")
    print(f"  - Stores: {df['STORE_CODE'].nunique():,}")
    print(f"  - Regions: {df['STORE_REGION'].nunique():,}")

    return df


def main():
    """Run customer-store affinity computation on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    output_path = project_root / 'data' / 'processed' / 'customer_store_affinity.parquet'

    # Load sample
    transactions_df = load_transactions_sample(str(raw_data_path), nrows=10000)

    # Run pipeline
    pipeline = CustomerStoreAffinityPipeline()
    affinity_df = pipeline.run(transactions_df)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    affinity_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Display sample and statistics
    print("\nSample output:")
    print(affinity_df.head(10).to_string())

    print("\nFeature statistics:")
    print(affinity_df.describe().to_string())

    return affinity_df


if __name__ == '__main__':
    main()
