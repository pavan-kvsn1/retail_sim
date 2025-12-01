"""
Layer 5: Store Context Features
================================
Generates store-level representations combining:
- Store identity embeddings
- Format and region features
- Operational features (derived from transaction patterns)

Output: store_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle
import warnings

warnings.filterwarnings('ignore')


class StoreContextEncoder:
    """
    Encodes store context for model input.

    Store Context [96d]:
    - Store identity: [32d]
    - Format + region: [32d]
    - Operational features: [32d]
    """

    def __init__(
        self,
        identity_dim: int = 32,
        format_region_dim: int = 32,
        operational_dim: int = 32
    ):
        """
        Parameters
        ----------
        identity_dim : int
            Dimension for store identity embedding
        format_region_dim : int
            Dimension for format and region features
        operational_dim : int
            Dimension for operational features
        """
        self.identity_dim = identity_dim
        self.format_region_dim = format_region_dim
        self.operational_dim = operational_dim
        self.output_dim = identity_dim + format_region_dim + operational_dim

        # Storage
        self.store_features = {}

    def run(
        self,
        transactions_df: pd.DataFrame,
        customer_affinity: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate store context features.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transaction data with store information
        customer_affinity : pd.DataFrame, optional
            Customer-store affinity metrics

        Returns
        -------
        pd.DataFrame
            Store features with 96d encoding per store
        """
        print("Layer 5: Store Context Features")
        print("=" * 50)

        # Step 1: Extract store metadata
        print("\nStep 1: Extracting store metadata...")
        store_meta = self._extract_store_metadata(transactions_df)
        print(f"  - Stores: {len(store_meta):,}")

        # Step 2: Generate identity embeddings
        print("\nStep 2: Generating store identity embeddings...")
        identity_features = self._generate_identity_embeddings(store_meta)

        # Step 3: Encode format and region
        print("\nStep 3: Encoding format and region features...")
        format_region_features = self._encode_format_region(store_meta)

        # Step 4: Compute operational features
        print("\nStep 4: Computing operational features...")
        operational_features = self._compute_operational_features(
            transactions_df, store_meta, customer_affinity
        )

        # Step 5: Combine all features
        print("\nStep 5: Combining all store features...")
        store_features = self._combine_features(
            store_meta,
            identity_features,
            format_region_features,
            operational_features
        )

        self.store_features = store_features

        print("\n" + "=" * 50)
        print("Store Context Features Complete!")
        print(f"  - Stores: {len(store_features):,}")
        print(f"  - Feature dimension: {self.output_dim}d")

        return store_features

    def _extract_store_metadata(
        self,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract unique store attributes from transactions."""
        store_meta = transactions_df.groupby('STORE_CODE').agg({
            'STORE_FORMAT': 'first',
            'STORE_REGION': 'first',
            'BASKET_ID': 'nunique',
            'CUST_CODE': 'nunique',
            'SPEND': 'sum',
            'QUANTITY': 'sum',
            'SHOP_WEEK': ['min', 'max', 'nunique']
        }).reset_index()

        # Flatten column names
        store_meta.columns = [
            'store_id', 'format', 'region',
            'total_baskets', 'unique_customers', 'total_spend', 'total_quantity',
            'first_week', 'last_week', 'active_weeks'
        ]

        return store_meta

    def _generate_identity_embeddings(
        self,
        store_meta: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Generate store identity embeddings.

        Uses hash-based embedding for consistency.
        """
        identity_embeddings = {}

        for store_id in store_meta['store_id']:
            # Hash-based embedding for consistency
            store_hash = hash(store_id) % (2**31)
            np.random.seed(store_hash)
            embedding = np.random.randn(self.identity_dim) * 0.1

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            identity_embeddings[store_id] = embedding

        return identity_embeddings

    def _encode_format_region(
        self,
        store_meta: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Encode store format and region as embeddings.
        """
        # Get unique formats and regions
        formats = store_meta['format'].dropna().unique()
        regions = store_meta['region'].dropna().unique()

        # Create embedding lookups
        np.random.seed(44)
        format_embeds = {f: np.random.randn(self.format_region_dim // 2) * 0.1
                        for f in formats}
        region_embeds = {r: np.random.randn(self.format_region_dim // 2) * 0.1
                        for r in regions}

        # Default embeddings for missing values
        default_format = np.zeros(self.format_region_dim // 2)
        default_region = np.zeros(self.format_region_dim // 2)

        format_region_embeddings = {}

        for _, row in store_meta.iterrows():
            store_id = row['store_id']
            store_format = row['format']
            store_region = row['region']

            format_embed = format_embeds.get(store_format, default_format)
            region_embed = region_embeds.get(store_region, default_region)

            combined = np.concatenate([format_embed, region_embed])
            format_region_embeddings[store_id] = combined

        return format_region_embeddings

    def _compute_operational_features(
        self,
        transactions_df: pd.DataFrame,
        store_meta: pd.DataFrame,
        customer_affinity: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute operational features from transaction patterns.

        Features:
        - Volume metrics (baskets/week, avg basket size)
        - Customer metrics (unique customers, loyalty)
        - Temporal patterns (peak hours, weekend share)
        - Category mix (fresh vs grocery ratio)
        """
        operational_features = {}

        # Compute store-level statistics
        store_stats = transactions_df.groupby('STORE_CODE').agg({
            'BASKET_ID': 'nunique',
            'CUST_CODE': 'nunique',
            'SPEND': ['sum', 'mean', 'std'],
            'QUANTITY': ['sum', 'mean'],
            'SHOP_HOUR': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12,
            'SHOP_WEEKDAY': lambda x: (x.isin([6, 7])).mean(),  # Weekend share
            'BASKET_SIZE': lambda x: (x == 'L').mean(),  # Large basket share
            'BASKET_DOMINANT_MISSION': lambda x: (x == 'Fresh').mean()  # Fresh focus
        }).reset_index()

        store_stats.columns = [
            'store_id',
            'num_baskets', 'num_customers',
            'total_spend', 'avg_spend', 'std_spend',
            'total_quantity', 'avg_quantity',
            'peak_hour', 'weekend_share',
            'large_basket_share', 'fresh_focus'
        ]

        # Merge with store_meta for active weeks
        store_stats = store_stats.merge(
            store_meta[['store_id', 'active_weeks']],
            on='store_id',
            how='left'
        )

        # Compute derived features
        store_stats['baskets_per_week'] = (
            store_stats['num_baskets'] / store_stats['active_weeks'].clip(1)
        )
        store_stats['customers_per_week'] = (
            store_stats['num_customers'] / store_stats['active_weeks'].clip(1)
        )
        store_stats['spend_cv'] = (
            store_stats['std_spend'] / store_stats['avg_spend'].clip(0.01)
        )

        # Add loyalty metrics if available
        if customer_affinity is not None:
            # Compute store-level loyalty
            store_loyalty = customer_affinity.groupby('primary_store').agg(
                mean_loyalty=('loyalty_score', 'mean'),
                loyal_customers=('loyalty_score', lambda x: (x > 0.8).sum())
            ).reset_index()
            store_loyalty.columns = ['store_id', 'mean_loyalty', 'loyal_customers']

            store_stats = store_stats.merge(store_loyalty, on='store_id', how='left')
            store_stats['mean_loyalty'] = store_stats['mean_loyalty'].fillna(0.5)
            store_stats['loyal_customers'] = store_stats['loyal_customers'].fillna(0)
        else:
            store_stats['mean_loyalty'] = 0.5
            store_stats['loyal_customers'] = 0

        # Normalize features
        numeric_cols = [
            'baskets_per_week', 'customers_per_week', 'avg_spend', 'spend_cv',
            'avg_quantity', 'peak_hour', 'weekend_share',
            'large_basket_share', 'fresh_focus', 'mean_loyalty'
        ]

        for col in numeric_cols:
            if col in store_stats.columns:
                values = store_stats[col].values
                # Min-max normalization
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    store_stats[f'{col}_norm'] = (values - min_val) / (max_val - min_val)
                else:
                    store_stats[f'{col}_norm'] = 0.5

        # Build feature vectors
        for _, row in store_stats.iterrows():
            store_id = row['store_id']

            # Core operational features
            features = [
                row.get('baskets_per_week_norm', 0.5),
                row.get('customers_per_week_norm', 0.5),
                row.get('avg_spend_norm', 0.5),
                row.get('spend_cv_norm', 0.5),
                row.get('avg_quantity_norm', 0.5),
                row.get('peak_hour_norm', 0.5),
                row.get('weekend_share_norm', 0.5),
                row.get('large_basket_share_norm', 0.5),
                row.get('fresh_focus_norm', 0.5),
                row.get('mean_loyalty_norm', 0.5)
            ]

            # Pad to operational_dim
            features = np.array(features)
            if len(features) < self.operational_dim:
                # Add interaction features
                interactions = []
                for i in range(len(features)):
                    for j in range(i+1, min(i+3, len(features))):
                        interactions.append(features[i] * features[j])
                features = np.concatenate([features, np.array(interactions)])

            features = features[:self.operational_dim]
            if len(features) < self.operational_dim:
                features = np.pad(features, (0, self.operational_dim - len(features)))

            operational_features[store_id] = features

        return operational_features

    def _combine_features(
        self,
        store_meta: pd.DataFrame,
        identity_features: Dict[str, np.ndarray],
        format_region_features: Dict[str, np.ndarray],
        operational_features: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Combine all store features into output DataFrame."""
        rows = []

        for _, row in store_meta.iterrows():
            store_id = row['store_id']

            # Get embeddings
            identity = identity_features.get(store_id, np.zeros(self.identity_dim))
            format_region = format_region_features.get(store_id, np.zeros(self.format_region_dim))
            operational = operational_features.get(store_id, np.zeros(self.operational_dim))

            # Concatenate
            full_features = np.concatenate([identity, format_region, operational])

            # Create row
            feature_dict = {
                'store_id': store_id,
                'format': row['format'],
                'region': row['region'],
                'total_baskets': row['total_baskets'],
                'unique_customers': row['unique_customers']
            }

            # Add feature columns
            for i, val in enumerate(identity):
                feature_dict[f'identity_{i}'] = val
            for i, val in enumerate(format_region):
                feature_dict[f'format_region_{i}'] = val
            for i, val in enumerate(operational):
                feature_dict[f'operational_{i}'] = val

            rows.append(feature_dict)

        return pd.DataFrame(rows)

    def get_store_embedding(self, store_id: str) -> Optional[np.ndarray]:
        """Get full embedding for a single store."""
        if self.store_features is None or self.store_features.empty:
            return None

        store_row = self.store_features[self.store_features['store_id'] == store_id]
        if store_row.empty:
            return None

        # Extract feature columns
        identity_cols = [c for c in self.store_features.columns if c.startswith('identity_')]
        fr_cols = [c for c in self.store_features.columns if c.startswith('format_region_')]
        op_cols = [c for c in self.store_features.columns if c.startswith('operational_')]

        embedding = np.concatenate([
            store_row[identity_cols].values.flatten(),
            store_row[fr_cols].values.flatten(),
            store_row[op_cols].values.flatten()
        ])

        return embedding


def main():
    """Run store context feature generation on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    transactions_path = project_root / 'raw_data' / 'transactions.csv'
    affinity_path = project_root / 'data' / 'processed' / 'customer_store_affinity.parquet'
    output_path = project_root / 'data' / 'features' / 'store_features.parquet'

    # Load data
    print("Loading data...")
    transactions_df = pd.read_csv(
        transactions_path,
        nrows=10000,
        usecols=[
            'STORE_CODE', 'STORE_FORMAT', 'STORE_REGION',
            'BASKET_ID', 'CUST_CODE', 'SPEND', 'QUANTITY',
            'SHOP_WEEK', 'SHOP_HOUR', 'SHOP_WEEKDAY',
            'BASKET_SIZE', 'BASKET_DOMINANT_MISSION'
        ]
    )
    print(f"  - Transactions: {len(transactions_df):,}")

    # Load customer affinity if available
    customer_affinity = None
    if affinity_path.exists():
        customer_affinity = pd.read_parquet(affinity_path)
        print(f"  - Customer affinity: {len(customer_affinity):,}")

    # Run encoder
    encoder = StoreContextEncoder()
    store_features = encoder.run(transactions_df, customer_affinity)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store_features.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Display sample
    print("\nSample output:")
    meta_cols = ['store_id', 'format', 'region', 'total_baskets', 'unique_customers']
    print(store_features[meta_cols].to_string())

    # Show feature dimensions
    identity_cols = [c for c in store_features.columns if c.startswith('identity_')]
    fr_cols = [c for c in store_features.columns if c.startswith('format_region_')]
    op_cols = [c for c in store_features.columns if c.startswith('operational_')]
    print(f"\nFeature dimensions:")
    print(f"  - Identity: {len(identity_cols)}d")
    print(f"  - Format/Region: {len(fr_cols)}d")
    print(f"  - Operational: {len(op_cols)}d")
    print(f"  - Total: {len(identity_cols) + len(fr_cols) + len(op_cols)}d")

    return store_features


if __name__ == '__main__':
    main()
