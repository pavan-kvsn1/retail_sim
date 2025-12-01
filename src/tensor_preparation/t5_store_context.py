"""
T5: Store Context Tensor [96d]
===============================
Encodes store context with emphasis on attributes over store ID.

Components:
- Store format [24d]: LS, MS, SS embeddings
- Store region [24d]: Regional patterns
- Operational features [32d]: Size, traffic, competition, age
- Store identity [16d]: Residual store-specific effects
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class StoreContextEncoder:
    """
    Encodes store context tensor T5 [96d].

    Design prioritizes attributes (80d) over store_id (16d)
    to prevent overfitting with limited customers per store.
    """

    # Store format vocabulary
    FORMAT_VOCAB = ['LS', 'MS', 'SS']  # Large/Medium/Small Super

    # Region vocabulary
    REGION_VOCAB = ['E01', 'E02', 'W01', 'W02', 'S01', 'S02', 'N01', 'N02', 'C01', 'C02']

    def __init__(
        self,
        format_dim: int = 24,
        region_dim: int = 24,
        operational_dim: int = 32,
        identity_dim: int = 16
    ):
        """
        Parameters
        ----------
        format_dim : int
            Dimension for format embedding (default 24)
        region_dim : int
            Dimension for region embedding (default 24)
        operational_dim : int
            Dimension for operational features (default 32)
        identity_dim : int
            Dimension for store identity (default 16)
        """
        self.format_dim = format_dim
        self.region_dim = region_dim
        self.operational_dim = operational_dim
        self.identity_dim = identity_dim

        self.output_dim = format_dim + region_dim + operational_dim + identity_dim

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding lookup tables."""
        np.random.seed(42)

        # Format embeddings
        self.format_embeddings = {
            fmt: np.random.randn(self.format_dim) * 0.1
            for fmt in self.FORMAT_VOCAB
        }
        self.format_embeddings['UNK'] = np.zeros(self.format_dim)

        # Region embeddings
        self.region_embeddings = {
            reg: np.random.randn(self.region_dim) * 0.1
            for reg in self.REGION_VOCAB
        }
        self.region_embeddings['UNK'] = np.zeros(self.region_dim)

    def encode_store(
        self,
        store_id: str,
        store_format: str,
        store_region: str,
        operational_features: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Encode a single store to T5 tensor.

        Parameters
        ----------
        store_id : str
            Store identifier
        store_format : str
            Store format (LS, MS, SS)
        store_region : str
            Store region code
        operational_features : Dict, optional
            Operational metrics (size, traffic, competition, age)

        Returns
        -------
        np.ndarray [96d]
            Store context tensor
        """
        # Component 1: Format embedding [24d]
        format_embed = self.format_embeddings.get(
            store_format, self.format_embeddings['UNK']
        )

        # Component 2: Region embedding [24d]
        region_embed = self.region_embeddings.get(
            store_region, self.region_embeddings['UNK']
        )

        # Component 3: Operational features [32d]
        operational = self._encode_operational(operational_features)

        # Component 4: Store identity [16d]
        identity = self._encode_store_identity(store_id)

        # Concatenate all components
        t5 = np.concatenate([format_embed, region_embed, operational, identity])

        return t5

    def _encode_operational(self, features: Optional[Dict]) -> np.ndarray:
        """Encode operational features [32d]."""
        if features is None:
            return np.zeros(self.operational_dim)

        # Extract features with defaults
        store_size = features.get('store_size', 0.5)  # Normalized [0,1]
        traffic = features.get('traffic', 0.5)  # Avg daily customers, normalized
        competition = features.get('competition', 0.5)  # # competitors, normalized
        store_age = features.get('store_age', 0.5)  # Years operational, normalized

        # Core features
        core = np.array([
            store_size,
            traffic,
            competition,
            store_age,
            1 - store_size,  # Inverse
            1 - traffic,
            1 - competition,
            1 - store_age
        ])

        # Fourier expansion for each feature
        fourier = []
        for val in [store_size, traffic, competition, store_age]:
            fourier.extend([
                np.sin(np.pi * val),
                np.cos(np.pi * val),
                np.sin(2 * np.pi * val),
                np.cos(2 * np.pi * val)
            ])
        fourier = np.array(fourier)

        # Interaction features
        interactions = np.array([
            store_size * traffic,
            competition * (1 - traffic),
            store_age * store_size,
            traffic * (1 - competition)
        ])

        # Combine and pad/truncate to operational_dim
        combined = np.concatenate([core, fourier[:16], interactions])
        if len(combined) < self.operational_dim:
            combined = np.pad(combined, (0, self.operational_dim - len(combined)))

        return combined[:self.operational_dim]

    def _encode_store_identity(self, store_id: str) -> np.ndarray:
        """
        Encode store identity [16d].

        Low-dimensional to prevent overfitting.
        """
        # Hash-based embedding for consistency
        store_hash = hash(store_id) % (2**31)
        np.random.seed(store_hash)
        identity = np.random.randn(self.identity_dim) * 0.1

        # Normalize
        norm = np.linalg.norm(identity)
        if norm > 0:
            identity = identity / norm * 0.5

        return identity

    def encode_batch(
        self,
        store_df: pd.DataFrame,
        store_features_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode store context for all stores.

        Parameters
        ----------
        store_df : pd.DataFrame
            Store metadata with format and region
        store_features_df : pd.DataFrame, optional
            Pre-computed store features (from Layer 5)

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from store_id to T5 tensor
        """
        # Build operational features lookup
        operational_lookup = {}
        if store_features_df is not None:
            for _, row in store_features_df.iterrows():
                store_id = row['store_id']
                operational_lookup[store_id] = {
                    'store_size': row.get('baskets_per_week_norm', 0.5),
                    'traffic': row.get('customers_per_week_norm', 0.5),
                    'competition': row.get('avg_spend_norm', 0.5),
                    'store_age': row.get('tenure_weeks_norm', 0.5)
                }

        store_tensors = {}

        for _, row in store_df.iterrows():
            store_id = row.get('store_id') or row.get('STORE_CODE')
            store_format = row.get('format') or row.get('STORE_FORMAT', 'MS')
            store_region = row.get('region') or row.get('STORE_REGION', 'E01')

            operational = operational_lookup.get(store_id)

            t5 = self.encode_store(store_id, store_format, store_region, operational)
            store_tensors[store_id] = t5

        return store_tensors


def main():
    """Test store context encoding."""
    project_root = Path(__file__).parent.parent.parent

    # Load store data
    print("Loading store data...")
    transactions_df = pd.read_csv(
        project_root / 'raw_data' / 'transactions.csv',
        nrows=10000,
        usecols=['STORE_CODE', 'STORE_FORMAT', 'STORE_REGION']
    )

    # Get unique stores
    stores_df = transactions_df.groupby('STORE_CODE').agg({
        'STORE_FORMAT': 'first',
        'STORE_REGION': 'first'
    }).reset_index()
    stores_df.columns = ['store_id', 'format', 'region']

    print(f"Unique stores: {len(stores_df)}")

    # Load pre-computed store features if available
    store_features_path = project_root / 'data' / 'features' / 'store_features.parquet'
    store_features_df = None
    if store_features_path.exists():
        store_features_df = pd.read_parquet(store_features_path)
        print(f"Loaded store features: {len(store_features_df)}")

    # Create encoder
    encoder = StoreContextEncoder()
    print(f"\nOutput dimension: {encoder.output_dim}d")

    # Test single encoding
    t5 = encoder.encode_store(
        store_id='STORE00001',
        store_format='LS',
        store_region='E02',
        operational_features={
            'store_size': 0.8,
            'traffic': 0.7,
            'competition': 0.3,
            'store_age': 0.9
        }
    )
    print(f"Single encoding shape: {t5.shape}")

    # Test batch encoding
    store_tensors = encoder.encode_batch(stores_df, store_features_df)
    print(f"\nEncoded {len(store_tensors)} stores")

    # Sample output
    for store_id in list(store_tensors.keys())[:3]:
        t5 = store_tensors[store_id]
        print(f"  {store_id}: shape={t5.shape}, norm={np.linalg.norm(t5):.3f}")

    return encoder


if __name__ == '__main__':
    main()
