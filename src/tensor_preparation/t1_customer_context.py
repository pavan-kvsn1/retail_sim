"""
T1: Customer Context Tensor [192d]
===================================
Combines segment embeddings, history encoding, and store affinity.

Components:
- Segment embeddings [64d]: Cold-start initialization
- History encoding [128d]: Behavioral signature + mission patterns
- Store affinity [32d]: Spatial loyalty patterns

Includes adaptive blending for cold-start customers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle
import warnings

warnings.filterwarnings('ignore')


class CustomerContextEncoder:
    """
    Encodes customer context tensor T1 [192d].

    Architecture:
    - Segment embeddings [64d]
    - History encoding [128d]
    - Store affinity [32d]
    """

    # Segment vocabularies
    SEG1_VOCAB = ['CT', 'AZ', 'BG', 'DY', 'EF', 'GH', 'IJ', 'KL', 'MN', 'OP',
                  'QR', 'ST', 'UV', 'WX', 'YZ', 'AA', 'BB', 'CC', 'DD', 'EE']
    SEG2_VOCAB = ['DI', 'FN', 'BU', 'CZ', 'EQ', 'AT', 'GR', 'HI', 'JK', 'LM',
                  'NO', 'PQ', 'RS', 'TU', 'VW', 'XY', 'ZA', 'AB', 'BC', 'CD',
                  'DE', 'EF', 'FG', 'GH', 'HJ', 'IK', 'JL', 'KM', 'LN', 'MO']

    def __init__(
        self,
        segment_dim: int = 64,
        history_dim: int = 96,
        affinity_dim: int = 32,
        cold_start_threshold: int = 5
    ):
        """
        Parameters
        ----------
        segment_dim : int
            Dimension for segment embeddings (default 64)
        history_dim : int
            Dimension for history encoding (default 96, truncated from 160d input)
        affinity_dim : int
            Dimension for store affinity (default 32)
        cold_start_threshold : int
            Minimum trips for full history usage
        """
        self.segment_dim = segment_dim
        self.history_dim = history_dim
        self.affinity_dim = affinity_dim
        self.cold_start_threshold = cold_start_threshold
        self.output_dim = segment_dim + history_dim + affinity_dim  # 192d

        # Initialize embedding tables
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding lookup tables."""
        np.random.seed(42)

        # Segment 1 embeddings (lifestage)
        self.seg1_embeddings = {
            seg: np.random.randn(self.segment_dim // 2) * 0.1
            for seg in self.SEG1_VOCAB
        }
        self.seg1_embeddings['UNK'] = np.zeros(self.segment_dim // 2)

        # Segment 2 embeddings (lifestyle)
        self.seg2_embeddings = {
            seg: np.random.randn(self.segment_dim // 2) * 0.1
            for seg in self.SEG2_VOCAB
        }
        self.seg2_embeddings['UNK'] = np.zeros(self.segment_dim // 2)

    def encode_customer(
        self,
        customer_id: str,
        seg1: str,
        seg2: str,
        history_embedding: Optional[np.ndarray] = None,
        affinity_features: Optional[Dict] = None,
        num_trips: int = 0
    ) -> np.ndarray:
        """
        Encode a single customer to T1 tensor.

        Parameters
        ----------
        customer_id : str
            Customer identifier
        seg1 : str
            Lifestage segment code
        seg2 : str
            Lifestyle segment code
        history_embedding : np.ndarray, optional
            Pre-computed history embedding [160d from Layer 4]
        affinity_features : Dict, optional
            Store affinity metrics from Stage 3
        num_trips : int
            Number of historical trips (for cold-start blending)

        Returns
        -------
        np.ndarray
            Customer context tensor [192d]
        """
        # Component 1: Segment embeddings [64d]
        segment_features = self._encode_segments(seg1, seg2)

        # Component 2: History encoding [96d]
        if history_embedding is not None:
            # Use pre-computed history (from Layer 4: 160d)
            # Truncate or pad to history_dim (96d)
            if len(history_embedding) >= self.history_dim:
                history_features = history_embedding[:self.history_dim]
            else:
                history_features = np.pad(
                    history_embedding,
                    (0, self.history_dim - len(history_embedding))
                )
        else:
            history_features = np.zeros(self.history_dim)

        # Component 3: Store affinity [32d]
        affinity_encoded = self._encode_affinity(affinity_features)

        # Adaptive blending for cold-start
        # alpha: weight for segment features (high for cold-start)
        if num_trips < self.cold_start_threshold:
            alpha = 0.8  # Heavy reliance on segments for new customers
        else:
            alpha = max(0.2, 1.0 / np.log(num_trips + 1))

        # Scale features based on cold-start status
        # Cold-start: rely more on segments, less on history/affinity
        # Warm: rely more on history/affinity
        segment_scaled = segment_features  # Keep segments as-is
        history_scaled = history_features * (1 - alpha * 0.5)  # Reduce for cold-start
        affinity_scaled = affinity_encoded * (1 - alpha * 0.5)  # Reduce for cold-start

        # Final T1: Concat all components [64 + 96 + 32 = 192d]
        t1 = np.concatenate([segment_scaled, history_scaled, affinity_scaled])

        return t1

    def _encode_segments(self, seg1: str, seg2: str) -> np.ndarray:
        """Encode segment features [64d]."""
        seg1_embed = self.seg1_embeddings.get(seg1, self.seg1_embeddings['UNK'])
        seg2_embed = self.seg2_embeddings.get(seg2, self.seg2_embeddings['UNK'])
        return np.concatenate([seg1_embed, seg2_embed])

    def _encode_affinity(self, affinity_features: Optional[Dict]) -> np.ndarray:
        """Encode store affinity features [32d]."""
        if affinity_features is None:
            return np.zeros(self.affinity_dim)

        # Extract features
        loyalty_score = affinity_features.get('loyalty_score', 0.5)
        switching_rate = affinity_features.get('switching_rate', 0.1)
        region_diversity = affinity_features.get('region_diversity', 1)
        primary_store = affinity_features.get('primary_store', 'STORE00001')

        # Primary store embedding [16d]
        store_hash = hash(primary_store) % (2**31)
        np.random.seed(store_hash)
        store_embed = np.random.randn(16) * 0.1

        # Continuous features [16d]
        continuous = np.array([
            loyalty_score,
            switching_rate,
            region_diversity / 10,  # Normalize
            1 - loyalty_score,  # Inverse
        ])

        # Expand continuous to 16d
        continuous_expanded = np.zeros(16)
        continuous_expanded[:len(continuous)] = continuous
        # Add polynomial features
        continuous_expanded[4:8] = continuous ** 2
        continuous_expanded[8:12] = np.sin(continuous * np.pi)
        continuous_expanded[12:16] = np.cos(continuous * np.pi)

        return np.concatenate([store_embed, continuous_expanded])

    def encode_batch(
        self,
        transactions_df: pd.DataFrame,
        history_embeddings: Dict[str, np.ndarray],
        affinity_df: pd.DataFrame,
        trip_counts: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """
        Encode customer context for all customers in transactions.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transactions with customer segments
        history_embeddings : Dict[str, np.ndarray]
            Pre-computed history embeddings
        affinity_df : pd.DataFrame
            Customer-store affinity data
        trip_counts : Dict[str, int]
            Number of trips per customer

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from customer_id to T1 tensor
        """
        # Build affinity lookup
        affinity_lookup = {}
        for _, row in affinity_df.iterrows():
            affinity_lookup[row['customer_id']] = {
                'primary_store': row.get('primary_store', 'STORE00001'),
                'loyalty_score': row.get('loyalty_score', 0.5),
                'switching_rate': row.get('switching_rate', 0.1),
                'region_diversity': row.get('region_diversity', 1)
            }

        # Get unique customers with segments
        customers = transactions_df.groupby('CUST_CODE').agg({
            'seg_1': 'first',
            'seg_2': 'first'
        }).reset_index()

        customer_tensors = {}

        for _, row in customers.iterrows():
            customer_id = row['CUST_CODE']
            if pd.isna(customer_id):
                continue

            seg1 = row['seg_1'] if pd.notna(row['seg_1']) else 'UNK'
            seg2 = row['seg_2'] if pd.notna(row['seg_2']) else 'UNK'

            history_embed = history_embeddings.get(customer_id)
            affinity = affinity_lookup.get(customer_id)
            num_trips = trip_counts.get(customer_id, 0)

            t1 = self.encode_customer(
                customer_id, seg1, seg2,
                history_embed, affinity, num_trips
            )
            customer_tensors[customer_id] = t1

        return customer_tensors


def main():
    """Test customer context encoding."""
    project_root = Path(__file__).parent.parent.parent

    # Load required data
    print("Loading data...")
    transactions_df = pd.read_csv(
        project_root / 'raw_data' / 'transactions.csv',
        nrows=10000,
        usecols=['CUST_CODE', 'seg_1', 'seg_2', 'BASKET_ID']
    )

    # Load pre-computed embeddings
    history_path = project_root / 'data' / 'features' / 'customer_history_embeddings.pkl'
    affinity_path = project_root / 'data' / 'processed' / 'customer_store_affinity.parquet'

    history_embeddings = {}
    if history_path.exists():
        with open(history_path, 'rb') as f:
            data = pickle.load(f)
            history_embeddings = data.get('embeddings', {})

    affinity_df = pd.DataFrame()
    if affinity_path.exists():
        affinity_df = pd.read_parquet(affinity_path)

    # Compute trip counts
    trip_counts = transactions_df.groupby('CUST_CODE')['BASKET_ID'].nunique().to_dict()

    # Encode customers
    encoder = CustomerContextEncoder()
    customer_tensors = encoder.encode_batch(
        transactions_df, history_embeddings, affinity_df, trip_counts
    )

    print(f"\nEncoded {len(customer_tensors)} customers")
    print(f"Tensor dimension: {encoder.output_dim}d")

    # Sample output
    sample_customers = list(customer_tensors.keys())[:3]
    for cust in sample_customers:
        t1 = customer_tensors[cust]
        print(f"  {cust}: shape={t1.shape}, norm={np.linalg.norm(t1):.3f}")

    return customer_tensors


if __name__ == '__main__':
    main()
