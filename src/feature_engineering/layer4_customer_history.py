"""
Layer 4: Customer History Encoding
===================================
Encodes customer purchase history using hierarchical architecture:
- Level 1: Trip-level encoding (products + mission per trip)
- Level 2: Customer-level aggregation (sequence of trips)
- Level 3: Statistical pattern extraction

Output: customer_history_embeddings.pkl ([N, 160] embeddings)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle
import warnings

warnings.filterwarnings('ignore')

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CustomerHistoryEncoder:
    """
    Hierarchical customer history encoder.

    Architecture:
    - Trip-level: Product sequence + Mission metadata -> [176d]
    - Customer-level: Trip sequence -> [128d]
    - Statistical: Mission pattern statistics -> [32d]

    Final output: [160d] customer embedding
    """

    # Mission metadata mappings
    MISSION_TYPE_MAP = {'Top Up': 0, 'Full Shop': 1, 'Small Shop': 2}
    MISSION_FOCUS_MAP = {'Fresh': 0, 'Grocery': 1, 'Mixed': 2, 'Nonfood': 3}
    PRICE_SENSITIVITY_MAP = {'LA': 0, 'MM': 1, 'UM': 2}
    BASKET_SIZE_MAP = {'S': 0, 'M': 1, 'L': 2}

    def __init__(
        self,
        max_trips: int = 20,
        max_products_per_trip: int = 50,
        product_embed_dim: int = 256,
        trip_embed_dim: int = 176,
        history_embed_dim: int = 128,
        stats_embed_dim: int = 32,
        cold_start_threshold: int = 5
    ):
        """
        Parameters
        ----------
        max_trips : int
            Maximum number of historical trips to consider
        max_products_per_trip : int
            Maximum products per trip
        product_embed_dim : int
            Dimension of product embeddings (from GraphSAGE)
        trip_embed_dim : int
            Output dimension of trip-level encoding
        history_embed_dim : int
            Output dimension of customer-level encoding
        stats_embed_dim : int
            Output dimension of statistical features
        cold_start_threshold : int
            Minimum trips for full history encoding
        """
        self.max_trips = max_trips
        self.max_products_per_trip = max_products_per_trip
        self.product_embed_dim = product_embed_dim
        self.trip_embed_dim = trip_embed_dim
        self.history_embed_dim = history_embed_dim
        self.stats_embed_dim = stats_embed_dim
        self.cold_start_threshold = cold_start_threshold

        # Total output dimension
        self.output_dim = history_embed_dim + stats_embed_dim  # 160d

        # Storage
        self.customer_embeddings = {}
        self.product_embeddings = {}

    def run(
        self,
        transactions_df: pd.DataFrame,
        product_embeddings: Dict[str, np.ndarray],
        mission_patterns: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate customer history embeddings.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transaction history with basket and mission info
        product_embeddings : Dict[str, np.ndarray]
            Product embeddings from GraphSAGE
        mission_patterns : pd.DataFrame, optional
            Pre-computed mission patterns from Stage 4

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from customer_id to 160d embedding
        """
        print("Layer 4: Customer History Encoding")
        print("=" * 50)

        self.product_embeddings = product_embeddings

        # Step 1: Build customer trip histories
        print("\nStep 1: Building customer trip histories...")
        customer_trips = self._build_customer_trips(transactions_df)
        print(f"  - Customers: {len(customer_trips):,}")

        # Step 2: Encode trip-level features
        print("\nStep 2: Encoding trip-level features...")
        trip_encodings = self._encode_trips(customer_trips)

        # Step 3: Encode customer-level features
        print("\nStep 3: Encoding customer-level sequences...")
        history_encodings = self._encode_customer_history(trip_encodings)

        # Step 4: Extract statistical patterns
        print("\nStep 4: Extracting statistical patterns...")
        if mission_patterns is not None:
            stats_encodings = self._encode_mission_stats(mission_patterns, customer_trips)
        else:
            stats_encodings = self._compute_stats_from_trips(customer_trips)

        # Step 5: Combine embeddings with cold-start blending
        print("\nStep 5: Combining embeddings with cold-start handling...")
        final_embeddings = self._combine_embeddings(
            customer_trips, history_encodings, stats_encodings
        )

        self.customer_embeddings = final_embeddings

        print("\n" + "=" * 50)
        print("Customer History Encoding Complete!")
        print(f"  - Customers with embeddings: {len(final_embeddings):,}")
        print(f"  - Embedding dimension: {self.output_dim}d")

        return final_embeddings

    def _build_customer_trips(
        self,
        transactions_df: pd.DataFrame
    ) -> Dict[str, List[Dict]]:
        """
        Build structured trip history for each customer.

        Returns
        -------
        Dict[str, List[Dict]]
            customer_trips[customer_id] = [
                {
                    'basket_id': ...,
                    'week': ...,
                    'products': [...],
                    'mission_type': ...,
                    'mission_focus': ...,
                    'price_sensitivity': ...,
                    'basket_size': ...
                },
                ...
            ]
        """
        # Filter to valid customers
        df = transactions_df[transactions_df['CUST_CODE'].notna()].copy()

        # Group by customer and basket
        customer_trips = defaultdict(list)

        # Aggregate products per basket
        basket_products = df.groupby(['CUST_CODE', 'BASKET_ID']).agg({
            'PROD_CODE': list,
            'SHOP_WEEK': 'first',
            'BASKET_TYPE': 'first',
            'BASKET_DOMINANT_MISSION': 'first',
            'BASKET_PRICE_SENSITIVITY': 'first',
            'BASKET_SIZE': 'first'
        }).reset_index()

        for _, row in basket_products.iterrows():
            customer_id = row['CUST_CODE']
            trip = {
                'basket_id': row['BASKET_ID'],
                'week': row['SHOP_WEEK'],
                'products': row['PROD_CODE'][:self.max_products_per_trip],
                'mission_type': row['BASKET_TYPE'],
                'mission_focus': row['BASKET_DOMINANT_MISSION'],
                'price_sensitivity': row['BASKET_PRICE_SENSITIVITY'],
                'basket_size': row['BASKET_SIZE']
            }
            customer_trips[customer_id].append(trip)

        # Sort trips by week and limit to max_trips
        for customer_id in customer_trips:
            trips = sorted(customer_trips[customer_id], key=lambda x: x['week'])
            customer_trips[customer_id] = trips[-self.max_trips:]

        return dict(customer_trips)

    def _encode_trips(
        self,
        customer_trips: Dict[str, List[Dict]]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Encode each trip using product embeddings + mission metadata.

        Trip encoding [176d]:
        - Product sequence pooling [128d]
        - Mission metadata [48d]
        """
        trip_encodings = {}

        # Mission embeddings (simple one-hot based)
        np.random.seed(42)
        mission_type_embeds = {k: np.random.randn(16) * 0.1
                              for k in self.MISSION_TYPE_MAP.values()}
        mission_focus_embeds = {k: np.random.randn(16) * 0.1
                               for k in self.MISSION_FOCUS_MAP.values()}
        sensitivity_embeds = {k: np.random.randn(8) * 0.1
                             for k in self.PRICE_SENSITIVITY_MAP.values()}
        size_embeds = {k: np.random.randn(8) * 0.1
                      for k in self.BASKET_SIZE_MAP.values()}

        for customer_id, trips in customer_trips.items():
            customer_trip_embeds = []

            for trip in trips:
                # Encode products
                product_embeds = []
                for prod in trip['products']:
                    if prod in self.product_embeddings:
                        product_embeds.append(self.product_embeddings[prod])

                if product_embeds:
                    # Mean pooling of product embeddings
                    product_vector = np.mean(product_embeds, axis=0)
                    # Reduce to 128d
                    if len(product_vector) > 128:
                        product_vector = product_vector[:128]
                    elif len(product_vector) < 128:
                        product_vector = np.pad(product_vector, (0, 128 - len(product_vector)))
                else:
                    product_vector = np.zeros(128)

                # Encode mission metadata
                mission_type_idx = self.MISSION_TYPE_MAP.get(trip['mission_type'], 0)
                mission_focus_idx = self.MISSION_FOCUS_MAP.get(trip['mission_focus'], 0)
                sensitivity_idx = self.PRICE_SENSITIVITY_MAP.get(trip['price_sensitivity'], 1)
                size_idx = self.BASKET_SIZE_MAP.get(trip['basket_size'], 1)

                mission_embed = np.concatenate([
                    mission_type_embeds.get(mission_type_idx, np.zeros(16)),
                    mission_focus_embeds.get(mission_focus_idx, np.zeros(16)),
                    sensitivity_embeds.get(sensitivity_idx, np.zeros(8)),
                    size_embeds.get(size_idx, np.zeros(8))
                ])  # [48d]

                # Combine
                trip_embed = np.concatenate([product_vector, mission_embed])  # [176d]
                customer_trip_embeds.append(trip_embed)

            trip_encodings[customer_id] = customer_trip_embeds

        return trip_encodings

    def _encode_customer_history(
        self,
        trip_encodings: Dict[str, List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Encode sequence of trips into customer-level representation.

        Uses attention-weighted pooling of trip embeddings.
        """
        history_encodings = {}

        # Simple projection matrix (would be learned in practice)
        np.random.seed(43)
        W_proj = np.random.randn(self.trip_embed_dim, self.history_embed_dim) * 0.1
        W_attn = np.random.randn(self.trip_embed_dim, 1) * 0.1

        for customer_id, trip_embeds in trip_encodings.items():
            if not trip_embeds:
                history_encodings[customer_id] = np.zeros(self.history_embed_dim)
                continue

            trip_matrix = np.array(trip_embeds)  # [num_trips, 176]

            # Compute attention weights (recency-biased)
            num_trips = len(trip_embeds)
            recency_weights = np.exp(np.linspace(-1, 0, num_trips))

            # Content-based attention
            content_scores = trip_matrix @ W_attn  # [num_trips, 1]
            content_weights = np.exp(content_scores.flatten())

            # Combine recency and content
            attention_weights = recency_weights * content_weights
            attention_weights = attention_weights / attention_weights.sum()

            # Weighted aggregation
            weighted_sum = (trip_matrix * attention_weights[:, np.newaxis]).sum(axis=0)

            # Project to output dimension
            history_embed = weighted_sum @ W_proj
            history_embed = np.tanh(history_embed)  # Activation

            history_encodings[customer_id] = history_embed

        return history_encodings

    def _encode_mission_stats(
        self,
        mission_patterns: pd.DataFrame,
        customer_trips: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
        """
        Encode pre-computed mission statistics.
        """
        stats_encodings = {}

        for _, row in mission_patterns.iterrows():
            customer_id = row['customer_id']

            # Extract key statistics
            features = []

            # Mission type distribution (up to 3 values)
            for mt in ['p_mission_top_up', 'p_mission_full_shop', 'p_mission_small_shop']:
                if mt in row:
                    features.append(row[mt] if not pd.isna(row[mt]) else 0.33)
                else:
                    features.append(0.33)

            # Mission focus distribution
            for mf in ['p_focus_fresh', 'p_focus_grocery', 'p_focus_mixed']:
                if mf in row:
                    features.append(row[mf] if not pd.isna(row[mf]) else 0.33)
                else:
                    features.append(0.33)

            # Sensitivity and size
            features.append(row.get('mean_price_sensitivity', 0.5))
            features.append(row.get('sensitivity_volatility', 0))
            features.append(row.get('mean_basket_size', 0.5))
            features.append(row.get('basket_size_variance', 0))

            # Entropy and consistency
            features.append(row.get('mission_type_entropy', 0))
            features.append(row.get('focus_entropy', 0))
            features.append(row.get('mission_consistency', 0.5))

            # Pad or truncate to stats_embed_dim
            features = np.array(features[:self.stats_embed_dim])
            if len(features) < self.stats_embed_dim:
                features = np.pad(features, (0, self.stats_embed_dim - len(features)))

            stats_encodings[customer_id] = features

        # Fill missing customers
        for customer_id in customer_trips:
            if customer_id not in stats_encodings:
                stats_encodings[customer_id] = np.ones(self.stats_embed_dim) * 0.5

        return stats_encodings

    def _compute_stats_from_trips(
        self,
        customer_trips: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute mission statistics directly from trip data.
        """
        stats_encodings = {}

        for customer_id, trips in customer_trips.items():
            if not trips:
                stats_encodings[customer_id] = np.zeros(self.stats_embed_dim)
                continue

            # Count mission types
            mission_types = [t['mission_type'] for t in trips]
            type_counts = defaultdict(int)
            for mt in mission_types:
                type_counts[mt] += 1
            total = len(trips)

            # Mission type distribution
            type_probs = [type_counts.get(mt, 0) / total
                         for mt in ['Top Up', 'Full Shop', 'Small Shop']]

            # Mission focus distribution
            focuses = [t['mission_focus'] for t in trips]
            focus_counts = defaultdict(int)
            for f in focuses:
                focus_counts[f] += 1
            focus_probs = [focus_counts.get(f, 0) / total
                          for f in ['Fresh', 'Grocery', 'Mixed']]

            # Sensitivity and size
            sensitivities = [self.PRICE_SENSITIVITY_MAP.get(t['price_sensitivity'], 1)
                            for t in trips]
            sizes = [self.BASKET_SIZE_MAP.get(t['basket_size'], 1)
                    for t in trips]

            mean_sens = np.mean(sensitivities) / 2  # Normalize to [0,1]
            std_sens = np.std(sensitivities) / 2
            mean_size = np.mean(sizes) / 2
            std_size = np.std(sizes) / 2

            # Entropy
            def entropy(probs):
                probs = np.array(probs)
                probs = probs[probs > 0]
                return -np.sum(probs * np.log(probs + 1e-10))

            type_entropy = entropy(type_probs)
            focus_entropy = entropy(focus_probs)

            # Consistency (inverse of total entropy)
            consistency = 1.0 / (1 + type_entropy + focus_entropy)

            # Combine features
            features = np.array(
                type_probs + focus_probs +
                [mean_sens, std_sens, mean_size, std_size,
                 type_entropy, focus_entropy, consistency]
            )

            # Pad to target dimension
            if len(features) < self.stats_embed_dim:
                features = np.pad(features, (0, self.stats_embed_dim - len(features)))
            else:
                features = features[:self.stats_embed_dim]

            stats_encodings[customer_id] = features

        return stats_encodings

    def _combine_embeddings(
        self,
        customer_trips: Dict[str, List[Dict]],
        history_encodings: Dict[str, np.ndarray],
        stats_encodings: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Combine history and stats embeddings with cold-start handling.

        Cold-start blending:
        - Few trips: Rely more on statistical defaults
        - Many trips: Rely more on learned history
        """
        final_embeddings = {}

        for customer_id in history_encodings:
            num_trips = len(customer_trips.get(customer_id, []))

            # Get embeddings
            history_embed = history_encodings.get(customer_id, np.zeros(self.history_embed_dim))
            stats_embed = stats_encodings.get(customer_id, np.zeros(self.stats_embed_dim))

            # Cold-start blending weight
            if num_trips < self.cold_start_threshold:
                # Heavy reliance on default patterns
                alpha = 0.8
            else:
                # Decay reliance as history grows
                alpha = max(0.2, 1.0 / np.log(num_trips + 1))

            # Default history embedding (population average approximation)
            default_history = np.zeros(self.history_embed_dim)

            # Blend history
            blended_history = alpha * default_history + (1 - alpha) * history_embed

            # Concatenate for final embedding
            final_embed = np.concatenate([blended_history, stats_embed])

            # L2 normalize
            norm = np.linalg.norm(final_embed)
            if norm > 0:
                final_embed = final_embed / norm

            final_embeddings[customer_id] = final_embed

        return final_embeddings

    def get_embedding(self, customer_id: str) -> Optional[np.ndarray]:
        """Get embedding for a single customer."""
        return self.customer_embeddings.get(customer_id)

    def save(self, filepath: str) -> None:
        """Save embeddings to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.customer_embeddings,
                'output_dim': self.output_dim
            }, f)
        print(f"Embeddings saved to: {filepath}")

    @staticmethod
    def load(filepath: str) -> Dict[str, np.ndarray]:
        """Load embeddings from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['embeddings']


def main():
    """Run customer history encoding on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    transactions_path = project_root / 'raw_data' / 'transactions.csv'
    product_embed_path = project_root / 'data' / 'features' / 'product_embeddings.pkl'
    mission_path = project_root / 'data' / 'processed' / 'customer_mission_patterns.parquet'
    output_path = project_root / 'data' / 'features' / 'customer_history_embeddings.pkl'

    # Load data
    print("Loading data...")
    transactions_df = pd.read_csv(
        transactions_path,
        nrows=10000,
        usecols=[
            'CUST_CODE', 'BASKET_ID', 'PROD_CODE', 'SHOP_WEEK',
            'BASKET_TYPE', 'BASKET_DOMINANT_MISSION',
            'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE'
        ]
    )
    print(f"  - Transactions: {len(transactions_df):,}")

    # Load product embeddings
    with open(product_embed_path, 'rb') as f:
        embed_data = pickle.load(f)
    product_embeddings = embed_data['embeddings']
    print(f"  - Product embeddings: {len(product_embeddings):,}")

    # Load mission patterns if available
    mission_patterns = None
    if mission_path.exists():
        mission_patterns = pd.read_parquet(mission_path)
        print(f"  - Mission patterns: {len(mission_patterns):,}")

    # Run encoder
    encoder = CustomerHistoryEncoder()
    embeddings = encoder.run(transactions_df, product_embeddings, mission_patterns)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(str(output_path))

    # Display sample
    print("\nSample embeddings:")
    sample_customers = list(embeddings.keys())[:5]
    for cust in sample_customers:
        emb = embeddings[cust]
        print(f"  {cust}: dim={len(emb)}, norm={np.linalg.norm(emb):.3f}")

    return embeddings


if __name__ == '__main__':
    main()
