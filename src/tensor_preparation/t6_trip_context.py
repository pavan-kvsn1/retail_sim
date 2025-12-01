"""
T6: Trip Context Tensor [48d]
==============================
Encodes current shopping mission context.

Components:
- Mission type [16d]: Top-up, Full-shop, etc.
- Mission focus [16d]: Fresh, Grocery, Mixed
- Price sensitivity mode [8d]: Low, Medium, High
- Expected basket scope [8d]: Small, Medium, Large

Dual-use: Training input + Auxiliary prediction target
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class TripContextEncoder:
    """
    Encodes trip context tensor T6 [48d].

    Represents current shopping mission for basket conditioning.
    """

    # Mission type vocabulary
    MISSION_TYPE_VOCAB = ['Top Up', 'Full Shop', 'Small Shop', 'Emergency']

    # Mission focus vocabulary
    MISSION_FOCUS_VOCAB = ['Fresh', 'Grocery', 'Mixed', 'Nonfood', 'General']

    # Price sensitivity modes
    PRICE_SENSITIVITY_VOCAB = ['LA', 'MM', 'UM']  # Low, Medium, High

    # Basket scope
    BASKET_SCOPE_VOCAB = ['S', 'M', 'L']  # Small, Medium, Large

    def __init__(
        self,
        mission_type_dim: int = 16,
        mission_focus_dim: int = 16,
        price_sensitivity_dim: int = 8,
        basket_scope_dim: int = 8
    ):
        """
        Parameters
        ----------
        mission_type_dim : int
            Dimension for mission type embedding (default 16)
        mission_focus_dim : int
            Dimension for mission focus embedding (default 16)
        price_sensitivity_dim : int
            Dimension for price sensitivity (default 8)
        basket_scope_dim : int
            Dimension for basket scope (default 8)
        """
        self.mission_type_dim = mission_type_dim
        self.mission_focus_dim = mission_focus_dim
        self.price_sensitivity_dim = price_sensitivity_dim
        self.basket_scope_dim = basket_scope_dim

        self.output_dim = (mission_type_dim + mission_focus_dim +
                          price_sensitivity_dim + basket_scope_dim)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding lookup tables."""
        np.random.seed(42)

        # Mission type embeddings
        self.mission_type_embeddings = {
            mt: np.random.randn(self.mission_type_dim) * 0.1
            for mt in self.MISSION_TYPE_VOCAB
        }
        self.mission_type_embeddings['UNK'] = np.zeros(self.mission_type_dim)

        # Mission focus embeddings
        self.mission_focus_embeddings = {
            mf: np.random.randn(self.mission_focus_dim) * 0.1
            for mf in self.MISSION_FOCUS_VOCAB
        }
        self.mission_focus_embeddings['UNK'] = np.zeros(self.mission_focus_dim)

        # Price sensitivity embeddings
        self.price_sensitivity_embeddings = {
            ps: np.random.randn(self.price_sensitivity_dim) * 0.1
            for ps in self.PRICE_SENSITIVITY_VOCAB
        }
        self.price_sensitivity_embeddings['UNK'] = np.zeros(self.price_sensitivity_dim)

        # Basket scope embeddings
        self.basket_scope_embeddings = {
            bs: np.random.randn(self.basket_scope_dim) * 0.1
            for bs in self.BASKET_SCOPE_VOCAB
        }
        self.basket_scope_embeddings['UNK'] = np.zeros(self.basket_scope_dim)

    def encode_trip(
        self,
        mission_type: str,
        mission_focus: str,
        price_sensitivity: str,
        basket_size: str
    ) -> np.ndarray:
        """
        Encode a single trip context to T6 tensor.

        Parameters
        ----------
        mission_type : str
            Type of shopping mission (Top Up, Full Shop, etc.)
        mission_focus : str
            Focus area (Fresh, Grocery, Mixed)
        price_sensitivity : str
            Price sensitivity mode (LA, MM, UM)
        basket_size : str
            Expected basket size (S, M, L)

        Returns
        -------
        np.ndarray [48d]
            Trip context tensor
        """
        # Component 1: Mission type [16d]
        type_embed = self.mission_type_embeddings.get(
            mission_type, self.mission_type_embeddings['UNK']
        )

        # Component 2: Mission focus [16d]
        focus_embed = self.mission_focus_embeddings.get(
            mission_focus, self.mission_focus_embeddings['UNK']
        )

        # Component 3: Price sensitivity [8d]
        sensitivity_embed = self.price_sensitivity_embeddings.get(
            price_sensitivity, self.price_sensitivity_embeddings['UNK']
        )

        # Component 4: Basket scope [8d]
        scope_embed = self.basket_scope_embeddings.get(
            basket_size, self.basket_scope_embeddings['UNK']
        )

        # Concatenate all components
        t6 = np.concatenate([type_embed, focus_embed, sensitivity_embed, scope_embed])

        return t6

    def encode_batch(
        self,
        transactions_df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Encode trip context for a batch of transactions.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transactions with BASKET_TYPE, BASKET_DOMINANT_MISSION,
            BASKET_PRICE_SENSITIVITY, BASKET_SIZE

        Returns
        -------
        tensors : np.ndarray [N, 48]
            Trip context tensors
        labels : Dict[str, np.ndarray]
            Label indices for auxiliary prediction
        """
        tensors = []
        labels = {
            'mission_type': [],
            'mission_focus': [],
            'price_sensitivity': [],
            'basket_size': []
        }

        for _, row in transactions_df.iterrows():
            mission_type = row.get('BASKET_TYPE', 'Top Up')
            mission_focus = row.get('BASKET_DOMINANT_MISSION', 'Mixed')
            price_sensitivity = row.get('BASKET_PRICE_SENSITIVITY', 'MM')
            basket_size = row.get('BASKET_SIZE', 'M')

            # Handle NaN values
            if pd.isna(mission_type):
                mission_type = 'Top Up'
            if pd.isna(mission_focus):
                mission_focus = 'Mixed'
            if pd.isna(price_sensitivity):
                price_sensitivity = 'MM'
            if pd.isna(basket_size):
                basket_size = 'M'

            t6 = self.encode_trip(mission_type, mission_focus, price_sensitivity, basket_size)
            tensors.append(t6)

            # Store label indices for auxiliary prediction
            labels['mission_type'].append(
                self.MISSION_TYPE_VOCAB.index(mission_type)
                if mission_type in self.MISSION_TYPE_VOCAB else 0
            )
            labels['mission_focus'].append(
                self.MISSION_FOCUS_VOCAB.index(mission_focus)
                if mission_focus in self.MISSION_FOCUS_VOCAB else 2
            )
            labels['price_sensitivity'].append(
                self.PRICE_SENSITIVITY_VOCAB.index(price_sensitivity)
                if price_sensitivity in self.PRICE_SENSITIVITY_VOCAB else 1
            )
            labels['basket_size'].append(
                self.BASKET_SCOPE_VOCAB.index(basket_size)
                if basket_size in self.BASKET_SCOPE_VOCAB else 1
            )

        # Convert to arrays
        tensors = np.array(tensors)
        for key in labels:
            labels[key] = np.array(labels[key])

        return tensors, labels

    def sample_from_distribution(
        self,
        mission_patterns: Dict
    ) -> Tuple[str, str, str, str]:
        """
        Sample trip context from customer's mission distribution.

        For inference/RL: Generate mission based on historical patterns.

        Parameters
        ----------
        mission_patterns : Dict
            Customer's mission distribution (from Stage 4)

        Returns
        -------
        Tuple of (mission_type, mission_focus, price_sensitivity, basket_size)
        """
        # Sample mission type
        type_probs = mission_patterns.get('mission_type_dist', {})
        if type_probs:
            types = list(type_probs.keys())
            probs = np.array([type_probs[t] for t in types])
            probs = probs / probs.sum()
            mission_type = np.random.choice(types, p=probs)
        else:
            mission_type = 'Top Up'

        # Sample mission focus
        focus_probs = mission_patterns.get('mission_focus_dist', {})
        if focus_probs:
            focuses = list(focus_probs.keys())
            probs = np.array([focus_probs[f] for f in focuses])
            probs = probs / probs.sum()
            mission_focus = np.random.choice(focuses, p=probs)
        else:
            mission_focus = 'Mixed'

        # Sample price sensitivity based on mean
        mean_sensitivity = mission_patterns.get('mean_price_sensitivity', 0.5)
        if mean_sensitivity < 0.33:
            price_sensitivity = 'LA'
        elif mean_sensitivity < 0.67:
            price_sensitivity = 'MM'
        else:
            price_sensitivity = 'UM'

        # Sample basket size based on mean
        mean_size = mission_patterns.get('mean_basket_size', 0.5)
        if mean_size < 0.4:
            basket_size = 'S'
        elif mean_size < 0.75:
            basket_size = 'M'
        else:
            basket_size = 'L'

        return mission_type, mission_focus, price_sensitivity, basket_size

    @property
    def num_mission_types(self) -> int:
        return len(self.MISSION_TYPE_VOCAB)

    @property
    def num_mission_focuses(self) -> int:
        return len(self.MISSION_FOCUS_VOCAB)

    @property
    def num_price_sensitivities(self) -> int:
        return len(self.PRICE_SENSITIVITY_VOCAB)

    @property
    def num_basket_sizes(self) -> int:
        return len(self.BASKET_SCOPE_VOCAB)


def main():
    """Test trip context encoding."""
    project_root = Path(__file__).parent.parent.parent

    # Load sample transactions
    print("Loading transactions...")
    transactions_df = pd.read_csv(
        project_root / 'raw_data' / 'transactions.csv',
        nrows=10000,
        usecols=[
            'BASKET_TYPE', 'BASKET_DOMINANT_MISSION',
            'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE'
        ]
    )

    # Create encoder
    encoder = TripContextEncoder()
    print(f"Output dimension: {encoder.output_dim}d")

    # Test single encoding
    t6 = encoder.encode_trip(
        mission_type='Full Shop',
        mission_focus='Fresh',
        price_sensitivity='MM',
        basket_size='L'
    )
    print(f"\nSingle encoding shape: {t6.shape}")

    # Test batch encoding
    tensors, labels = encoder.encode_batch(transactions_df.head(100))
    print(f"\nBatch encoding:")
    print(f"  Tensors shape: {tensors.shape}")
    print(f"  Label shapes: {[(k, v.shape) for k, v in labels.items()]}")

    # Test sampling
    mission_patterns = {
        'mission_type_dist': {'Top Up': 0.7, 'Full Shop': 0.3},
        'mission_focus_dist': {'Fresh': 0.5, 'Grocery': 0.3, 'Mixed': 0.2},
        'mean_price_sensitivity': 0.6,
        'mean_basket_size': 0.45
    }

    sampled = encoder.sample_from_distribution(mission_patterns)
    print(f"\nSampled trip: {sampled}")

    return encoder


if __name__ == '__main__':
    main()
