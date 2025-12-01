"""
T4: Price Context Tensor [64d per item]
========================================
Sparse tensor with Fourier price encoding for each product in basket.

Components (from Layer 2):
- Fourier features [24d]
- Log-price features [16d]
- Relative price features [16d]
- Price velocity features [8d]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PriceContextBatch:
    """Container for batched price context data."""
    # Price features [B, S, 64]
    features: np.ndarray
    # Mask indicating valid positions [B, S]
    mask: np.ndarray
    # Sequence lengths [B]
    lengths: np.ndarray


class PriceContextEncoder:
    """
    Encodes price context tensor T4 [64d per product].

    Uses Fourier encoding for rich price representation.
    Sparse format: Only products in basket get price features.
    """

    # Frequencies for Fourier encoding
    FREQUENCIES = [1/7, 1/14, 1/30, 1/90, 1/2, 1/3, 1/4, 1/5]

    def __init__(
        self,
        fourier_dim: int = 24,
        log_dim: int = 16,
        relative_dim: int = 16,
        velocity_dim: int = 8,
        epsilon: float = 1e-6
    ):
        """
        Parameters
        ----------
        fourier_dim : int
            Dimension for Fourier features
        log_dim : int
            Dimension for log-price features
        relative_dim : int
            Dimension for relative price features
        velocity_dim : int
            Dimension for velocity features
        """
        self.fourier_dim = fourier_dim
        self.log_dim = log_dim
        self.relative_dim = relative_dim
        self.velocity_dim = velocity_dim
        self.epsilon = epsilon

        self.output_dim = fourier_dim + log_dim + relative_dim + velocity_dim

    def encode_price(
        self,
        actual_price: float,
        base_price: float,
        category_avg_price: float,
        prior_price: Optional[float] = None
    ) -> np.ndarray:
        """
        Encode a single price to 64d vector.

        Parameters
        ----------
        actual_price : float
            Current actual price
        base_price : float
            Base (shelf) price
        category_avg_price : float
            Average price in category
        prior_price : float, optional
            Prior week price (for velocity)

        Returns
        -------
        np.ndarray [64d]
            Price encoding
        """
        # Component 1: Fourier features [24d]
        fourier = self._encode_fourier(actual_price)

        # Component 2: Log-price features [16d]
        log_features = self._encode_log(actual_price, base_price)

        # Component 3: Relative price features [16d]
        relative = self._encode_relative(actual_price, category_avg_price)

        # Component 4: Velocity features [8d]
        if prior_price is None:
            prior_price = actual_price
        velocity = self._encode_velocity(actual_price, prior_price)

        return np.concatenate([fourier, log_features, relative, velocity])

    def _encode_fourier(self, price: float) -> np.ndarray:
        """Encode price using Fourier features [24d]."""
        # Generate sin/cos pairs
        fourier_raw = []
        for freq in self.FREQUENCIES:
            phase = 2 * np.pi * freq * price
            fourier_raw.extend([np.sin(phase), np.cos(phase)])

        fourier_raw = np.array(fourier_raw)  # [16d]

        # Expand to 24d with interaction terms
        if len(fourier_raw) < self.fourier_dim:
            interactions = []
            for i in range(4):
                sin_val = fourier_raw[i * 2]
                cos_val = fourier_raw[i * 2 + 1]
                interactions.append(sin_val * cos_val)
                interactions.append(sin_val ** 2)

            extra = np.array(interactions)[:self.fourier_dim - len(fourier_raw)]
            return np.concatenate([fourier_raw, extra])

        return fourier_raw[:self.fourier_dim]

    def _encode_log(self, actual_price: float, base_price: float) -> np.ndarray:
        """Encode log-price features [16d]."""
        log_actual = np.log(actual_price + self.epsilon)
        log_base = np.log(base_price + self.epsilon)
        log_diff = log_base - log_actual  # Discount in log space

        features = np.array([
            log_actual,
            log_base,
            log_diff,
            log_actual ** 2,
            np.exp(-log_actual),
            np.tanh(log_actual),
            log_actual * log_base,
            np.abs(log_diff)
        ])

        # Expand to 16d
        if len(features) < self.log_dim:
            features = np.pad(features, (0, self.log_dim - len(features)))

        return features[:self.log_dim]

    def _encode_relative(self, actual_price: float, category_avg: float) -> np.ndarray:
        """Encode relative price features [16d]."""
        relative = actual_price / (category_avg + self.epsilon)

        features = np.array([
            relative,
            np.log(relative + self.epsilon),
            relative ** 2,
            float(relative > 1.3),  # Premium
            float(relative < 0.7),  # Value
            float(0.7 <= relative <= 1.3),  # Mid
            np.sin(np.pi * min(relative, 2)),
            np.cos(np.pi * min(relative, 2))
        ])

        # Expand to 16d
        if len(features) < self.relative_dim:
            # Add polynomial expansions
            extra = []
            for i in range(self.relative_dim - len(features)):
                extra.append(relative ** (i % 3 + 1) * ((-1) ** i))
            features = np.concatenate([features, np.array(extra)])

        return features[:self.relative_dim]

    def _encode_velocity(self, current_price: float, prior_price: float) -> np.ndarray:
        """Encode price velocity features [8d]."""
        velocity = (current_price - prior_price) / (prior_price + self.epsilon)
        velocity = np.clip(velocity, -1, 1)

        features = np.array([
            velocity,
            np.abs(velocity),
            velocity ** 2,
            float(velocity > 0.05),   # Increasing
            float(velocity < -0.05),  # Decreasing
            float(-0.05 <= velocity <= 0.05),  # Stable
            np.sin(np.pi * velocity),
            np.cos(np.pi * velocity)
        ])

        return features[:self.velocity_dim]

    def encode_basket_prices(
        self,
        product_ids: List[str],
        price_lookup: Dict[str, Dict],
        category_avg_lookup: Dict[str, float]
    ) -> np.ndarray:
        """
        Encode prices for all products in a basket.

        Parameters
        ----------
        product_ids : List[str]
            Products in basket
        price_lookup : Dict[str, Dict]
            price_lookup[product_id] = {
                'actual_price': float,
                'base_price': float,
                'prior_price': float (optional)
            }
        category_avg_lookup : Dict[str, float]
            Average price by product category

        Returns
        -------
        np.ndarray [N, 64]
            Price encodings for each product
        """
        encodings = []

        for prod_id in product_ids:
            price_info = price_lookup.get(prod_id, {})
            actual = price_info.get('actual_price', 1.0)
            base = price_info.get('base_price', actual)
            prior = price_info.get('prior_price', actual)
            category_avg = category_avg_lookup.get(prod_id, actual)

            encoding = self.encode_price(actual, base, category_avg, prior)
            encodings.append(encoding)

        if encodings:
            return np.array(encodings)
        return np.zeros((0, self.output_dim))

    def encode_batch(
        self,
        baskets: List[List[str]],
        price_lookup: Dict[str, Dict],
        category_avg_lookup: Dict[str, float],
        max_seq_len: int = 50
    ) -> PriceContextBatch:
        """
        Encode price context for a batch of baskets.

        Parameters
        ----------
        baskets : List[List[str]]
            List of baskets
        price_lookup : Dict[str, Dict]
            Price information per product
        category_avg_lookup : Dict[str, float]
            Category average prices
        max_seq_len : int
            Maximum sequence length for padding

        Returns
        -------
        PriceContextBatch
            Batched price context
        """
        batch_size = len(baskets)

        # Encode all baskets
        all_encodings = []
        lengths = []

        for basket in baskets:
            encoding = self.encode_basket_prices(
                basket, price_lookup, category_avg_lookup
            )
            all_encodings.append(encoding)
            lengths.append(len(encoding))

        # Find max length - use max_seq_len to ensure alignment with product sequence
        # (which includes EOS token)
        actual_max = max(lengths) if lengths else 0
        pad_len = max_seq_len  # Use the provided max to match product sequence length

        # Pad to batch
        features = np.zeros((batch_size, pad_len, self.output_dim))
        mask = np.zeros((batch_size, pad_len), dtype=np.int32)

        for i, (enc, length) in enumerate(zip(all_encodings, lengths)):
            actual_len = min(length, pad_len)
            if actual_len > 0:
                features[i, :actual_len] = enc[:actual_len]
                mask[i, :actual_len] = 1

        return PriceContextBatch(
            features=features,
            mask=mask,
            lengths=np.array(lengths)
        )


def main():
    """Test price context encoding."""
    project_root = Path(__file__).parent.parent.parent

    # Create encoder
    encoder = PriceContextEncoder()
    print(f"Output dimension: {encoder.output_dim}d")

    # Test single price encoding
    t4 = encoder.encode_price(
        actual_price=1.99,
        base_price=2.49,
        category_avg_price=2.20,
        prior_price=2.19
    )
    print(f"\nSingle price encoding shape: {t4.shape}")
    print(f"Components: fourier={encoder.fourier_dim}, log={encoder.log_dim}, "
          f"relative={encoder.relative_dim}, velocity={encoder.velocity_dim}")

    # Test basket encoding
    products = ['PRD001', 'PRD002', 'PRD003']
    price_lookup = {
        'PRD001': {'actual_price': 1.99, 'base_price': 2.49, 'prior_price': 2.19},
        'PRD002': {'actual_price': 3.50, 'base_price': 3.50, 'prior_price': 3.50},
        'PRD003': {'actual_price': 0.99, 'base_price': 1.29, 'prior_price': 1.29},
    }
    category_avg = {'PRD001': 2.20, 'PRD002': 3.80, 'PRD003': 1.10}

    basket_encoding = encoder.encode_basket_prices(products, price_lookup, category_avg)
    print(f"\nBasket encoding shape: {basket_encoding.shape}")

    # Test batch encoding
    baskets = [
        ['PRD001', 'PRD002'],
        ['PRD001', 'PRD002', 'PRD003'],
        ['PRD003']
    ]
    batch = encoder.encode_batch(baskets, price_lookup, category_avg)
    print(f"\nBatch encoding:")
    print(f"  Features shape: {batch.features.shape}")
    print(f"  Mask shape: {batch.mask.shape}")
    print(f"  Lengths: {batch.lengths}")

    return encoder


if __name__ == '__main__':
    main()
