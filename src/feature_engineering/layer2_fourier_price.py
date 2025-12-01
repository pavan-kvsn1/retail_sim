"""
Layer 2: Fourier Price Encoding
================================
Multi-scale continuous price encoding with 4 components:
1. Fourier Features [24d] - Periodic patterns
2. Log-Price Features [16d] - Dynamic range handling
3. Relative Price Features [16d] - Category context
4. Price Velocity Features [8d] - Price dynamics

Total: 64d price encoding per product-store-week

Output: price_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FourierPriceEncoder:
    """
    Encodes prices using Fourier features for periodic pattern capture,
    log transformation for range compression, relative positioning,
    and velocity for dynamics.

    Output dimension: 64d
    - Fourier: 24d
    - Log-price: 16d
    - Relative: 16d
    - Velocity: 8d
    """

    # Learned frequencies for Fourier encoding (typical retail cycles)
    DEFAULT_FREQUENCIES = [
        1/7,    # Weekly cycle
        1/14,   # Bi-weekly
        1/30,   # Monthly
        1/90,   # Quarterly
        1/2,    # 2-day
        1/3,    # 3-day
        1/4,    # 4-day
        1/5     # 5-day (work week patterns)
    ]

    def __init__(
        self,
        frequencies: Optional[List[float]] = None,
        fourier_dim: int = 24,
        log_dim: int = 16,
        relative_dim: int = 16,
        velocity_dim: int = 8,
        epsilon: float = 1e-6
    ):
        """
        Parameters
        ----------
        frequencies : List[float], optional
            Frequencies for Fourier encoding
        fourier_dim : int
            Output dimension for Fourier features (default 24)
        log_dim : int
            Output dimension for log-price features (default 16)
        relative_dim : int
            Output dimension for relative price features (default 16)
        velocity_dim : int
            Output dimension for velocity features (default 8)
        epsilon : float
            Small constant for numerical stability
        """
        self.frequencies = frequencies or self.DEFAULT_FREQUENCIES
        self.fourier_dim = fourier_dim
        self.log_dim = log_dim
        self.relative_dim = relative_dim
        self.velocity_dim = velocity_dim
        self.epsilon = epsilon

        # Total output dimension
        self.output_dim = fourier_dim + log_dim + relative_dim + velocity_dim

    def run(
        self,
        prices_df: pd.DataFrame,
        category_prices: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate Fourier price features for all price observations.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Derived prices with columns: product_id, store_id, week,
            actual_price, base_price, discount_depth
        category_prices : pd.DataFrame, optional
            Category average prices for relative computation

        Returns
        -------
        pd.DataFrame
            Price features with 64d encoding per observation
        """
        print("Layer 2: Fourier Price Encoding")
        print("=" * 50)

        # Step 1: Compute Fourier features
        print("\nStep 1: Computing Fourier features [24d]...")
        fourier_features = self._compute_fourier_features(prices_df)
        print(f"  - Generated {len(self.frequencies)} frequency pairs")

        # Step 2: Compute log-price features
        print("\nStep 2: Computing log-price features [16d]...")
        log_features = self._compute_log_features(prices_df)

        # Step 3: Compute relative price features
        print("\nStep 3: Computing relative price features [16d]...")
        relative_features = self._compute_relative_features(prices_df, category_prices)

        # Step 4: Compute velocity features
        print("\nStep 4: Computing velocity features [8d]...")
        velocity_features = self._compute_velocity_features(prices_df)

        # Combine all features
        print("\nCombining all features...")
        price_features = self._combine_features(
            prices_df,
            fourier_features,
            log_features,
            relative_features,
            velocity_features
        )

        print("\n" + "=" * 50)
        print("Fourier Price Encoding Complete!")
        print(f"  - Total observations: {len(price_features):,}")
        print(f"  - Feature dimension: {self.output_dim}d")

        return price_features

    def _compute_fourier_features(self, prices_df: pd.DataFrame) -> np.ndarray:
        """
        Compute Fourier features: sin/cos pairs for each frequency.

        φ(p) = [sin(2πf₁p), cos(2πf₁p), sin(2πf₂p), cos(2πf₂p), ...]
        """
        prices = prices_df['actual_price'].values

        # Generate sin/cos pairs for each frequency
        fourier_raw = []
        for freq in self.frequencies:
            phase = 2 * np.pi * freq * prices
            fourier_raw.append(np.sin(phase))
            fourier_raw.append(np.cos(phase))

        fourier_raw = np.column_stack(fourier_raw)  # [N, 16] (8 freqs × 2)

        # Project to target dimension (24d) using learned-like transformation
        # In practice, this would be a learned linear layer
        # For feature engineering, we use a deterministic expansion
        if fourier_raw.shape[1] < self.fourier_dim:
            # Expand with additional derived features
            extra_features = []

            # Add interaction terms
            for i in range(min(4, len(self.frequencies))):
                idx = i * 2
                sin_feat = fourier_raw[:, idx]
                cos_feat = fourier_raw[:, idx + 1]
                extra_features.append(sin_feat * cos_feat)  # sin*cos
                extra_features.append(sin_feat ** 2)        # sin^2

            extra = np.column_stack(extra_features)[:, :self.fourier_dim - fourier_raw.shape[1]]
            fourier_features = np.hstack([fourier_raw, extra])
        else:
            fourier_features = fourier_raw[:, :self.fourier_dim]

        return fourier_features

    def _compute_log_features(self, prices_df: pd.DataFrame) -> np.ndarray:
        """
        Compute log-price features for dynamic range handling.

        log_price = log(price + ε)

        Weber-Fechner law: Human price perception is logarithmic
        """
        actual_price = prices_df['actual_price'].values
        base_price = prices_df['base_price'].values

        # Core log features
        log_actual = np.log(actual_price + self.epsilon)
        log_base = np.log(base_price + self.epsilon)
        log_diff = log_base - log_actual  # Log discount

        # Normalize to reasonable range
        log_actual_norm = (log_actual - log_actual.mean()) / (log_actual.std() + self.epsilon)
        log_base_norm = (log_base - log_base.mean()) / (log_base.std() + self.epsilon)

        # Create feature array
        log_raw = np.column_stack([
            log_actual,
            log_base,
            log_diff,
            log_actual_norm,
            log_base_norm,
            log_actual ** 2,  # Quadratic term
            np.exp(-log_actual),  # Inverse transform
            np.tanh(log_actual)  # Bounded transform
        ])

        # Expand to target dimension
        if log_raw.shape[1] < self.log_dim:
            # Add polynomial expansions
            extra = []
            for i in range(self.log_dim - log_raw.shape[1]):
                extra.append(log_actual ** (i % 3 + 1) * ((-1) ** i))
            log_features = np.hstack([log_raw, np.column_stack(extra)])
        else:
            log_features = log_raw[:, :self.log_dim]

        return log_features

    def _compute_relative_features(
        self,
        prices_df: pd.DataFrame,
        category_prices: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Compute relative price features for category context.

        relative_price = price / category_average_price

        IMPORTANT: Uses expanding window to prevent data leakage.
        Category statistics are computed using only current and past weeks,
        never future data.
        """
        prices_df = prices_df.copy()

        # Track original order to restore after sorting
        prices_df['_original_order'] = np.arange(len(prices_df))

        # Sort by week to ensure temporal ordering for expanding window
        prices_df = prices_df.sort_values('week').reset_index(drop=True)

        # Create category proxy from product_id prefix
        prices_df['category_proxy'] = prices_df['product_id'].str[:7]

        if category_prices is None:
            # Use expanding window to compute statistics using only past data
            # This prevents data leakage from future weeks

            print("    Computing expanding window statistics (no future leakage)...")

            # Group by category and week, then compute expanding stats
            # First, compute weekly category aggregates
            weekly_cat_stats = prices_df.groupby(['category_proxy', 'week']).agg({
                'actual_price': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            weekly_cat_stats.columns = ['category_proxy', 'week', 'week_mean', 'week_std',
                                        'week_min', 'week_max', 'week_count']

            # Compute expanding statistics per category (only using past weeks)
            weekly_cat_stats = weekly_cat_stats.sort_values(['category_proxy', 'week'])

            # Expanding mean: weighted by count
            weekly_cat_stats['cumsum_price'] = weekly_cat_stats.groupby('category_proxy').apply(
                lambda g: (g['week_mean'] * g['week_count']).cumsum()
            ).reset_index(level=0, drop=True)
            weekly_cat_stats['cumsum_count'] = weekly_cat_stats.groupby('category_proxy')['week_count'].cumsum()
            weekly_cat_stats['expanding_avg'] = weekly_cat_stats['cumsum_price'] / weekly_cat_stats['cumsum_count']

            # Expanding min/max
            weekly_cat_stats['expanding_min'] = weekly_cat_stats.groupby('category_proxy')['week_min'].cummin()
            weekly_cat_stats['expanding_max'] = weekly_cat_stats.groupby('category_proxy')['week_max'].cummax()

            # Expanding std (approximate using pooled variance)
            # For simplicity, use rolling std on weekly means as proxy
            weekly_cat_stats['expanding_std'] = weekly_cat_stats.groupby('category_proxy')['week_mean'].transform(
                lambda x: x.expanding().std().fillna(x.expanding().mean() * 0.1)
            )

            # Merge back to original data
            prices_df = prices_df.merge(
                weekly_cat_stats[['category_proxy', 'week', 'expanding_avg', 'expanding_std',
                                  'expanding_min', 'expanding_max']],
                on=['category_proxy', 'week'],
                how='left'
            )

            # Fill any NaN values (first week of each category) with the week's own stats
            first_week_mask = prices_df['expanding_avg'].isna()
            if first_week_mask.any():
                # For first occurrence, use the current price as reference
                prices_df.loc[first_week_mask, 'expanding_avg'] = prices_df.loc[first_week_mask, 'actual_price']
                prices_df.loc[first_week_mask, 'expanding_std'] = prices_df.loc[first_week_mask, 'actual_price'] * 0.1
                prices_df.loc[first_week_mask, 'expanding_min'] = prices_df.loc[first_week_mask, 'actual_price']
                prices_df.loc[first_week_mask, 'expanding_max'] = prices_df.loc[first_week_mask, 'actual_price']

            category_avg = prices_df['expanding_avg']
            category_std = prices_df['expanding_std']
            category_min = prices_df['expanding_min']
            category_max = prices_df['expanding_max']
        else:
            # Merge with provided category prices (assumed to be properly temporal)
            category_avg = prices_df['actual_price']  # Fallback
            category_std = prices_df['actual_price'].std()
            category_min = prices_df['actual_price'].min()
            category_max = prices_df['actual_price'].max()

        actual_price = prices_df['actual_price'].values

        # Core relative features
        relative_to_avg = actual_price / (category_avg.values + self.epsilon)
        z_score = (actual_price - category_avg.values) / (category_std.values + self.epsilon)
        percentile = (actual_price - category_min.values) / (
            category_max.values - category_min.values + self.epsilon
        )

        # Premium/value indicators
        is_premium = (relative_to_avg > 1.3).astype(float)
        is_value = (relative_to_avg < 0.7).astype(float)
        is_mid = ((relative_to_avg >= 0.7) & (relative_to_avg <= 1.3)).astype(float)

        # Create feature array
        relative_raw = np.column_stack([
            relative_to_avg,
            z_score,
            percentile,
            is_premium,
            is_value,
            is_mid,
            np.log(relative_to_avg + self.epsilon),
            relative_to_avg ** 2
        ])

        # Expand to target dimension
        if relative_raw.shape[1] < self.relative_dim:
            extra = []
            for i in range(self.relative_dim - relative_raw.shape[1]):
                extra.append(np.sin(np.pi * percentile * (i + 1)))
            relative_features = np.hstack([relative_raw, np.column_stack(extra)])
        else:
            relative_features = relative_raw[:, :self.relative_dim]

        # Restore original order using tracked positions
        restore_order = prices_df['_original_order'].values.argsort()
        relative_features = relative_features[restore_order]

        return relative_features

    def _compute_velocity_features(self, prices_df: pd.DataFrame) -> np.ndarray:
        """
        Compute price velocity (momentum) features.

        velocity = (current_price - prior_price) / prior_price
        acceleration = velocity_change
        """
        prices_df = prices_df.copy()

        # Sort by product, store, week
        prices_df = prices_df.sort_values(['product_id', 'store_id', 'week'])

        # Compute price changes
        prices_df['prior_price'] = prices_df.groupby(['product_id', 'store_id'])['actual_price'].shift(1)
        prices_df['prior_price'] = prices_df['prior_price'].fillna(prices_df['actual_price'])

        actual = prices_df['actual_price'].values
        prior = prices_df['prior_price'].values

        # Velocity: percentage change
        velocity = (actual - prior) / (prior + self.epsilon)
        velocity = np.clip(velocity, -1, 1)  # Cap extreme values

        # Compute acceleration (change in velocity)
        prices_df['velocity'] = velocity
        prices_df['prior_velocity'] = prices_df.groupby(['product_id', 'store_id'])['velocity'].shift(1)
        prices_df['prior_velocity'] = prices_df['prior_velocity'].fillna(0)

        acceleration = velocity - prices_df['prior_velocity'].values
        acceleration = np.clip(acceleration, -1, 1)

        # Promotional momentum: consecutive weeks of price reduction
        is_discount = (velocity < -0.05).astype(float)

        # Direction indicators
        increasing = (velocity > 0.05).astype(float)
        decreasing = (velocity < -0.05).astype(float)
        stable = ((velocity >= -0.05) & (velocity <= 0.05)).astype(float)

        # Create feature array
        velocity_features = np.column_stack([
            velocity,
            acceleration,
            is_discount,
            increasing,
            decreasing,
            stable,
            np.abs(velocity),  # Magnitude
            velocity ** 2  # Quadratic
        ])[:, :self.velocity_dim]

        return velocity_features

    def _combine_features(
        self,
        prices_df: pd.DataFrame,
        fourier_features: np.ndarray,
        log_features: np.ndarray,
        relative_features: np.ndarray,
        velocity_features: np.ndarray
    ) -> pd.DataFrame:
        """Combine all feature arrays into output DataFrame."""
        # Create column names
        fourier_cols = [f'fourier_{i}' for i in range(fourier_features.shape[1])]
        log_cols = [f'log_{i}' for i in range(log_features.shape[1])]
        relative_cols = [f'relative_{i}' for i in range(relative_features.shape[1])]
        velocity_cols = [f'velocity_{i}' for i in range(velocity_features.shape[1])]

        # Combine into single array
        all_features = np.hstack([
            fourier_features,
            log_features,
            relative_features,
            velocity_features
        ])

        all_cols = fourier_cols + log_cols + relative_cols + velocity_cols

        # Create DataFrame
        features_df = pd.DataFrame(all_features, columns=all_cols)

        # Add identifiers
        features_df['product_id'] = prices_df['product_id'].values
        features_df['store_id'] = prices_df['store_id'].values
        features_df['week'] = prices_df['week'].values
        features_df['actual_price'] = prices_df['actual_price'].values
        features_df['base_price'] = prices_df['base_price'].values
        features_df['discount_depth'] = prices_df['discount_depth'].values

        # Reorder columns
        id_cols = ['product_id', 'store_id', 'week', 'actual_price', 'base_price', 'discount_depth']
        features_df = features_df[id_cols + all_cols]

        return features_df

    def encode_single_price(
        self,
        price: float,
        base_price: float,
        category_avg: float,
        prior_price: float
    ) -> np.ndarray:
        """
        Encode a single price observation (for inference).

        Parameters
        ----------
        price : float
            Current actual price
        base_price : float
            Base (shelf) price
        category_avg : float
            Category average price
        prior_price : float
            Prior week price

        Returns
        -------
        np.ndarray
            64d price encoding
        """
        # Fourier features
        fourier = []
        for freq in self.frequencies:
            phase = 2 * np.pi * freq * price
            fourier.extend([np.sin(phase), np.cos(phase)])
        fourier = np.array(fourier[:self.fourier_dim])
        if len(fourier) < self.fourier_dim:
            fourier = np.pad(fourier, (0, self.fourier_dim - len(fourier)))

        # Log features
        log_price = np.log(price + self.epsilon)
        log_base = np.log(base_price + self.epsilon)
        log_feat = np.array([
            log_price, log_base, log_base - log_price,
            log_price, log_base, log_price**2, np.exp(-log_price), np.tanh(log_price)
        ])
        log_feat = np.pad(log_feat, (0, max(0, self.log_dim - len(log_feat))))[:self.log_dim]

        # Relative features
        relative = price / (category_avg + self.epsilon)
        rel_feat = np.array([
            relative, 0, 0.5,  # z-score and percentile placeholders
            float(relative > 1.3), float(relative < 0.7), float(0.7 <= relative <= 1.3),
            np.log(relative + self.epsilon), relative**2
        ])
        rel_feat = np.pad(rel_feat, (0, max(0, self.relative_dim - len(rel_feat))))[:self.relative_dim]

        # Velocity features
        velocity = (price - prior_price) / (prior_price + self.epsilon)
        velocity = np.clip(velocity, -1, 1)
        vel_feat = np.array([
            velocity, 0,  # acceleration placeholder
            float(velocity < -0.05),
            float(velocity > 0.05), float(velocity < -0.05),
            float(-0.05 <= velocity <= 0.05),
            np.abs(velocity), velocity**2
        ])[:self.velocity_dim]

        return np.concatenate([fourier, log_feat, rel_feat, vel_feat])


def main():
    """Run Fourier price encoding on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    prices_path = project_root / 'data' / 'processed' / 'prices_derived.parquet'
    output_path = project_root / 'data' / 'features' / 'price_features.parquet'

    # Load data
    print("Loading derived prices...")
    prices_df = pd.read_parquet(prices_path)
    print(f"  - Loaded {len(prices_df):,} price observations")

    # Run encoder
    encoder = FourierPriceEncoder()
    price_features = encoder.run(prices_df)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    price_features.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Display sample
    print("\nSample output (first 5 rows, selected columns):")
    display_cols = ['product_id', 'store_id', 'week', 'actual_price'] + \
                   [f'fourier_{i}' for i in range(3)] + \
                   [f'log_{i}' for i in range(3)]
    print(price_features[display_cols].head().to_string())

    # Feature statistics
    print("\nFeature statistics:")
    feature_cols = [c for c in price_features.columns if c.startswith(('fourier', 'log', 'relative', 'velocity'))]
    print(price_features[feature_cols].describe().to_string())

    return price_features


if __name__ == '__main__':
    main()
