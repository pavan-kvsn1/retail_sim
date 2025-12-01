"""
T3: Temporal Context Tensor [64d]
==================================
Encodes temporal features for each transaction.

Components:
- Calendar features [32d]: Week, weekday, hour embeddings
- Derived features [32d]: Holiday, season, trend, recency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class TemporalContextEncoder:
    """
    Encodes temporal context tensor T3 [64d].

    Components:
    - Week of year [16d]
    - Weekday [8d]
    - Hour of day [8d]
    - Holiday indicator [8d]
    - Season [8d]
    - Trend [8d]
    - Recency [8d]
    """

    # Holiday weeks (approximate)
    HOLIDAY_WEEKS = {
        # Christmas/New Year
        51, 52, 1,
        # Easter (varies)
        13, 14, 15,
        # Summer holidays
        26, 27, 28, 29, 30, 31, 32, 33,
        # Thanksgiving (US)
        47,
    }

    # Season mapping (by month)
    SEASON_MAP = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    SEASON_IDX = {'spring': 0, 'summer': 1, 'fall': 2, 'winter': 3}

    def __init__(
        self,
        week_dim: int = 16,
        weekday_dim: int = 8,
        hour_dim: int = 8,
        holiday_dim: int = 8,
        season_dim: int = 8,
        trend_dim: int = 8,
        recency_dim: int = 8
    ):
        """
        Parameters
        ----------
        Various embedding dimensions for each component.
        Total: 64d
        """
        self.week_dim = week_dim
        self.weekday_dim = weekday_dim
        self.hour_dim = hour_dim
        self.holiday_dim = holiday_dim
        self.season_dim = season_dim
        self.trend_dim = trend_dim
        self.recency_dim = recency_dim

        self.output_dim = (week_dim + weekday_dim + hour_dim +
                          holiday_dim + season_dim + trend_dim + recency_dim)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding lookup tables."""
        np.random.seed(42)

        # Week embeddings (52 weeks)
        self.week_embeddings = np.random.randn(53, self.week_dim) * 0.1

        # Weekday embeddings (7 days, 1=Monday to 7=Sunday)
        self.weekday_embeddings = np.random.randn(8, self.weekday_dim) * 0.1

        # Hour embeddings (24 hours)
        self.hour_embeddings = np.random.randn(25, self.hour_dim) * 0.1

        # Season embeddings (4 seasons)
        self.season_embeddings = np.random.randn(4, self.season_dim) * 0.1

    def encode_temporal(
        self,
        shop_week: int,
        shop_weekday: int,
        shop_hour: int,
        shop_date: Optional[int] = None,
        last_visit_week: Optional[int] = None,
        min_week: int = 200607,
        max_week: int = 200844
    ) -> np.ndarray:
        """
        Encode temporal features for a single transaction.

        Parameters
        ----------
        shop_week : int
            Week identifier (e.g., 200626)
        shop_weekday : int
            Day of week (1=Monday to 7=Sunday)
        shop_hour : int
            Hour of day (0-23)
        shop_date : int, optional
            Date in YYYYMMDD format
        last_visit_week : int, optional
            Customer's last visit week (for recency)
        min_week, max_week : int
            Week range for trend normalization

        Returns
        -------
        np.ndarray
            Temporal context tensor [64d]
        """
        # Week of year (extract from week identifier)
        week_of_year = shop_week % 100
        week_of_year = min(max(week_of_year, 1), 52)

        # Component 1: Calendar features [32d]
        week_embed = self.week_embeddings[week_of_year]
        weekday_embed = self.weekday_embeddings[min(shop_weekday, 7)]
        hour_embed = self.hour_embeddings[min(shop_hour, 24)]

        # Component 2: Derived features [32d]

        # Holiday indicator
        is_holiday = week_of_year in self.HOLIDAY_WEEKS
        holiday_embed = self._encode_holiday(is_holiday)

        # Season
        month = self._week_to_month(shop_week)
        season = self.SEASON_MAP.get(month, 'fall')
        season_idx = self.SEASON_IDX[season]
        season_embed = self.season_embeddings[season_idx]

        # Trend (normalized week index)
        trend_value = (shop_week - min_week) / max(max_week - min_week, 1)
        trend_embed = self._encode_continuous(trend_value, self.trend_dim)

        # Recency (days since last visit)
        if last_visit_week is not None and last_visit_week > 0:
            recency_weeks = shop_week - last_visit_week
            recency_value = min(recency_weeks / 52, 1.0)  # Normalize to year
        else:
            recency_value = 0.5  # Default for new customers
        recency_embed = self._encode_continuous(recency_value, self.recency_dim)

        # Concatenate all components
        t3 = np.concatenate([
            week_embed,      # [16d]
            weekday_embed,   # [8d]
            hour_embed,      # [8d]
            holiday_embed,   # [8d]
            season_embed,    # [8d]
            trend_embed,     # [8d]
            recency_embed    # [8d]
        ])

        return t3

    def _encode_holiday(self, is_holiday: bool) -> np.ndarray:
        """Encode holiday indicator [8d]."""
        if is_holiday:
            # Distinct pattern for holidays
            return np.array([1, 0, 1, 0, 1, 0, 1, 0]) * 0.5
        else:
            return np.array([0, 1, 0, 1, 0, 1, 0, 1]) * 0.5

    def _encode_continuous(self, value: float, dim: int) -> np.ndarray:
        """Encode continuous value using Fourier-like features."""
        features = np.zeros(dim)
        for i in range(dim // 2):
            freq = (i + 1) * np.pi
            features[2 * i] = np.sin(freq * value)
            features[2 * i + 1] = np.cos(freq * value)
        return features

    def _week_to_month(self, shop_week: int) -> int:
        """Approximate month from week identifier."""
        # Week format: YYYYWW
        year = shop_week // 100
        week = shop_week % 100

        # Approximate: week 1-4 = Jan, 5-8 = Feb, etc.
        month = ((week - 1) // 4) + 1
        return min(max(month, 1), 12)

    def encode_batch(
        self,
        transactions_df: pd.DataFrame,
        last_visit_lookup: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """
        Encode temporal context for a batch of transactions.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transactions with SHOP_WEEK, SHOP_WEEKDAY, SHOP_HOUR
        last_visit_lookup : Dict[str, int], optional
            Mapping from customer_id to last visit week

        Returns
        -------
        np.ndarray [N, 64]
            Temporal context tensors
        """
        if last_visit_lookup is None:
            last_visit_lookup = {}

        # Get week range for trend normalization
        min_week = transactions_df['SHOP_WEEK'].min()
        max_week = transactions_df['SHOP_WEEK'].max()

        tensors = []

        for _, row in transactions_df.iterrows():
            shop_week = row['SHOP_WEEK']
            shop_weekday = row.get('SHOP_WEEKDAY', 1)
            shop_hour = row.get('SHOP_HOUR', 12)

            customer_id = row.get('CUST_CODE')
            last_visit = last_visit_lookup.get(customer_id) if customer_id else None

            t3 = self.encode_temporal(
                shop_week, shop_weekday, shop_hour,
                last_visit_week=last_visit,
                min_week=min_week, max_week=max_week
            )
            tensors.append(t3)

        return np.array(tensors)


def main():
    """Test temporal context encoding."""
    project_root = Path(__file__).parent.parent.parent

    # Load sample transactions
    print("Loading transactions...")
    transactions_df = pd.read_csv(
        project_root / 'raw_data' / 'transactions.csv',
        nrows=10000,
        usecols=['SHOP_WEEK', 'SHOP_WEEKDAY', 'SHOP_HOUR', 'CUST_CODE', 'BASKET_ID']
    )

    # Build last visit lookup
    transactions_sorted = transactions_df.sort_values(['CUST_CODE', 'SHOP_WEEK'])
    last_visit_lookup = {}

    # For simplicity, use previous week as last visit
    for cust_id in transactions_df['CUST_CODE'].dropna().unique():
        cust_weeks = transactions_sorted[transactions_sorted['CUST_CODE'] == cust_id]['SHOP_WEEK']
        if len(cust_weeks) > 1:
            last_visit_lookup[cust_id] = cust_weeks.iloc[-2]

    # Encode temporal context
    encoder = TemporalContextEncoder()
    print(f"Output dimension: {encoder.output_dim}d")

    # Test single encoding
    t3 = encoder.encode_temporal(
        shop_week=200626,
        shop_weekday=3,
        shop_hour=14,
        last_visit_week=200625
    )
    print(f"\nSingle encoding shape: {t3.shape}")

    # Test batch encoding
    sample_df = transactions_df.head(100)
    batch_t3 = encoder.encode_batch(sample_df, last_visit_lookup)
    print(f"Batch encoding shape: {batch_t3.shape}")
    print(f"Mean norm: {np.linalg.norm(batch_t3, axis=1).mean():.3f}")

    return encoder


if __name__ == '__main__':
    main()
