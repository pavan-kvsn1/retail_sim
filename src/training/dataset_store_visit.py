"""
Store Visit Prediction Dataset.

Provides samples for predicting which store a customer will visit next.
Reuses the next-basket samples but focuses on store prediction.

Each sample contains:
- customer_context: Customer history embedding [192]
- temporal_context: When is the next visit [64]
- previous_store_idx: Which store they visited last (index)
- previous_basket: Products from last visit (optional, for summary)
- target_store_idx: Which store they will visit (label)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass, field
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StoreVisitBatch:
    """Container for a batch of store visit prediction samples."""
    # Context tensors
    customer_context: np.ndarray      # [B, 192]
    temporal_context: np.ndarray      # [B, 64]
    previous_store_idx: np.ndarray    # [B] - index of previous store

    # Target
    target_store_idx: np.ndarray      # [B] - index of target store (label)

    # Previous basket (optional, for basket-aware prediction)
    previous_basket_embeddings: Optional[np.ndarray] = None  # [B, S, 256]
    previous_basket_mask: Optional[np.ndarray] = None        # [B, S]

    # Metadata
    customer_ids: Optional[np.ndarray] = None
    previous_store_ids: Optional[np.ndarray] = None
    target_store_ids: Optional[np.ndarray] = None

    @property
    def batch_size(self) -> int:
        return self.customer_context.shape[0]


class StoreVisitDataset:
    """
    Dataset for store visit prediction.

    Predicts: P(next_store | customer, time, previous_store, previous_basket)
    """

    def __init__(
        self,
        project_root: Path,
        split: str = 'train',
        include_basket: bool = True,
        max_basket_len: int = 50,
        vocabulary: Optional[Dict] = None,
    ):
        """
        Initialize store visit dataset.

        Args:
            project_root: Path to retail_sim project root
            split: One of 'train', 'validation', 'test'
            include_basket: Whether to include previous basket for prediction
            max_basket_len: Maximum products to include from previous basket
            vocabulary: Optional shared vocabulary from training set
        """
        self.project_root = Path(project_root)
        self.split = split
        self.include_basket = include_basket
        self.max_basket_len = max_basket_len

        # Load samples (reuse next-basket samples)
        self._load_samples()

        # Load embeddings
        self._load_embeddings()

        # Build store vocabulary
        if vocabulary is not None:
            self._use_shared_vocabulary(vocabulary)
        else:
            self._build_store_vocabulary()

        # Initialize encoders
        self._init_encoders()

        logger.info(f"StoreVisitDataset initialized: {len(self)} samples ({split})")
        logger.info(f"  - {self.num_stores} stores")

    def _load_samples(self):
        """Load samples from next-basket parquet files."""
        samples_path = self.project_root / 'data' / 'prepared' / f'{self.split}_next_basket.parquet'

        if not samples_path.exists():
            raise FileNotFoundError(
                f"Samples not found at {samples_path}. "
                f"Run: python src/data_preparation/stage4_next_basket_samples.py"
            )

        self.samples = pd.read_parquet(samples_path)

        # Deserialize product lists if needed
        if self.include_basket and 'input_products' in self.samples.columns:
            self.samples['input_products'] = self.samples['input_products'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        logger.info(f"Loaded {len(self.samples):,} {self.split} samples")

        # Get week range for temporal encoding
        self.min_week = self.samples['target_week'].min()
        self.max_week = self.samples['target_week'].max()

    def _load_embeddings(self):
        """Load pre-computed embeddings."""
        cache_dir = self.project_root / 'data' / 'tensor_cache'

        # Customer embeddings
        cust_hist_path = cache_dir / 'customer_history_embeddings.npy'
        if cust_hist_path.exists():
            self.customer_history = np.load(cust_hist_path)
        else:
            self.customer_history = np.random.randn(100000, 160).astype(np.float32) * 0.1

        cust_static_path = cache_dir / 'customer_static_embeddings.npy'
        if cust_static_path.exists():
            self.customer_static = np.load(cust_static_path)
        else:
            self.customer_static = np.random.randn(100000, 160).astype(np.float32) * 0.1

        # Product embeddings (for basket summary)
        if self.include_basket:
            product_path = cache_dir / 'product_embeddings.npy'
            if product_path.exists():
                self.product_embeddings = np.load(product_path)
            else:
                self.product_embeddings = np.random.randn(5000, 256).astype(np.float32) * 0.1

        # Vocabulary mapping
        vocab_path = cache_dir / 'vocab.json'
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}

        logger.info(f"Loaded embeddings from {cache_dir}")

    def _build_store_vocabulary(self):
        """Build store vocabulary from samples."""
        # Get all unique stores
        all_stores = set(self.samples['input_store_id'].unique())
        all_stores.update(self.samples['target_store_id'].unique())

        self.store_list = sorted(all_stores)
        self.store_to_idx = {s: i for i, s in enumerate(self.store_list)}
        self.idx_to_store = {i: s for i, s in enumerate(self.store_list)}
        self.num_stores = len(self.store_list)

        # Build product vocabulary for basket encoding
        if self.include_basket:
            all_products = set()
            for products in self.samples['input_products'].values:
                all_products.update(products)
            self.product_list = sorted(all_products)
            self.product_to_idx = {p: i + 1 for i, p in enumerate(self.product_list)}

        logger.info(f"Store vocabulary: {self.num_stores} stores")

    def _use_shared_vocabulary(self, vocabulary: Dict):
        """Use shared vocabulary from training set."""
        self.store_to_idx = vocabulary['store_to_idx']
        self.idx_to_store = vocabulary['idx_to_store']
        self.store_list = vocabulary['store_list']
        self.num_stores = vocabulary['num_stores']

        if self.include_basket and 'product_to_idx' in vocabulary:
            self.product_to_idx = vocabulary['product_to_idx']
            self.product_list = vocabulary.get('product_list', [])

        logger.info(f"Using shared vocabulary: {self.num_stores} stores")

    def get_vocabulary(self) -> Dict:
        """Export vocabulary for sharing."""
        vocab = {
            'store_to_idx': self.store_to_idx,
            'idx_to_store': self.idx_to_store,
            'store_list': self.store_list,
            'num_stores': self.num_stores,
        }
        if self.include_basket:
            vocab['product_to_idx'] = self.product_to_idx
            vocab['product_list'] = self.product_list
        return vocab

    def _init_encoders(self):
        """Initialize temporal encoding matrices."""
        np.random.seed(42)

        self.week_matrix = np.random.randn(53, 16).astype(np.float32) * 0.1
        self.weekday_matrix = np.random.randn(8, 8).astype(np.float32) * 0.1
        self.hour_matrix = np.random.randn(25, 8).astype(np.float32) * 0.1
        self.season_matrix = np.random.randn(4, 8).astype(np.float32) * 0.1

        self.holiday_weeks = {51, 52, 1, 13, 14, 15, 26, 27, 28, 29, 30, 31, 32, 33, 47}

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, indices: List[int]) -> StoreVisitBatch:
        """Get a batch of store visit prediction samples."""
        batch_size = len(indices)
        batch_samples = self.samples.iloc[indices]

        # Extract data
        customer_ids = batch_samples['customer_id'].values
        input_store_ids = batch_samples['input_store_id'].values
        target_store_ids = batch_samples['target_store_id'].values
        target_weeks = batch_samples['target_week'].values
        target_weekdays = batch_samples['target_weekday'].values
        target_hours = batch_samples['target_hour'].values

        # Encode customer context [B, 192]
        customer_context = self._encode_customer_batch(customer_ids)

        # Encode temporal context [B, 64]
        temporal_context = self._encode_temporal_batch(target_weeks, target_weekdays, target_hours)

        # Encode previous store index [B]
        previous_store_idx = np.array([
            self.store_to_idx.get(sid, 0) for sid in input_store_ids
        ], dtype=np.int64)

        # Encode target store index [B]
        target_store_idx = np.array([
            self.store_to_idx.get(sid, 0) for sid in target_store_ids
        ], dtype=np.int64)

        # Encode previous basket (optional)
        if self.include_basket:
            input_products_list = batch_samples['input_products'].tolist()
            basket_embeddings, basket_mask = self._encode_basket_batch(input_products_list)
        else:
            basket_embeddings = None
            basket_mask = None

        return StoreVisitBatch(
            customer_context=customer_context,
            temporal_context=temporal_context,
            previous_store_idx=previous_store_idx,
            previous_basket_embeddings=basket_embeddings,
            previous_basket_mask=basket_mask,
            target_store_idx=target_store_idx,
            customer_ids=customer_ids,
            previous_store_ids=input_store_ids,
            target_store_ids=target_store_ids,
        )

    def _encode_customer_batch(self, customer_ids: np.ndarray) -> np.ndarray:
        """Encode customer context [B, 192]."""
        batch_size = len(customer_ids)

        customer_indices = np.array([
            self.vocab.get('customer_to_idx', {}).get(cid, 0)
            for cid in customer_ids
        ])

        hist_dim = min(96, self.customer_history.shape[1])
        history = self.customer_history[np.clip(customer_indices, 0, len(self.customer_history) - 1), :hist_dim]
        if hist_dim < 96:
            history = np.pad(history, ((0, 0), (0, 96 - hist_dim)))

        static_dim = min(64, self.customer_static.shape[1])
        static = self.customer_static[np.clip(customer_indices, 0, len(self.customer_static) - 1), :static_dim]
        if static_dim < 64:
            static = np.pad(static, ((0, 0), (0, 64 - static_dim)))

        # Customer affinity placeholder
        affinity = np.zeros((batch_size, 32), dtype=np.float32)

        return np.concatenate([static, history, affinity], axis=1).astype(np.float32)

    def _encode_temporal_batch(
        self,
        weeks: np.ndarray,
        weekdays: np.ndarray,
        hours: np.ndarray
    ) -> np.ndarray:
        """Encode temporal context [B, 64]."""
        batch_size = len(weeks)

        week_of_year = np.clip(weeks % 100, 1, 52).astype(np.int32)
        week_embed = self.week_matrix[week_of_year]
        weekday_embed = self.weekday_matrix[np.clip(weekdays, 0, 7).astype(np.int32)]
        hour_embed = self.hour_matrix[np.clip(hours, 0, 24).astype(np.int32)]

        is_holiday = np.isin(week_of_year, list(self.holiday_weeks))
        holiday_pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32) * 0.5
        non_holiday_pattern = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32) * 0.5
        holiday_embed = np.where(is_holiday[:, None], holiday_pattern, non_holiday_pattern)

        month = np.clip(((week_of_year - 1) // 4) + 1, 1, 12)
        season_idx = np.where(month <= 2, 3,
                     np.where(month <= 5, 0,
                     np.where(month <= 8, 1,
                     np.where(month <= 11, 2, 3))))
        season_embed = self.season_matrix[season_idx]

        trend_value = (weeks - self.min_week) / max(self.max_week - self.min_week, 1)
        trend_embed = self._fourier_encode(trend_value, 8)
        recency_embed = self._fourier_encode(np.full(batch_size, 0.5), 8)

        return np.concatenate([
            week_embed, weekday_embed, hour_embed,
            holiday_embed, season_embed, trend_embed, recency_embed
        ], axis=1).astype(np.float32)

    def _encode_basket_batch(
        self,
        products_list: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode previous basket products [B, S, 256]."""
        batch_size = len(products_list)
        max_len = self.max_basket_len

        embeddings = np.zeros((batch_size, max_len, 256), dtype=np.float32)
        mask = np.zeros((batch_size, max_len), dtype=np.float32)

        for i, products in enumerate(products_list):
            seq_len = min(len(products), max_len)
            for j, prod in enumerate(products[:seq_len]):
                prod_idx = self.product_to_idx.get(prod, 0)
                if prod_idx > 0 and prod_idx < len(self.product_embeddings):
                    embeddings[i, j] = self.product_embeddings[prod_idx]
            mask[i, :seq_len] = 1.0

        return embeddings, mask

    def _fourier_encode(self, values: np.ndarray, dim: int) -> np.ndarray:
        """Fourier position encoding."""
        batch_size = len(values)
        features = np.zeros((batch_size, dim), dtype=np.float32)
        for i in range(dim // 2):
            freq = (i + 1) * np.pi
            features[:, 2 * i] = np.sin(freq * values)
            features[:, 2 * i + 1] = np.cos(freq * values)
        return features


class StoreVisitDataLoader:
    """DataLoader for store visit prediction."""

    def __init__(
        self,
        dataset: StoreVisitDataset,
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._build_batches()

    def _build_batches(self):
        """Build batch indices."""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        self.batch_indices = [
            indices[start:start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self.batch_indices)

    def __iter__(self) -> Iterator[StoreVisitBatch]:
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

        for batch_idx in self.batch_indices:
            yield self.dataset.get_batch(batch_idx.tolist())

    def reset(self):
        """Reset for new epoch."""
        self._build_batches()


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent

    print("Testing StoreVisitDataset...")
    try:
        dataset = StoreVisitDataset(project_root, split='train', include_basket=True)
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of stores: {dataset.num_stores}")

        batch = dataset.get_batch([0, 1, 2, 3])
        print(f"\nBatch:")
        print(f"  - Customer context: {batch.customer_context.shape}")
        print(f"  - Temporal context: {batch.temporal_context.shape}")
        print(f"  - Previous store idx: {batch.previous_store_idx}")
        print(f"  - Target store idx: {batch.target_store_idx}")
        if batch.previous_basket_embeddings is not None:
            print(f"  - Basket embeddings: {batch.previous_basket_embeddings.shape}")
            print(f"  - Basket mask: {batch.previous_basket_mask.shape}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
