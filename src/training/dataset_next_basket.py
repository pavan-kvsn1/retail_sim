"""
Next-Basket Prediction Dataset for World Model Training.

This dataset supports autoregressive next-basket prediction:
- Input: Customer history + basket at time t + context for t+1
- Target: Full basket at time t+1

Key differences from masked prediction:
1. No masking - we see the full input basket
2. Target is the ENTIRE next basket (multi-label classification)
3. Temporal context is for t+1 (when we're predicting)

Reference: RetailSim World Model Design
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
class NextBasketBatch:
    """Container for a batch of next-basket prediction samples."""
    # Context tensors
    customer_context: np.ndarray      # T1: [B, 192]
    temporal_context: np.ndarray      # T3: [B, 64] - for t+1
    store_context: np.ndarray         # T5: [B, 96] - for t+1
    trip_context: np.ndarray          # T6: [B, 48]

    # Input basket at time t (full sequence, no masking)
    input_product_ids: np.ndarray     # [B, S_in] - product token IDs
    input_embeddings: np.ndarray      # [B, S_in, 256] - product embeddings
    input_price_features: np.ndarray  # [B, S_in, 64]
    input_attention_mask: np.ndarray  # [B, S_in]
    input_lengths: np.ndarray         # [B]

    # Target: next basket as multi-hot vector
    target_products: np.ndarray       # [B, V] - multi-hot over vocabulary
    target_product_ids: np.ndarray    # [B, S_out] - padded sequence (for teacher forcing)
    target_lengths: np.ndarray        # [B] - actual target sequence lengths

    # Auxiliary labels (for target basket)
    auxiliary_labels: Dict[str, np.ndarray] = field(default_factory=dict)

    # Metadata
    customer_ids: Optional[np.ndarray] = None
    input_basket_ids: Optional[np.ndarray] = None
    target_basket_ids: Optional[np.ndarray] = None
    week_gaps: Optional[np.ndarray] = None

    @property
    def batch_size(self) -> int:
        return self.customer_context.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.target_products.shape[1]


class NextBasketDataset:
    """
    Dataset for next-basket prediction (autoregressive world model).

    Each sample is a (basket_t, basket_t+1) pair where we predict
    the full contents of basket_t+1 given basket_t and history.
    """

    MISSION_TYPE_VOCAB = {'Top Up': 1, 'Full Shop': 2, 'Small Shop': 3, 'Emergency': 4}
    MISSION_FOCUS_VOCAB = {'Fresh': 1, 'Grocery': 2, 'Mixed': 3, 'Nonfood': 4, 'General': 5}
    PRICE_SENS_VOCAB = {'LA': 1, 'MM': 2, 'UM': 3}
    BASKET_SIZE_VOCAB = {'S': 1, 'M': 2, 'L': 3}

    def __init__(
        self,
        project_root: Path,
        split: str = 'train',
        max_input_len: int = 50,
        max_target_len: int = 50,
        max_history_baskets: int = 10,
        vocabulary: Optional[Dict] = None,
    ):
        """
        Initialize next-basket dataset.

        Args:
            project_root: Path to retail_sim project root
            split: One of 'train', 'validation', 'test'
            max_input_len: Maximum sequence length for input basket
            max_target_len: Maximum sequence length for target basket
            max_history_baskets: Maximum history baskets to encode
            vocabulary: Optional shared vocabulary dict from training set.
                        If None, builds vocabulary from this split's data.
                        Should contain: product_to_idx, idx_to_product, product_list,
                        n_products, vocab_size, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
        """
        self.project_root = Path(project_root)
        self.split = split
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.max_history_baskets = max_history_baskets

        # Load samples
        self._load_samples()

        # Load tensor cache (embeddings)
        self._load_tensor_cache()

        # Load transactions for basket attributes
        self._load_basket_attributes()

        # Build or use shared product vocabulary
        if vocabulary is not None:
            self._use_shared_vocabulary(vocabulary)
        else:
            self._build_vocabulary()

        # Initialize temporal encoder
        self._init_temporal_encoder()

        # Build bucket indices
        self._build_bucket_indices()

        logger.info(f"NextBasketDataset initialized: {len(self)} samples ({split})")

    def _load_samples(self):
        """Load next-basket samples."""
        samples_path = self.project_root / 'data' / 'prepared' / f'{self.split}_next_basket.parquet'

        if not samples_path.exists():
            raise FileNotFoundError(
                f"Next-basket samples not found at {samples_path}. "
                f"Run: python src/data_preparation/stage4_next_basket_samples.py"
            )

        self.samples = pd.read_parquet(samples_path)

        # Deserialize list columns
        for col in ['input_products', 'target_products']:
            if col in self.samples.columns:
                self.samples[col] = self.samples[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )

        logger.info(f"Loaded {len(self.samples):,} {self.split} samples")

        # Get week range
        self.min_week = self.samples['target_week'].min()
        self.max_week = self.samples['target_week'].max()

    def _load_tensor_cache(self):
        """Load pre-computed embeddings."""
        cache_dir = self.project_root / 'data' / 'tensor_cache'

        # Product embeddings
        product_path = cache_dir / 'product_embeddings.npy'
        if product_path.exists():
            self.product_embeddings = np.load(product_path)
            logger.info(f"Loaded product embeddings: {self.product_embeddings.shape}")
        else:
            logger.warning("Product embeddings not found, using random")
            self.product_embeddings = np.random.randn(5000, 256).astype(np.float32) * 0.1

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

        # Store embeddings
        store_path = cache_dir / 'store_embeddings.npy'
        if store_path.exists():
            self.store_embeddings = np.load(store_path)
        else:
            self.store_embeddings = np.random.randn(800, 96).astype(np.float32) * 0.1

        # Vocabulary mapping
        vocab_path = cache_dir / 'vocab.json'
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}

        # Price features lookup
        price_path = cache_dir / 'price_features_indexed.parquet'
        if price_path.exists():
            logger.info("Loading price features (this may take a minute)...")
            price_df = pd.read_parquet(price_path)

            # Feature columns (64 total)
            feature_cols = (
                [f'fourier_{i}' for i in range(24)] +
                [f'log_{i}' for i in range(16)] +
                [f'relative_{i}' for i in range(16)] +
                [f'velocity_{i}' for i in range(8)]
            )

            # Create lookup dict efficiently (much faster than iterrows)
            product_ids = price_df['product_id'].values
            store_ids = price_df['store_id'].values
            weeks = price_df['week'].astype(int).values
            values = price_df[feature_cols].values.astype(np.float32)

            self.price_features_lookup = {
                (product_ids[i], store_ids[i], weeks[i]): values[i]
                for i in range(len(price_df))
            }

            logger.info(f"Loaded price features for {len(self.price_features_lookup):,} (product, store, week) combinations")
            del price_df  # Free memory
        else:
            logger.warning("Price features not found, using random")
            self.price_features_lookup = None

    def _load_basket_attributes(self):
        """Load basket attributes for auxiliary tasks."""
        # Try parquet first (smaller, faster), fall back to transactions.csv
        parquet_path = self.project_root / 'data' / 'prepared' / 'basket_attributes.parquet'
        transactions_path = self.project_root / 'raw_data' / 'transactions.csv'

        if parquet_path.exists():
            logger.info("Loading basket attributes from parquet...")
            df = pd.read_parquet(parquet_path)
            self.basket_attrs = df.set_index('BASKET_ID').to_dict('index')
            logger.info(f"Loaded attributes for {len(self.basket_attrs):,} baskets")
        elif transactions_path.exists():
            logger.info("Loading basket attributes from transactions.csv...")
            df = pd.read_csv(
                transactions_path,
                usecols=['BASKET_ID', 'BASKET_TYPE', 'BASKET_DOMINANT_MISSION',
                         'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE']
            )
            basket_attrs = df.groupby('BASKET_ID').first()
            self.basket_attrs = basket_attrs.to_dict('index')
            logger.info(f"Loaded attributes for {len(self.basket_attrs):,} baskets")
        else:
            logger.warning("Basket attributes not found, auxiliary labels unavailable")
            self.basket_attrs = {}

    def _build_vocabulary(self):
        """Build product vocabulary from samples (input + target products only)."""
        logger.info("Building vocabulary...")
        all_products = set()

        # Build vocabulary from input and target products only
        # (history_products was removed to save memory during data generation)
        # Use vectorized approach instead of iterrows for speed
        for products_list in self.samples['input_products'].values:
            all_products.update(products_list)
        for products_list in self.samples['target_products'].values:
            all_products.update(products_list)

        self.product_list = sorted(all_products)
        self.product_to_idx = {p: i + 1 for i, p in enumerate(self.product_list)}  # 0 = PAD
        self.idx_to_product = {i + 1: p for i, p in enumerate(self.product_list)}
        self.n_products = len(self.product_list)

        # Special tokens
        self.PAD_TOKEN = 0
        self.EOS_TOKEN = self.n_products + 1
        self.BOS_TOKEN = self.n_products + 2
        self.vocab_size = self.n_products + 3  # products + PAD + EOS + BOS

        logger.info(f"Vocabulary: {self.n_products} products, vocab_size={self.vocab_size}")

    def _use_shared_vocabulary(self, vocabulary: Dict):
        """Use a shared vocabulary from another dataset (e.g., training set)."""
        self.product_to_idx = vocabulary['product_to_idx']
        self.idx_to_product = vocabulary['idx_to_product']
        self.product_list = vocabulary['product_list']
        self.n_products = vocabulary['n_products']
        self.vocab_size = vocabulary['vocab_size']
        self.PAD_TOKEN = vocabulary['PAD_TOKEN']
        self.EOS_TOKEN = vocabulary['EOS_TOKEN']
        self.BOS_TOKEN = vocabulary['BOS_TOKEN']
        logger.info(f"Using shared vocabulary: {self.n_products} products, vocab_size={self.vocab_size}")

    def get_vocabulary(self) -> Dict:
        """Export vocabulary for sharing with other datasets."""
        return {
            'product_to_idx': self.product_to_idx,
            'idx_to_product': self.idx_to_product,
            'product_list': self.product_list,
            'n_products': self.n_products,
            'vocab_size': self.vocab_size,
            'PAD_TOKEN': self.PAD_TOKEN,
            'EOS_TOKEN': self.EOS_TOKEN,
            'BOS_TOKEN': self.BOS_TOKEN,
        }

    def _init_temporal_encoder(self):
        """Initialize temporal encoding."""
        np.random.seed(42)

        self.week_matrix = np.random.randn(53, 16).astype(np.float32) * 0.1
        self.weekday_matrix = np.random.randn(8, 8).astype(np.float32) * 0.1
        self.hour_matrix = np.random.randn(25, 8).astype(np.float32) * 0.1
        self.season_matrix = np.random.randn(4, 8).astype(np.float32) * 0.1

        self.holiday_weeks = {51, 52, 1, 13, 14, 15, 26, 27, 28, 29, 30, 31, 32, 33, 47}

        # Trip context
        self.mission_type_matrix = np.random.randn(5, 16).astype(np.float32) * 0.1
        self.mission_focus_matrix = np.random.randn(6, 16).astype(np.float32) * 0.1
        self.price_sens_matrix = np.random.randn(4, 8).astype(np.float32) * 0.1
        self.basket_size_matrix = np.random.randn(4, 8).astype(np.float32) * 0.1

    def _build_bucket_indices(self):
        """Build indices for bucket-based batching."""
        self.bucket_indices = defaultdict(list)

        for idx in range(len(self.samples)):
            row = self.samples.iloc[idx]
            bucket = row.get('bucket', 1)
            self.bucket_indices[bucket].append(idx)

        for bucket in self.bucket_indices:
            self.bucket_indices[bucket] = np.array(self.bucket_indices[bucket])

        logger.info(f"Bucket distribution: {dict((k, len(v)) for k, v in self.bucket_indices.items())}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, indices: List[int]) -> NextBasketBatch:
        """
        Get a batch of next-basket prediction samples.

        Args:
            indices: Sample indices

        Returns:
            NextBasketBatch with all tensors
        """
        batch_size = len(indices)
        batch_samples = self.samples.iloc[indices]

        # Extract IDs
        customer_ids = batch_samples['customer_id'].values
        input_basket_ids = batch_samples['input_basket_id'].values
        target_basket_ids = batch_samples['target_basket_id'].values
        target_store_ids = batch_samples['target_store_id'].values
        target_weeks = batch_samples['target_week'].values
        target_weekdays = batch_samples['target_weekday'].values
        target_hours = batch_samples['target_hour'].values
        week_gaps = batch_samples['week_gap'].values

        # T1: Customer Context [B, 192]
        customer_context = self._encode_customer_batch(customer_ids)

        # T3: Temporal Context [B, 64] - for target time t+1
        temporal_context = self._encode_temporal_batch(target_weeks, target_weekdays, target_hours)

        # T5: Store Context [B, 96] - for target store
        store_context = self._encode_store_batch(target_store_ids)

        # T6: Trip Context [B, 48]
        trip_context, auxiliary_labels = self._encode_trip_batch(target_basket_ids)

        # Input basket at time t (use input store/week for price lookup)
        input_products_list = batch_samples['input_products'].tolist()
        input_store_ids = batch_samples['input_store_id'].values
        input_weeks = batch_samples['input_week'].values
        (input_product_ids, input_embeddings, input_price_features,
         input_attention_mask, input_lengths) = self._encode_product_sequence(
            input_products_list, self.max_input_len, input_store_ids, input_weeks
        )

        # Target basket (multi-hot and sequence)
        target_products_list = batch_samples['target_products'].tolist()
        target_multihot = self._encode_multihot(target_products_list)
        target_product_ids, target_lengths = self._encode_target_sequence(
            target_products_list, self.max_target_len
        )

        return NextBasketBatch(
            customer_context=customer_context,
            temporal_context=temporal_context,
            store_context=store_context,
            trip_context=trip_context,
            input_product_ids=input_product_ids,
            input_embeddings=input_embeddings,
            input_price_features=input_price_features,
            input_attention_mask=input_attention_mask,
            input_lengths=input_lengths,
            target_products=target_multihot,
            target_product_ids=target_product_ids,
            target_lengths=target_lengths,
            auxiliary_labels=auxiliary_labels,
            customer_ids=customer_ids,
            input_basket_ids=input_basket_ids,
            target_basket_ids=target_basket_ids,
            week_gaps=week_gaps,
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

        # Customer affinity placeholder - zeros for now (no real affinity data available)
        # TODO: Replace with real customer-category affinity scores if available
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

    def _encode_store_batch(self, store_ids: np.ndarray) -> np.ndarray:
        """Encode store context [B, 96]."""
        store_indices = np.array([
            self.vocab.get('store_to_idx', {}).get(sid, 0)
            for sid in store_ids
        ])
        return self.store_embeddings[np.clip(store_indices, 0, len(self.store_embeddings) - 1)]

    def _encode_trip_batch(
        self,
        basket_ids: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Encode trip context [B, 48] and auxiliary labels."""
        batch_size = len(basket_ids)

        type_codes = []
        focus_codes = []
        sens_codes = []
        size_codes = []

        for bid in basket_ids:
            attrs = self.basket_attrs.get(bid, {})
            type_codes.append(self.MISSION_TYPE_VOCAB.get(str(attrs.get('BASKET_TYPE', '')), 0))
            focus_codes.append(self.MISSION_FOCUS_VOCAB.get(str(attrs.get('BASKET_DOMINANT_MISSION', '')), 0))
            sens_codes.append(self.PRICE_SENS_VOCAB.get(str(attrs.get('BASKET_PRICE_SENSITIVITY', '')), 0))
            size_codes.append(self.BASKET_SIZE_VOCAB.get(str(attrs.get('BASKET_SIZE', '')), 0))

        type_codes = np.array(type_codes)
        focus_codes = np.array(focus_codes)
        sens_codes = np.array(sens_codes)
        size_codes = np.array(size_codes)

        type_embed = self.mission_type_matrix[type_codes]
        focus_embed = self.mission_focus_matrix[focus_codes]
        sens_embed = self.price_sens_matrix[sens_codes]
        size_embed = self.basket_size_matrix[size_codes]

        trip_context = np.concatenate([type_embed, focus_embed, sens_embed, size_embed], axis=1)

        auxiliary_labels = {
            'mission_type': type_codes,
            'mission_focus': focus_codes,
            'price_sensitivity': sens_codes,
            'basket_size': size_codes,
        }

        return trip_context.astype(np.float32), auxiliary_labels

    def _encode_product_sequence(
        self,
        products_list: List[List[str]],
        max_len: int,
        store_ids: np.ndarray = None,
        weeks: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode product sequences with embeddings and price features."""
        batch_size = len(products_list)

        product_ids = np.zeros((batch_size, max_len), dtype=np.int32)
        embeddings = np.zeros((batch_size, max_len, 256), dtype=np.float32)
        price_features = np.zeros((batch_size, max_len, 64), dtype=np.float32)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)
        lengths = np.zeros(batch_size, dtype=np.int32)

        for i, products in enumerate(products_list):
            seq_len = min(len(products), max_len - 1)  # -1 for EOS
            store_id = store_ids[i] if store_ids is not None else None
            week = int(weeks[i]) if weeks is not None else None

            for j, prod in enumerate(products[:seq_len]):
                prod_idx = self.product_to_idx.get(prod, 0)
                product_ids[i, j] = prod_idx

                if prod_idx > 0 and prod_idx < len(self.product_embeddings):
                    embeddings[i, j] = self.product_embeddings[prod_idx]

                # Look up real price features if available
                if self.price_features_lookup is not None and store_id and week:
                    key = (prod, store_id, week)
                    if key in self.price_features_lookup:
                        price_features[i, j] = self.price_features_lookup[key]
                    else:
                        # Fallback to random if not found
                        price_features[i, j] = np.random.randn(64).astype(np.float32) * 0.1
                else:
                    price_features[i, j] = np.random.randn(64).astype(np.float32) * 0.1

            # EOS token
            product_ids[i, seq_len] = self.EOS_TOKEN
            attention_mask[i, :seq_len + 1] = 1
            lengths[i] = seq_len + 1

        return product_ids, embeddings, price_features, attention_mask, lengths

    def _encode_multihot(self, products_list: List[List[str]]) -> np.ndarray:
        """Encode target basket as multi-hot vector [B, vocab_size]."""
        batch_size = len(products_list)
        multihot = np.zeros((batch_size, self.vocab_size), dtype=np.float32)

        for i, products in enumerate(products_list):
            for prod in products:
                prod_idx = self.product_to_idx.get(prod, 0)
                if prod_idx > 0:
                    multihot[i, prod_idx] = 1.0

        return multihot

    def _encode_target_sequence(
        self,
        products_list: List[List[str]],
        max_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode target as sequence (for teacher forcing)."""
        batch_size = len(products_list)

        product_ids = np.zeros((batch_size, max_len), dtype=np.int32)
        lengths = np.zeros(batch_size, dtype=np.int32)

        for i, products in enumerate(products_list):
            seq_len = min(len(products), max_len - 1)

            for j, prod in enumerate(products[:seq_len]):
                prod_idx = self.product_to_idx.get(prod, 0)
                product_ids[i, j] = prod_idx

            product_ids[i, seq_len] = self.EOS_TOKEN
            lengths[i] = seq_len + 1

        return product_ids, lengths

    def _fourier_encode(self, values: np.ndarray, dim: int) -> np.ndarray:
        """Fourier position encoding."""
        batch_size = len(values)
        features = np.zeros((batch_size, dim), dtype=np.float32)
        for i in range(dim // 2):
            freq = (i + 1) * np.pi
            features[:, 2 * i] = np.sin(freq * values)
            features[:, 2 * i + 1] = np.cos(freq * values)
        return features


class NextBasketDataLoader:
    """DataLoader with bucket batching for next-basket prediction."""

    def __init__(
        self,
        dataset: NextBasketDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        bucket_batching: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_batching = bucket_batching

        self._build_batches()

    def _build_batches(self):
        """Pre-build batch indices."""
        if self.bucket_batching:
            self.batch_indices = []
            for bucket, indices in self.dataset.bucket_indices.items():
                indices = indices.copy()
                if self.shuffle:
                    np.random.shuffle(indices)
                for start in range(0, len(indices), self.batch_size):
                    batch = indices[start:start + self.batch_size]
                    self.batch_indices.append(batch)
        else:
            indices = np.arange(len(self.dataset))
            if self.shuffle:
                np.random.shuffle(indices)
            self.batch_indices = [
                indices[start:start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            ]

    def __len__(self) -> int:
        return len(self.batch_indices)

    def __iter__(self) -> Iterator[NextBasketBatch]:
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

        for batch_idx in self.batch_indices:
            yield self.dataset.get_batch(batch_idx.tolist())

    def reset(self):
        """Reset for new epoch."""
        self._build_batches()


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent

    print("Testing NextBasketDataset...")
    try:
        dataset = NextBasketDataset(project_root, split='train')
        print(f"Dataset size: {len(dataset)}")
        print(f"Vocabulary size: {dataset.vocab_size}")

        batch = dataset.get_batch([0, 1, 2])
        print(f"\nBatch:")
        print(f"  - Customer context: {batch.customer_context.shape}")
        print(f"  - Input embeddings: {batch.input_embeddings.shape}")
        print(f"  - Target multi-hot: {batch.target_products.shape}")
        print(f"  - Target products sum: {batch.target_products.sum(axis=1)}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run stage4_next_basket_samples.py first to generate data.")
