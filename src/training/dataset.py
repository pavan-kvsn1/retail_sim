"""
World Model Dataset and DataLoader for RetailSim Training.

Integrates:
- Temporal split samples (train/validation/test)
- Tensor cache (pre-computed embeddings)
- Bucket-based batching for efficient padding
- Day/hour level evaluation for validation/test

Key features:
- In-memory loading for fast training (~10GB)
- Vectorized batch encoding
- Bucket batching to minimize padding waste
- All 4 auxiliary tasks enabled by default
- Price features from Fourier-encoded parquet

Reference: RetailSim_Data_Pipeline_and_World_Model_Design.md Section 4.7, 5.6
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Union
from dataclasses import dataclass, field
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class WorldModelBatch:
    """Container for a batch of World Model training samples."""
    # Dense context tensors (per design doc)
    customer_context: np.ndarray      # T1: [B, 192] (segment 64 + history 96 + affinity 32)
    temporal_context: np.ndarray      # T3: [B, 64]
    store_context: np.ndarray         # T5: [B, 96]
    trip_context: np.ndarray          # T6: [B, 48]

    # Sequence tensors
    product_embeddings: np.ndarray    # T2: [B, S, 256] GraphSage embeddings
    product_token_ids: np.ndarray     # [B, S] for output prediction
    price_features: np.ndarray        # T4: [B, S, 64] Fourier-encoded prices
    attention_mask: np.ndarray        # [B, S]
    sequence_lengths: np.ndarray      # [B]

    # Labels for auxiliary prediction
    auxiliary_labels: Dict[str, np.ndarray] = field(default_factory=dict)

    # Masked LM targets (for training)
    masked_positions: Optional[np.ndarray] = None
    masked_targets: Optional[np.ndarray] = None

    # Metadata for evaluation
    basket_ids: Optional[np.ndarray] = None
    customer_ids: Optional[np.ndarray] = None
    shop_dates: Optional[np.ndarray] = None  # For day-level evaluation
    shop_hours: Optional[np.ndarray] = None  # For hour-level evaluation
    weeks: Optional[np.ndarray] = None

    @property
    def batch_size(self) -> int:
        return self.customer_context.shape[0]

    @property
    def max_seq_len(self) -> int:
        return self.product_embeddings.shape[1]

    def get_dense_context(self) -> np.ndarray:
        """Concatenate all dense context tensors [B, 400]."""
        return np.concatenate([
            self.customer_context,  # 192
            self.temporal_context,  # 64
            self.store_context,     # 96
            self.trip_context       # 48
        ], axis=1)

    def get_sequence_features(self) -> np.ndarray:
        """Concatenate product and price features [B, S, 320]."""
        return np.concatenate([
            self.product_embeddings,  # 256
            self.price_features       # 64
        ], axis=2)


class WorldModelDataset:
    """
    Dataset for World Model training with temporal splits and tensor caching.

    Loads pre-computed samples and embeddings for efficient training.
    Supports bucket-based batching and auxiliary task labels.
    """

    # Auxiliary task vocabularies
    MISSION_TYPE_VOCAB = {'Top Up': 1, 'Full Shop': 2, 'Small Shop': 3, 'Emergency': 4}
    MISSION_FOCUS_VOCAB = {'Fresh': 1, 'Grocery': 2, 'Mixed': 3, 'Nonfood': 4, 'General': 5}
    PRICE_SENS_VOCAB = {'LA': 1, 'MM': 2, 'UM': 3}
    BASKET_SIZE_VOCAB = {'S': 1, 'M': 2, 'L': 3}

    def __init__(
        self,
        project_root: Path,
        split: str = 'train',
        max_seq_len: int = 50,
        mask_prob: float = 0.15,
        enable_auxiliary_tasks: bool = True,
        load_transactions: bool = True,
    ):
        """
        Initialize World Model dataset.

        Args:
            project_root: Path to retail_sim project root
            split: One of 'train', 'validation', 'test'
            max_seq_len: Maximum sequence length for product items
            mask_prob: Probability of masking tokens (for MLM training)
            enable_auxiliary_tasks: Enable all 4 auxiliary prediction tasks
            load_transactions: Load raw transactions for product sequences
        """
        self.project_root = Path(project_root)
        self.split = split
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.enable_auxiliary_tasks = enable_auxiliary_tasks

        # Load samples
        self._load_samples()

        # Load tensor cache
        self._load_tensor_cache()

        # Load transactions for product sequences
        if load_transactions:
            self._load_transactions()

        # Build bucket indices for efficient batching
        self._build_bucket_indices()

        # Initialize temporal encoder
        self._init_temporal_encoder()

        logger.info(f"WorldModelDataset initialized: {len(self)} samples ({split})")

    def _load_samples(self):
        """Load split samples from parquet."""
        samples_path = self.project_root / 'data' / 'prepared' / f'{self.split}_samples.parquet'

        if not samples_path.exists():
            # Fall back to temporal_metadata
            logger.warning(f"Split samples not found, using temporal_metadata")
            tm_path = self.project_root / 'data' / 'prepared' / 'temporal_metadata.parquet'
            all_samples = pd.read_parquet(tm_path)
            self.samples = all_samples[all_samples['split'] == self.split].copy()
        else:
            self.samples = pd.read_parquet(samples_path)

        logger.info(f"Loaded {len(self.samples):,} {self.split} samples")

        # Get week range for temporal encoding
        self.min_week = self.samples['week'].min() if 'week' in self.samples.columns else 1
        self.max_week = self.samples['week'].max() if 'week' in self.samples.columns else 117

    def _load_tensor_cache(self):
        """Load pre-computed embedding tensors."""
        cache_dir = self.project_root / 'data' / 'tensor_cache'

        if not cache_dir.exists():
            logger.warning("Tensor cache not found, using random embeddings")
            self._init_random_embeddings()
            return

        # Load vocabulary
        vocab_path = cache_dir / 'vocab.json'
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}

        # Load embedding tensors
        product_path = cache_dir / 'product_embeddings.npy'
        if product_path.exists():
            self.product_embeddings = np.load(product_path)
            logger.info(f"Loaded product embeddings: {self.product_embeddings.shape}")
        else:
            self._init_random_product_embeddings()

        cust_hist_path = cache_dir / 'customer_history_embeddings.npy'
        if cust_hist_path.exists():
            self.customer_history = np.load(cust_hist_path)
            logger.info(f"Loaded customer history: {self.customer_history.shape}")
        else:
            self.customer_history = np.random.randn(100000, 160).astype(np.float32) * 0.1

        cust_static_path = cache_dir / 'customer_static_embeddings.npy'
        if cust_static_path.exists():
            self.customer_static = np.load(cust_static_path)
            logger.info(f"Loaded customer static: {self.customer_static.shape}")
        else:
            self.customer_static = np.random.randn(100000, 160).astype(np.float32) * 0.1

        store_path = cache_dir / 'store_embeddings.npy'
        if store_path.exists():
            self.store_embeddings = np.load(store_path)
            logger.info(f"Loaded store embeddings: {self.store_embeddings.shape}")
        else:
            self.store_embeddings = np.random.randn(800, 96).astype(np.float32) * 0.1

        # Price features loaded lazily in batch encoding
        self._price_df = None
        self._price_lookup = None

    def _init_random_embeddings(self):
        """Initialize random embeddings when cache not available."""
        logger.info("Initializing random embeddings (tensor cache not found)")
        self.vocab = {
            'product_to_idx': {},
            'customer_to_idx': {},
            'store_to_idx': {},
            'dimensions': {
                'product': 256,
                'customer_history': 160,
                'customer_static': 160,
                'store': 96,
                'price': 64,
            }
        }
        self.product_embeddings = np.random.randn(5000, 256).astype(np.float32) * 0.1
        self.customer_history = np.random.randn(100000, 160).astype(np.float32) * 0.1
        self.customer_static = np.random.randn(100000, 160).astype(np.float32) * 0.1
        self.store_embeddings = np.random.randn(800, 96).astype(np.float32) * 0.1

    def _init_random_product_embeddings(self):
        """Initialize random product embeddings."""
        n_products = 5000
        self.product_embeddings = np.random.randn(n_products + 1, 256).astype(np.float32) * 0.1
        self.product_embeddings[0] = 0  # PAD

    def _load_transactions(self):
        """Load raw transactions for product sequence extraction."""
        transactions_path = self.project_root / 'raw_data' / 'transactions.csv'

        logger.info("Loading transactions for product sequences...")

        # Load with efficient dtypes
        dtype = {
            'BASKET_ID': 'int64',
            'PROD_CODE': 'category',
            'CUST_CODE': 'category',
            'STORE_CODE': 'category',
            'SHOP_WEEK': 'int32',
            'BASKET_TYPE': 'category',
            'BASKET_DOMINANT_MISSION': 'category',
            'BASKET_PRICE_SENSITIVITY': 'category',
            'BASKET_SIZE': 'category',
        }

        usecols = ['BASKET_ID', 'PROD_CODE', 'CUST_CODE', 'STORE_CODE', 'SHOP_WEEK',
                   'BASKET_TYPE', 'BASKET_DOMINANT_MISSION', 'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE']

        self.transactions = pd.read_csv(
            transactions_path,
            usecols=usecols,
            dtype=dtype
        )

        logger.info(f"Loaded {len(self.transactions):,} transactions")

        # Build basket -> products mapping
        logger.info("Building basket product index...")
        self.basket_products = self.transactions.groupby('BASKET_ID')['PROD_CODE'].apply(list).to_dict()

        # Build basket -> attributes mapping (for auxiliary tasks)
        basket_attrs = self.transactions.groupby('BASKET_ID').first()
        self.basket_attrs = basket_attrs[['BASKET_TYPE', 'BASKET_DOMINANT_MISSION',
                                          'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE']].to_dict('index')

        # Build product vocabulary
        self.product_categories = self.transactions['PROD_CODE'].cat.categories.tolist()
        self.product_to_idx = {prod: idx + 1 for idx, prod in enumerate(self.product_categories)}
        self.n_products = len(self.product_categories)

        logger.info(f"Indexed {len(self.basket_products):,} baskets, {self.n_products:,} products")

    def _build_bucket_indices(self):
        """Build indices for bucket-based batching."""
        self.bucket_indices = defaultdict(list)

        for idx, row in self.samples.iterrows():
            bucket = row['bucket'] if 'bucket' in row else 1
            self.bucket_indices[bucket].append(idx)

        # Convert to numpy arrays
        for bucket in self.bucket_indices:
            self.bucket_indices[bucket] = np.array(self.bucket_indices[bucket])

        logger.info(f"Bucket distribution: {dict((k, len(v)) for k, v in self.bucket_indices.items())}")

    def _init_temporal_encoder(self):
        """Initialize temporal encoding matrices."""
        np.random.seed(42)

        # Week embeddings (53 weeks)
        self.week_matrix = np.random.randn(53, 16).astype(np.float32) * 0.1

        # Weekday embeddings (8 including 0)
        self.weekday_matrix = np.random.randn(8, 8).astype(np.float32) * 0.1

        # Hour embeddings (25 including 0)
        self.hour_matrix = np.random.randn(25, 8).astype(np.float32) * 0.1

        # Season embeddings (4)
        self.season_matrix = np.random.randn(4, 8).astype(np.float32) * 0.1

        # Holiday weeks
        self.holiday_weeks = {51, 52, 1, 13, 14, 15, 26, 27, 28, 29, 30, 31, 32, 33, 47}

        # Trip context embeddings
        self.mission_type_matrix = np.random.randn(5, 16).astype(np.float32) * 0.1
        self.mission_focus_matrix = np.random.randn(6, 16).astype(np.float32) * 0.1
        self.price_sens_matrix = np.random.randn(4, 8).astype(np.float32) * 0.1
        self.basket_size_matrix = np.random.randn(4, 8).astype(np.float32) * 0.1

        # Positional encoding for sequences
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, 256, 2) * (-np.log(10000.0) / 256))
        self.positional_encoding = np.zeros((self.max_seq_len, 256), dtype=np.float32)
        self.positional_encoding[:, 0::2] = np.sin(position * div_term)
        self.positional_encoding[:, 1::2] = np.cos(position * div_term)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get single sample (for debugging)."""
        batch = self.get_batch([idx], apply_masking=False)
        return {
            'basket_id': batch.basket_ids[0] if batch.basket_ids is not None else None,
            'customer_context': batch.customer_context[0],
            'temporal_context': batch.temporal_context[0],
            'store_context': batch.store_context[0],
            'trip_context': batch.trip_context[0],
            'product_embeddings': batch.product_embeddings[0],
            'price_features': batch.price_features[0],
            'sequence_length': batch.sequence_lengths[0],
        }

    def get_batch(
        self,
        indices: List[int],
        apply_masking: bool = True
    ) -> WorldModelBatch:
        """
        Get a batch of samples using vectorized operations.

        Args:
            indices: Sample indices to include
            apply_masking: Apply BERT-style masking for training

        Returns:
            WorldModelBatch with all tensors
        """
        batch_size = len(indices)
        batch_samples = self.samples.iloc[indices]

        # Extract metadata
        basket_ids = batch_samples['basket_id'].values
        customer_ids = batch_samples['customer_id'].values
        store_ids = batch_samples['store_id'].values

        # Week info for temporal encoding
        if 'week_original' in batch_samples.columns:
            weeks_original = batch_samples['week_original'].values
        else:
            weeks_original = batch_samples['week'].values

        # =============================================
        # T1: Customer Context [B, 192]
        # =============================================
        customer_context = self._encode_customer_batch(customer_ids)

        # =============================================
        # T3: Temporal Context [B, 64]
        # =============================================
        shop_weekdays = batch_samples['shop_weekday'].values if 'shop_weekday' in batch_samples.columns else np.ones(batch_size, dtype=np.int8)
        shop_hours = batch_samples['shop_hour'].values if 'shop_hour' in batch_samples.columns else np.full(batch_size, 12, dtype=np.int8)

        temporal_context = self._encode_temporal_batch(weeks_original, shop_weekdays, shop_hours)

        # =============================================
        # T5: Store Context [B, 96]
        # =============================================
        store_context = self._encode_store_batch(store_ids)

        # =============================================
        # T6: Trip Context [B, 48] + auxiliary labels
        # =============================================
        trip_context, auxiliary_labels = self._encode_trip_batch(basket_ids)

        # =============================================
        # T2: Product Sequence [B, S, 256]
        # T4: Price Features [B, S, 64]
        # =============================================
        (product_embeddings, product_token_ids, price_features,
         attention_mask, sequence_lengths) = self._encode_sequence_batch(
            basket_ids, store_ids, weeks_original
        )

        # Apply masking for training AND validation (needed to compute loss)
        masked_positions = None
        masked_targets = None
        if apply_masking:
            product_token_ids, masked_positions, masked_targets = self._apply_masking(
                product_token_ids, attention_mask
            )

        # Evaluation metadata
        shop_dates = batch_samples['shop_date'].values if 'shop_date' in batch_samples.columns else None
        weeks = batch_samples['week'].values if 'week' in batch_samples.columns else None

        return WorldModelBatch(
            customer_context=customer_context,
            temporal_context=temporal_context,
            store_context=store_context,
            trip_context=trip_context,
            product_embeddings=product_embeddings,
            product_token_ids=product_token_ids,
            price_features=price_features,
            attention_mask=attention_mask,
            sequence_lengths=sequence_lengths,
            auxiliary_labels=auxiliary_labels,
            masked_positions=masked_positions,
            masked_targets=masked_targets,
            basket_ids=basket_ids,
            customer_ids=customer_ids,
            shop_dates=shop_dates,
            shop_hours=shop_hours,
            weeks=weeks,
        )

    def _encode_customer_batch(self, customer_ids: np.ndarray) -> np.ndarray:
        """Encode customer context T1 [B, 192]."""
        batch_size = len(customer_ids)

        # Get customer indices
        customer_indices = np.array([
            self.vocab.get('customer_to_idx', {}).get(cid, 0)
            for cid in customer_ids
        ])

        # History embeddings (96d from 160d, per design doc)
        hist_dim = min(96, self.customer_history.shape[1])
        history = self.customer_history[np.clip(customer_indices, 0, len(self.customer_history) - 1), :hist_dim]
        if hist_dim < 96:
            history = np.pad(history, ((0, 0), (0, 96 - hist_dim)))

        # Static/segment embeddings (64d)
        static_dim = min(64, self.customer_static.shape[1])
        static = self.customer_static[np.clip(customer_indices, 0, len(self.customer_static) - 1), :static_dim]
        if static_dim < 64:
            static = np.pad(static, ((0, 0), (0, 64 - static_dim)))

        # Affinity placeholder (32d)
        affinity = np.random.randn(batch_size, 32).astype(np.float32) * 0.1

        return np.concatenate([static, history, affinity], axis=1).astype(np.float32)

    def _encode_temporal_batch(
        self,
        weeks: np.ndarray,
        weekdays: np.ndarray,
        hours: np.ndarray
    ) -> np.ndarray:
        """Encode temporal context T3 [B, 64]."""
        batch_size = len(weeks)

        # Week of year (from YYYYWW)
        week_of_year = np.clip(weeks % 100, 1, 52).astype(np.int32)

        # Embeddings
        week_embed = self.week_matrix[week_of_year]  # [B, 16]
        weekday_embed = self.weekday_matrix[np.clip(weekdays, 0, 7).astype(np.int32)]  # [B, 8]
        hour_embed = self.hour_matrix[np.clip(hours, 0, 24).astype(np.int32)]  # [B, 8]

        # Holiday
        is_holiday = np.isin(week_of_year, list(self.holiday_weeks))
        holiday_pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32) * 0.5
        non_holiday_pattern = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32) * 0.5
        holiday_embed = np.where(is_holiday[:, None], holiday_pattern, non_holiday_pattern)

        # Season
        month = np.clip(((week_of_year - 1) // 4) + 1, 1, 12)
        season_idx = np.where(month <= 2, 3,
                     np.where(month <= 5, 0,
                     np.where(month <= 8, 1,
                     np.where(month <= 11, 2, 3))))
        season_embed = self.season_matrix[season_idx]

        # Trend (Fourier)
        trend_value = (weeks - self.min_week) / max(self.max_week - self.min_week, 1)
        trend_embed = self._fourier_encode(trend_value, 8)

        # Recency placeholder
        recency_embed = self._fourier_encode(np.full(batch_size, 0.5), 8)

        return np.concatenate([
            week_embed, weekday_embed, hour_embed,
            holiday_embed, season_embed, trend_embed, recency_embed
        ], axis=1).astype(np.float32)

    def _encode_store_batch(self, store_ids: np.ndarray) -> np.ndarray:
        """Encode store context T5 [B, 96]."""
        store_indices = np.array([
            self.vocab.get('store_to_idx', {}).get(sid, 0)
            for sid in store_ids
        ])
        return self.store_embeddings[np.clip(store_indices, 0, len(self.store_embeddings) - 1)]

    def _encode_trip_batch(
        self,
        basket_ids: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Encode trip context T6 [B, 48] and auxiliary labels."""
        batch_size = len(basket_ids)

        # Get basket attributes
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

        # Embeddings
        type_embed = self.mission_type_matrix[type_codes]
        focus_embed = self.mission_focus_matrix[focus_codes]
        sens_embed = self.price_sens_matrix[sens_codes]
        size_embed = self.basket_size_matrix[size_codes]

        trip_context = np.concatenate([type_embed, focus_embed, sens_embed, size_embed], axis=1)

        # Auxiliary labels
        auxiliary_labels = {}
        if self.enable_auxiliary_tasks:
            auxiliary_labels = {
                'mission_type': type_codes,
                'mission_focus': focus_codes,
                'price_sensitivity': sens_codes,
                'basket_size': size_codes,
            }

        return trip_context.astype(np.float32), auxiliary_labels

    def _encode_sequence_batch(
        self,
        basket_ids: np.ndarray,
        store_ids: np.ndarray,
        weeks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode product sequence T2 and price features T4."""
        batch_size = len(basket_ids)

        product_embeddings = np.zeros((batch_size, self.max_seq_len, 256), dtype=np.float32)
        product_token_ids = np.zeros((batch_size, self.max_seq_len), dtype=np.int32)
        price_features = np.zeros((batch_size, self.max_seq_len, 64), dtype=np.float32)
        attention_mask = np.zeros((batch_size, self.max_seq_len), dtype=np.int32)
        sequence_lengths = np.zeros(batch_size, dtype=np.int32)

        for i, (bid, sid, week) in enumerate(zip(basket_ids, store_ids, weeks)):
            products = self.basket_products.get(bid, [])

            if len(products) == 0:
                continue

            seq_len = min(len(products), self.max_seq_len - 1)  # -1 for EOS

            for j, prod in enumerate(products[:seq_len]):
                prod_idx = self.product_to_idx.get(prod, 0)
                product_token_ids[i, j] = prod_idx

                # Product embedding
                if prod_idx < len(self.product_embeddings):
                    product_embeddings[i, j] = self.product_embeddings[prod_idx]

                # Price features (placeholder - would load from price_features.parquet)
                price_features[i, j] = np.random.randn(64).astype(np.float32) * 0.1

            # EOS token
            product_token_ids[i, seq_len] = self.n_products + 1  # EOS

            # Positional encoding
            product_embeddings[i, :seq_len + 1] += self.positional_encoding[:seq_len + 1]

            attention_mask[i, :seq_len + 1] = 1
            sequence_lengths[i] = seq_len + 1

        return product_embeddings, product_token_ids, price_features, attention_mask, sequence_lengths

    def _fourier_encode(self, values: np.ndarray, dim: int) -> np.ndarray:
        """Fourier position encoding."""
        batch_size = len(values)
        features = np.zeros((batch_size, dim), dtype=np.float32)
        for i in range(dim // 2):
            freq = (i + 1) * np.pi
            features[:, 2 * i] = np.sin(freq * values)
            features[:, 2 * i + 1] = np.cos(freq * values)
        return features

    def _apply_masking(
        self,
        token_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply BERT-style masking."""
        batch_size, seq_len = token_ids.shape
        masked_ids = token_ids.copy()

        max_masks = max(1, int(seq_len * self.mask_prob))
        masked_positions = np.zeros((batch_size, max_masks), dtype=np.int32)
        masked_targets = np.zeros((batch_size, max_masks), dtype=np.int32)

        MASK_TOKEN = self.n_products + 2  # After EOS

        for i in range(batch_size):
            valid_mask = (attention_mask[i] == 1) & (token_ids[i] > 0) & (token_ids[i] <= self.n_products)
            valid_positions = np.where(valid_mask)[0]

            if len(valid_positions) == 0:
                continue

            n_to_mask = min(max_masks, max(1, int(len(valid_positions) * self.mask_prob)))
            positions_to_mask = np.random.choice(valid_positions, size=n_to_mask, replace=False)

            for j, pos in enumerate(positions_to_mask):
                masked_targets[i, j] = token_ids[i, pos]
                masked_positions[i, j] = pos

                rand = np.random.random()
                if rand < 0.8:
                    masked_ids[i, pos] = MASK_TOKEN
                elif rand < 0.9:
                    masked_ids[i, pos] = np.random.randint(1, self.n_products + 1)

        return masked_ids, masked_positions, masked_targets


class WorldModelDataLoader:
    """
    DataLoader with bucket batching for efficient training.

    Batches samples from the same bucket to minimize padding waste.
    """

    def __init__(
        self,
        dataset: WorldModelDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        apply_masking: bool = True,
        bucket_batching: bool = True,
    ):
        """
        Initialize DataLoader.

        Args:
            dataset: WorldModelDataset instance
            batch_size: Samples per batch
            shuffle: Shuffle samples each epoch
            apply_masking: Apply MLM masking during training
            bucket_batching: Use bucket-based batching for efficiency
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.apply_masking = apply_masking
        self.bucket_batching = bucket_batching

        self._build_batches()

    def _build_batches(self):
        """Pre-build batch indices."""
        if self.bucket_batching:
            # Build batches from each bucket
            self.batch_indices = []
            for bucket, indices in self.dataset.bucket_indices.items():
                if self.shuffle:
                    np.random.shuffle(indices)
                for start in range(0, len(indices), self.batch_size):
                    batch = indices[start:start + self.batch_size]
                    self.batch_indices.append(batch)
        else:
            # Simple sequential batching
            indices = np.arange(len(self.dataset))
            if self.shuffle:
                np.random.shuffle(indices)
            self.batch_indices = [
                indices[start:start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            ]

    def __len__(self) -> int:
        return len(self.batch_indices)

    def __iter__(self) -> Iterator[WorldModelBatch]:
        """Iterate over batches."""
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

        for batch_idx in self.batch_indices:
            yield self.dataset.get_batch(batch_idx.tolist(), self.apply_masking)

    def reset(self):
        """Reset for new epoch."""
        self._build_batches()


class EvaluationDataLoader:
    """
    DataLoader for evaluation with day/hour level granularity.

    Groups batches by date and hour for fine-grained metric computation.
    """

    def __init__(
        self,
        dataset: WorldModelDataset,
        batch_size: int = 64,
        group_by: str = 'date',  # 'date', 'hour', or 'week'
    ):
        """
        Initialize evaluation DataLoader.

        Args:
            dataset: WorldModelDataset (validation or test split)
            batch_size: Samples per batch
            group_by: Grouping for evaluation metrics
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_by = group_by

        self._build_groups()

    def _build_groups(self):
        """Build evaluation groups."""
        if self.group_by == 'hour' and 'shop_hour' in self.dataset.samples.columns:
            group_col = 'shop_hour'
        elif self.group_by == 'date' and 'shop_date' in self.dataset.samples.columns:
            group_col = 'shop_date'
        else:
            group_col = 'week'

        self.groups = defaultdict(list)
        for idx, row in self.dataset.samples.iterrows():
            group_key = row[group_col]
            self.groups[group_key].append(idx)

        logger.info(f"Built {len(self.groups)} evaluation groups by {self.group_by}")

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )

    def __iter__(self) -> Iterator[Tuple[str, WorldModelBatch]]:
        """Iterate over batches with group labels."""
        for group_key, indices in sorted(self.groups.items()):
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch = self.dataset.get_batch(batch_indices, apply_masking=False)
                yield group_key, batch


if __name__ == '__main__':
    # Quick test
    project_root = Path(__file__).parent.parent.parent

    print("Testing WorldModelDataset...")
    dataset = WorldModelDataset(
        project_root,
        split='train',
        max_seq_len=50,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test batch
    batch = dataset.get_batch([0, 1, 2], apply_masking=True)
    print(f"Batch size: {batch.batch_size}")
    print(f"Customer context: {batch.customer_context.shape}")
    print(f"Product embeddings: {batch.product_embeddings.shape}")
    print(f"Dense context: {batch.get_dense_context().shape}")

    print("\nTesting DataLoader...")
    dataloader = WorldModelDataLoader(dataset, batch_size=32)
    print(f"Num batches: {len(dataloader)}")

    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        print(f"Batch {i}: size={batch.batch_size}, seq_len={batch.max_seq_len}")
