"""
RetailSim Dataset and DataLoader
=================================
Combines all 6 tensors into batched training samples.

Tensor Assembly:
- T1: Customer Context [192d] - Dense, per transaction
- T2: Product Sequence [256d/item] - Sparse, variable length
- T3: Temporal Context [64d] - Dense, per transaction
- T4: Price Context [64d/item] - Sparse, aligned with T2
- T5: Store Context [96d] - Dense, per transaction
- T6: Trip Context [48d] - Dense, per transaction (dual-use)

Total Dense Context: 192 + 64 + 96 + 48 = 400d
Sequence Features: 256 + 64 = 320d per item
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import pickle
import warnings

from t1_customer_context import CustomerContextEncoder
from t2_product_sequence import ProductSequenceEncoder, ProductSequenceBatch
from t3_temporal_context import TemporalContextEncoder
from t4_price_context import PriceContextEncoder, PriceContextBatch
from t5_store_context import StoreContextEncoder
from t6_trip_context import TripContextEncoder

warnings.filterwarnings('ignore')


@dataclass
class RetailSimBatch:
    """Container for a batch of RetailSim training samples."""
    # Dense context tensors
    customer_context: np.ndarray      # [B, 192] T1
    temporal_context: np.ndarray      # [B, 64] T3
    store_context: np.ndarray         # [B, 96] T5
    trip_context: np.ndarray          # [B, 48] T6

    # Sequence tensors
    product_embeddings: np.ndarray    # [B, S, 256] T2
    product_token_ids: np.ndarray     # [B, S]
    price_features: np.ndarray        # [B, S, 64] T4
    attention_mask: np.ndarray        # [B, S]
    sequence_lengths: np.ndarray      # [B]

    # Labels for auxiliary prediction
    trip_labels: Dict[str, np.ndarray]  # mission_type, mission_focus, etc.

    # Optional: Masked LM targets
    masked_positions: Optional[np.ndarray] = None
    masked_targets: Optional[np.ndarray] = None

    # Metadata
    basket_ids: Optional[List[str]] = None
    customer_ids: Optional[List[str]] = None

    @property
    def batch_size(self) -> int:
        return self.customer_context.shape[0]

    @property
    def dense_context_dim(self) -> int:
        """Total dimension of dense context (T1 + T3 + T5 + T6)."""
        return 192 + 64 + 96 + 48  # 400d

    @property
    def sequence_feature_dim(self) -> int:
        """Feature dimension per sequence item (T2 + T4)."""
        return 256 + 64  # 320d

    def get_dense_context(self) -> np.ndarray:
        """Concatenate all dense context tensors [B, 400]."""
        return np.concatenate([
            self.customer_context,
            self.temporal_context,
            self.store_context,
            self.trip_context
        ], axis=1)

    def get_sequence_features(self) -> np.ndarray:
        """Concatenate product and price features [B, S, 320]."""
        return np.concatenate([
            self.product_embeddings,
            self.price_features
        ], axis=2)


class RetailSimDataset:
    """
    Dataset for RetailSim training.

    Loads pre-computed features and prepares batched tensors.
    """

    def __init__(
        self,
        project_root: Path,
        max_seq_len: int = 50,
        mask_prob: float = 0.15
    ):
        """
        Parameters
        ----------
        project_root : Path
            Project root directory
        max_seq_len : int
            Maximum sequence length for padding
        mask_prob : float
            Probability of masking tokens during training
        """
        self.project_root = Path(project_root)
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob

        # Load all pre-computed features
        self._load_features()

        # Initialize encoders
        self._init_encoders()

        # Process transactions
        self._prepare_data()

    def _load_features(self):
        """Load pre-computed features from feature engineering."""
        features_dir = self.project_root / 'data' / 'features'

        # Product embeddings (from Layer 3)
        print("Loading product embeddings...")
        with open(features_dir / 'product_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
            self.product_embeddings = data.get('embeddings', {})
        print(f"  Loaded {len(self.product_embeddings)} products")

        # Customer embeddings (from Layer 4)
        print("Loading customer embeddings...")
        self.customer_embeddings = pd.read_parquet(
            features_dir / 'customer_embeddings.parquet'
        )
        print(f"  Loaded {len(self.customer_embeddings)} customers")

        # Store features (from Layer 5)
        print("Loading store features...")
        self.store_features = pd.read_parquet(
            features_dir / 'store_features.parquet'
        )
        print(f"  Loaded {len(self.store_features)} stores")

        # Price features (from Layer 2)
        print("Loading price features...")
        self.price_features_df = pd.read_parquet(
            features_dir / 'price_features.parquet'
        )
        print(f"  Loaded {len(self.price_features_df)} price records")

        # Mission patterns (from Stage 4)
        print("Loading mission patterns...")
        self.mission_patterns = pd.read_parquet(
            self.project_root / 'data' / 'processed' / 'customer_mission_patterns.parquet'
        )
        print(f"  Loaded {len(self.mission_patterns)} customers with patterns")

    def _init_encoders(self):
        """Initialize all tensor encoders."""
        # T1: Customer Context
        self.customer_embed_dict = {}
        for _, row in self.customer_embeddings.iterrows():
            cust_id = row['customer_id']
            embed = row[[c for c in row.index if c.startswith('embed_')]].values.astype(np.float32)
            self.customer_embed_dict[cust_id] = embed

        # Build trip counts for cold-start handling
        self.trip_counts = self.customer_embeddings.set_index('customer_id')['total_trips'].to_dict()

        self.customer_encoder = CustomerContextEncoder()

        # T2: Product Sequence
        self.product_encoder = ProductSequenceEncoder(
            product_embeddings=self.product_embeddings,
            max_seq_len=self.max_seq_len,
            mask_prob=self.mask_prob
        )

        # T3: Temporal Context
        self.temporal_encoder = TemporalContextEncoder()

        # T4: Price Context
        self.price_encoder = PriceContextEncoder()

        # T5: Store Context
        self.store_encoder = StoreContextEncoder()

        # T6: Trip Context
        self.trip_encoder = TripContextEncoder()

        # Build price lookup from price features
        self._build_price_lookup()

        # Build store tensor lookup
        self._build_store_lookup()

    def _build_price_lookup(self):
        """Build price lookup dictionary."""
        self.price_lookup = {}
        self.category_avg_lookup = {}

        # Group by product to get average prices
        for _, row in self.price_features_df.iterrows():
            prod_id = row['product_id']
            self.price_lookup[prod_id] = {
                'actual_price': row.get('actual_price', 1.0),
                'base_price': row.get('base_price', row.get('actual_price', 1.0)),
                'prior_price': row.get('actual_price', 1.0)  # Simplified
            }

        # Compute category averages (simplified: use product price as its own category avg)
        for prod_id, info in self.price_lookup.items():
            self.category_avg_lookup[prod_id] = info['actual_price']

    def _build_store_lookup(self):
        """Pre-encode all store contexts."""
        self.store_tensor_lookup = {}

        # Build store metadata from features
        store_df = self.store_features[['store_id']].drop_duplicates()
        store_df['format'] = 'MS'  # Default format
        store_df['region'] = 'E01'  # Default region

        self.store_tensor_lookup = self.store_encoder.encode_batch(
            store_df, self.store_features
        )

    def _prepare_data(self):
        """Load and prepare transaction data."""
        print("Loading transactions...")
        self.transactions = pd.read_csv(
            self.project_root / 'raw_data' / 'transactions.csv',
            nrows=10000
        )
        print(f"  Loaded {len(self.transactions)} transaction rows")

        # Group by basket
        self.baskets = self.transactions.groupby('BASKET_ID').agg({
            'CUST_CODE': 'first',
            'STORE_CODE': 'first',
            'SHOP_WEEK': 'first',
            'SHOP_WEEKDAY': 'first',
            'SHOP_HOUR': 'first',
            'PROD_CODE': list,
            'BASKET_TYPE': 'first',
            'BASKET_DOMINANT_MISSION': 'first',
            'BASKET_PRICE_SENSITIVITY': 'first',
            'BASKET_SIZE': 'first'
        }).reset_index()

        print(f"  Grouped into {len(self.baskets)} baskets")

        # Build customer last visit lookup
        self._build_last_visit_lookup()

    def _build_last_visit_lookup(self):
        """Build lookup for customer's last visit week."""
        self.last_visit_lookup = {}

        basket_sorted = self.baskets.sort_values(['CUST_CODE', 'SHOP_WEEK'])

        for cust_id in basket_sorted['CUST_CODE'].dropna().unique():
            cust_baskets = basket_sorted[basket_sorted['CUST_CODE'] == cust_id]
            weeks = cust_baskets['SHOP_WEEK'].tolist()

            # For each basket, store the previous week
            for i, basket_id in enumerate(cust_baskets['BASKET_ID'].tolist()):
                if i > 0:
                    self.last_visit_lookup[(cust_id, basket_id)] = weeks[i - 1]

    def __len__(self) -> int:
        return len(self.baskets)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample (for debugging)."""
        row = self.baskets.iloc[idx]
        return self._encode_sample(row)

    def _encode_sample(self, row: pd.Series) -> Dict:
        """Encode a single basket into tensors."""
        basket_id = row['BASKET_ID']
        customer_id = row['CUST_CODE']
        store_id = row['STORE_CODE']
        products = row['PROD_CODE']

        # T1: Customer context
        history_embed = self.customer_embed_dict.get(customer_id)
        num_trips = self.trip_counts.get(customer_id, 0)

        t1 = self.customer_encoder.encode_customer(
            customer_id=customer_id,
            seg1='CT',  # Default segment - would come from customer data
            seg2='DI',  # Default segment - would come from customer data
            history_embedding=history_embed,
            num_trips=num_trips
        )

        # T2: Product sequence (single sample, no masking for now)
        emb, tids, length, _, _ = self.product_encoder.encode_sequence(
            products, add_eos=True, apply_masking=False
        )

        # T3: Temporal context
        last_visit = self.last_visit_lookup.get((customer_id, basket_id))
        t3 = self.temporal_encoder.encode_temporal(
            shop_week=row['SHOP_WEEK'],
            shop_weekday=row.get('SHOP_WEEKDAY', 1),
            shop_hour=row.get('SHOP_HOUR', 12),
            last_visit_week=last_visit
        )

        # T4: Price context
        t4 = self.price_encoder.encode_basket_prices(
            products, self.price_lookup, self.category_avg_lookup
        )

        # T5: Store context
        t5 = self.store_tensor_lookup.get(
            store_id,
            np.zeros(self.store_encoder.output_dim)
        )

        # T6: Trip context
        t6 = self.trip_encoder.encode_trip(
            mission_type=row.get('BASKET_TYPE', 'Top Up'),
            mission_focus=row.get('BASKET_DOMINANT_MISSION', 'Mixed'),
            price_sensitivity=row.get('BASKET_PRICE_SENSITIVITY', 'MM'),
            basket_size=row.get('BASKET_SIZE', 'M')
        )

        return {
            'basket_id': basket_id,
            'customer_id': customer_id,
            't1': t1,
            't2_embeddings': emb,
            't2_token_ids': tids,
            't3': t3,
            't4': t4,
            't5': t5,
            't6': t6,
            'length': length,
            'products': products
        }

    def get_batch(
        self,
        indices: List[int],
        apply_masking: bool = False
    ) -> RetailSimBatch:
        """
        Get a batch of samples.

        Parameters
        ----------
        indices : List[int]
            Indices of samples to include
        apply_masking : bool
            Whether to apply BERT-style masking

        Returns
        -------
        RetailSimBatch
            Batched tensor data
        """
        batch_size = len(indices)
        rows = [self.baskets.iloc[i] for i in indices]

        # Collect all products for batch encoding
        all_products = [row['PROD_CODE'] for row in rows]
        customer_ids = [row['CUST_CODE'] for row in rows]
        store_ids = [row['STORE_CODE'] for row in rows]
        basket_ids = [row['BASKET_ID'] for row in rows]

        # T1: Customer contexts [B, 192]
        customer_contexts = []
        for row in rows:
            cust_id = row['CUST_CODE']
            history_embed = self.customer_embed_dict.get(cust_id)
            num_trips = self.trip_counts.get(cust_id, 0)
            t1 = self.customer_encoder.encode_customer(
                customer_id=cust_id,
                seg1='CT',  # Default segment
                seg2='DI',  # Default segment
                history_embedding=history_embed,
                num_trips=num_trips
            )
            customer_contexts.append(t1)
        customer_contexts = np.array(customer_contexts)

        # T2: Product sequences [B, S, 256]
        product_batch = self.product_encoder.encode_batch(
            all_products, apply_masking=apply_masking
        )

        # T3: Temporal contexts [B, 64]
        temporal_contexts = []
        for row in rows:
            basket_id = row['BASKET_ID']
            cust_id = row['CUST_CODE']
            last_visit = self.last_visit_lookup.get((cust_id, basket_id))
            t3 = self.temporal_encoder.encode_temporal(
                shop_week=row['SHOP_WEEK'],
                shop_weekday=row.get('SHOP_WEEKDAY', 1),
                shop_hour=row.get('SHOP_HOUR', 12),
                last_visit_week=last_visit
            )
            temporal_contexts.append(t3)
        temporal_contexts = np.array(temporal_contexts)

        # T4: Price contexts [B, S, 64]
        price_batch = self.price_encoder.encode_batch(
            all_products,
            self.price_lookup,
            self.category_avg_lookup,
            max_seq_len=product_batch.embeddings.shape[1]
        )

        # T5: Store contexts [B, 96]
        store_contexts = []
        for store_id in store_ids:
            t5 = self.store_tensor_lookup.get(
                store_id,
                np.zeros(self.store_encoder.output_dim)
            )
            store_contexts.append(t5)
        store_contexts = np.array(store_contexts)

        # T6: Trip contexts [B, 48] and labels
        trip_contexts = []
        trip_labels = {
            'mission_type': [],
            'mission_focus': [],
            'price_sensitivity': [],
            'basket_size': []
        }

        for row in rows:
            mission_type = row.get('BASKET_TYPE', 'Top Up')
            mission_focus = row.get('BASKET_DOMINANT_MISSION', 'Mixed')
            price_sensitivity = row.get('BASKET_PRICE_SENSITIVITY', 'MM')
            basket_size = row.get('BASKET_SIZE', 'M')

            # Handle NaN
            if pd.isna(mission_type):
                mission_type = 'Top Up'
            if pd.isna(mission_focus):
                mission_focus = 'Mixed'
            if pd.isna(price_sensitivity):
                price_sensitivity = 'MM'
            if pd.isna(basket_size):
                basket_size = 'M'

            t6 = self.trip_encoder.encode_trip(
                mission_type, mission_focus, price_sensitivity, basket_size
            )
            trip_contexts.append(t6)

            # Store labels for auxiliary prediction
            trip_labels['mission_type'].append(
                self.trip_encoder.MISSION_TYPE_VOCAB.index(mission_type)
                if mission_type in self.trip_encoder.MISSION_TYPE_VOCAB else 0
            )
            trip_labels['mission_focus'].append(
                self.trip_encoder.MISSION_FOCUS_VOCAB.index(mission_focus)
                if mission_focus in self.trip_encoder.MISSION_FOCUS_VOCAB else 2
            )
            trip_labels['price_sensitivity'].append(
                self.trip_encoder.PRICE_SENSITIVITY_VOCAB.index(price_sensitivity)
                if price_sensitivity in self.trip_encoder.PRICE_SENSITIVITY_VOCAB else 1
            )
            trip_labels['basket_size'].append(
                self.trip_encoder.BASKET_SCOPE_VOCAB.index(basket_size)
                if basket_size in self.trip_encoder.BASKET_SCOPE_VOCAB else 1
            )

        trip_contexts = np.array(trip_contexts)
        for key in trip_labels:
            trip_labels[key] = np.array(trip_labels[key])

        return RetailSimBatch(
            customer_context=customer_contexts,
            temporal_context=temporal_contexts,
            store_context=store_contexts,
            trip_context=trip_contexts,
            product_embeddings=product_batch.embeddings,
            product_token_ids=product_batch.token_ids,
            price_features=price_batch.features,
            attention_mask=product_batch.attention_mask,
            sequence_lengths=product_batch.lengths,
            trip_labels=trip_labels,
            masked_positions=product_batch.masked_positions,
            masked_targets=product_batch.masked_targets,
            basket_ids=basket_ids,
            customer_ids=customer_ids
        )


class RetailSimDataLoader:
    """
    DataLoader for RetailSim dataset.

    Provides batched iteration over the dataset.
    """

    def __init__(
        self,
        dataset: RetailSimDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        apply_masking: bool = True
    ):
        """
        Parameters
        ----------
        dataset : RetailSimDataset
            The dataset to load from
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle data each epoch
        apply_masking : bool
            Whether to apply BERT-style masking
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.apply_masking = apply_masking

        self.indices = list(range(len(dataset)))

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[RetailSimBatch]:
        """Iterate over batches."""
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start:start + self.batch_size]
            yield self.dataset.get_batch(batch_indices, self.apply_masking)


def main():
    """Test dataset and dataloader."""
    project_root = Path(__file__).parent.parent.parent

    print("=" * 60)
    print("RetailSim Dataset Test")
    print("=" * 60)

    # Create dataset
    print("\nInitializing dataset...")
    dataset = RetailSimDataset(project_root)

    print(f"\nDataset size: {len(dataset)} baskets")

    # Test single sample
    print("\n" + "-" * 40)
    print("Testing single sample encoding...")
    sample = dataset[0]
    print(f"  Basket ID: {sample['basket_id']}")
    print(f"  Customer ID: {sample['customer_id']}")
    print(f"  Products: {len(sample['products'])}")
    print(f"  T1 shape: {sample['t1'].shape}")
    print(f"  T2 embeddings shape: {sample['t2_embeddings'].shape}")
    print(f"  T3 shape: {sample['t3'].shape}")
    print(f"  T4 shape: {sample['t4'].shape}")
    print(f"  T5 shape: {sample['t5'].shape}")
    print(f"  T6 shape: {sample['t6'].shape}")

    # Test batch
    print("\n" + "-" * 40)
    print("Testing batch encoding...")
    batch = dataset.get_batch([0, 1, 2, 3], apply_masking=True)
    print(f"  Batch size: {batch.batch_size}")
    print(f"  Customer context: {batch.customer_context.shape}")
    print(f"  Temporal context: {batch.temporal_context.shape}")
    print(f"  Store context: {batch.store_context.shape}")
    print(f"  Trip context: {batch.trip_context.shape}")
    print(f"  Product embeddings: {batch.product_embeddings.shape}")
    print(f"  Price features: {batch.price_features.shape}")
    print(f"  Attention mask: {batch.attention_mask.shape}")
    print(f"  Sequence lengths: {batch.sequence_lengths}")

    print(f"\n  Dense context: {batch.get_dense_context().shape}")
    print(f"  Sequence features: {batch.get_sequence_features().shape}")

    if batch.masked_positions is not None:
        print(f"  Masked positions: {batch.masked_positions.shape}")

    print(f"\n  Trip labels:")
    for key, val in batch.trip_labels.items():
        print(f"    {key}: {val}")

    # Test dataloader
    print("\n" + "-" * 40)
    print("Testing DataLoader...")
    dataloader = RetailSimDataLoader(dataset, batch_size=16, shuffle=True)
    print(f"  Number of batches: {len(dataloader)}")

    # Iterate through a few batches
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        print(f"  Batch {i}: size={batch.batch_size}, "
              f"seq_len={batch.product_embeddings.shape[1]}")

    print("\n" + "=" * 60)
    print("Dataset test complete!")
    print("=" * 60)

    return dataset, dataloader


if __name__ == '__main__':
    main()
