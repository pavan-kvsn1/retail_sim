"""
Stage 2: Prepare Tensor Cache for World Model Training.

Converts feature files to efficient numpy arrays/memory-mapped files for fast DataLoader access.
Creates lookup indices for O(1) embedding retrieval during training.

Input feature files:
- data/features/product_embeddings.pkl (GraphSage, 4997 products x 256d)
- data/features/customer_history_embeddings.pkl (99999 customers x 160d)
- data/features/customer_embeddings.parquet (99999 customers x 160d static)
- data/features/store_features.parquet (761 stores x 96d)
- data/features/price_features.parquet (104M rows x 64d Fourier)

Output tensor cache:
- data/tensor_cache/product_embeddings.npy (4997 x 256, indexed by product_idx)
- data/tensor_cache/customer_history_embeddings.npy (99999 x 160)
- data/tensor_cache/customer_static_embeddings.npy (99999 x 160)
- data/tensor_cache/store_embeddings.npy (761 x 96)
- data/tensor_cache/price_features.npy (memory-mapped, product x store x week x 64)
- data/tensor_cache/vocab.json (product/customer/store ID -> index mappings)

Reference: RetailSim_Data_Pipeline_and_World_Model_Design.md Section 4.7
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_tensor_cache(
    project_root: Path,
    output_dir: Optional[Path] = None,
    use_mmap_for_prices: bool = True,
) -> Dict[str, Path]:
    """
    Convert feature files to efficient numpy tensor cache.

    Args:
        project_root: Path to retail_sim project root
        output_dir: Output directory (default: project_root/data/tensor_cache)
        use_mmap_for_prices: Use memory-mapped file for large price tensor

    Returns:
        Dict mapping tensor names to output file paths
    """
    project_root = Path(project_root)

    if output_dir is None:
        output_dir = project_root / 'data' / 'tensor_cache'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}
    vocab = {
        'product_to_idx': {},
        'customer_to_idx': {},
        'store_to_idx': {},
        'idx_to_product': {},
        'idx_to_customer': {},
        'idx_to_store': {},
        'special_tokens': {
            'PAD': 0,
            'MASK': 5001,
            'EOS': 5002,
        },
        'product_offset': 1,  # Products start at index 1 (0 is PAD)
    }

    # === 1. Product Embeddings (GraphSage) ===
    logger.info("Processing product embeddings (GraphSage)...")
    prod_emb_path = project_root / 'data' / 'features' / 'product_embeddings.pkl'
    with open(prod_emb_path, 'rb') as f:
        prod_data = pickle.load(f)

    product_embeddings = prod_data['embeddings']
    product_dim = prod_data['output_dim']

    # Sort products for consistent indexing
    product_ids = sorted(product_embeddings.keys())
    n_products = len(product_ids)
    logger.info(f"Found {n_products} products, dim={product_dim}")

    # Create index mappings (products get indices 1 to n_products, 0 is PAD)
    for idx, prod_id in enumerate(product_ids):
        vocab['product_to_idx'][prod_id] = idx + 1  # 1-indexed
        vocab['idx_to_product'][idx + 1] = prod_id

    # Stack into numpy array (with padding row at index 0)
    product_tensor = np.zeros((n_products + 1, product_dim), dtype=np.float32)
    for prod_id in product_ids:
        idx = vocab['product_to_idx'][prod_id]
        product_tensor[idx] = product_embeddings[prod_id]

    product_path = output_dir / 'product_embeddings.npy'
    np.save(product_path, product_tensor)
    output_files['product_embeddings'] = product_path
    logger.info(f"Saved product embeddings: {product_tensor.shape} to {product_path}")

    # === 2. Customer History Embeddings ===
    logger.info("Processing customer history embeddings...")
    cust_hist_path = project_root / 'data' / 'features' / 'customer_history_embeddings.pkl'
    with open(cust_hist_path, 'rb') as f:
        cust_hist_data = pickle.load(f)

    customer_history = cust_hist_data['embeddings']
    history_dim = cust_hist_data['output_dim']

    # Sort customers for consistent indexing
    customer_ids = sorted(customer_history.keys())
    n_customers = len(customer_ids)
    logger.info(f"Found {n_customers} customers, history_dim={history_dim}")

    # Create index mappings
    for idx, cust_id in enumerate(customer_ids):
        vocab['customer_to_idx'][cust_id] = idx
        vocab['idx_to_customer'][idx] = cust_id

    # Stack into numpy array
    customer_history_tensor = np.zeros((n_customers, history_dim), dtype=np.float32)
    for cust_id in customer_ids:
        idx = vocab['customer_to_idx'][cust_id]
        customer_history_tensor[idx] = customer_history[cust_id]

    cust_hist_path_out = output_dir / 'customer_history_embeddings.npy'
    np.save(cust_hist_path_out, customer_history_tensor)
    output_files['customer_history_embeddings'] = cust_hist_path_out
    logger.info(f"Saved customer history: {customer_history_tensor.shape} to {cust_hist_path_out}")

    # === 3. Customer Static Embeddings (demographics) ===
    logger.info("Processing customer static embeddings...")
    cust_static_path = project_root / 'data' / 'features' / 'customer_embeddings.parquet'
    cust_static_df = pd.read_parquet(cust_static_path)

    # Get embedding columns
    embed_cols = sorted([c for c in cust_static_df.columns if c.startswith('embed_')])
    static_dim = len(embed_cols)
    logger.info(f"Static embedding dim: {static_dim}")

    # Ensure same customer ordering
    cust_static_df = cust_static_df.set_index('customer_id')
    customer_static_tensor = np.zeros((n_customers, static_dim), dtype=np.float32)

    for cust_id in customer_ids:
        if cust_id in cust_static_df.index:
            idx = vocab['customer_to_idx'][cust_id]
            customer_static_tensor[idx] = cust_static_df.loc[cust_id, embed_cols].values

    cust_static_path_out = output_dir / 'customer_static_embeddings.npy'
    np.save(cust_static_path_out, customer_static_tensor)
    output_files['customer_static_embeddings'] = cust_static_path_out
    logger.info(f"Saved customer static: {customer_static_tensor.shape} to {cust_static_path_out}")

    # === 4. Store Features ===
    logger.info("Processing store features...")
    store_path = project_root / 'data' / 'features' / 'store_features.parquet'
    store_df = pd.read_parquet(store_path)

    # Get feature columns (identity, format_region, operational)
    identity_cols = sorted([c for c in store_df.columns if c.startswith('identity_')])
    format_region_cols = sorted([c for c in store_df.columns if c.startswith('format_region_')])
    operational_cols = sorted([c for c in store_df.columns if c.startswith('operational_')])
    store_feature_cols = identity_cols + format_region_cols + operational_cols
    store_dim = len(store_feature_cols)
    logger.info(f"Store feature dim: {store_dim} (identity={len(identity_cols)}, format_region={len(format_region_cols)}, operational={len(operational_cols)})")

    # Sort stores for consistent indexing
    store_ids = sorted(store_df['store_id'].unique())
    n_stores = len(store_ids)

    for idx, store_id in enumerate(store_ids):
        vocab['store_to_idx'][store_id] = idx
        vocab['idx_to_store'][idx] = store_id

    store_df = store_df.set_index('store_id')
    store_tensor = np.zeros((n_stores, store_dim), dtype=np.float32)

    for store_id in store_ids:
        idx = vocab['store_to_idx'][store_id]
        store_tensor[idx] = store_df.loc[store_id, store_feature_cols].values

    store_path_out = output_dir / 'store_embeddings.npy'
    np.save(store_path_out, store_tensor)
    output_files['store_embeddings'] = store_path_out
    logger.info(f"Saved store embeddings: {store_tensor.shape} to {store_path_out}")

    # === 5. Price Features ===
    logger.info("Processing price features...")
    price_path = project_root / 'data' / 'features' / 'price_features.parquet'

    # Get price feature columns (Fourier encoded)
    price_df_sample = pd.read_parquet(price_path, columns=['product_id', 'store_id', 'week'])
    price_cols = pd.read_parquet(price_path).columns.tolist()

    # Identify Fourier feature columns
    fourier_cols = [c for c in price_cols if c.startswith('fourier_')]
    log_cols = [c for c in price_cols if c.startswith('log_')]
    relative_cols = [c for c in price_cols if c.startswith('relative_')]
    velocity_cols = [c for c in price_cols if c.startswith('velocity_')]
    price_feature_cols = fourier_cols + log_cols + relative_cols + velocity_cols

    if not price_feature_cols:
        # Fallback: check for different naming
        price_feature_cols = [c for c in price_cols if c not in ['product_id', 'store_id', 'week', 'actual_price', 'base_price', 'discount_depth']]

    price_dim = len(price_feature_cols)
    logger.info(f"Price feature dim: {price_dim}")
    logger.info(f"Price feature columns: {price_feature_cols[:10]}...")

    # Get week range (convert to native int for JSON serialization)
    weeks = sorted([int(w) for w in price_df_sample['week'].unique()])
    n_weeks = len(weeks)
    week_to_idx = {w: i for i, w in enumerate(weeks)}
    logger.info(f"Week range: {weeks[0]} - {weeks[-1]} ({n_weeks} weeks)")

    vocab['week_to_idx'] = {str(w): i for w, i in week_to_idx.items()}  # JSON keys must be strings
    vocab['idx_to_week'] = {i: w for w, i in week_to_idx.items()}

    # For price features, create a sparse lookup or memory-mapped tensor
    # Shape: (n_products, n_stores, n_weeks, price_dim) would be too large
    # Instead, store as DataFrame indexed for fast lookup

    if use_mmap_for_prices:
        logger.info("Creating indexed price lookup (parquet with indices)...")

        # Read full price data with features
        price_df = pd.read_parquet(price_path)

        # Add index columns for fast lookup
        price_df['product_idx'] = price_df['product_id'].map(vocab['product_to_idx'])
        price_df['store_idx'] = price_df['store_id'].map(vocab['store_to_idx'])
        price_df['week_idx'] = price_df['week'].map(week_to_idx)

        # Save indexed version
        price_indexed_path = output_dir / 'price_features_indexed.parquet'
        price_df.to_parquet(price_indexed_path, index=False)
        output_files['price_features'] = price_indexed_path
        logger.info(f"Saved indexed price features: {price_df.shape} to {price_indexed_path}")

        # Also save feature column names
        vocab['price_feature_cols'] = price_feature_cols

    # === 6. Save Vocabulary ===
    vocab['dimensions'] = {
        'product': product_dim,
        'customer_history': history_dim,
        'customer_static': static_dim,
        'store': store_dim,
        'price': price_dim,
        'n_products': n_products,
        'n_customers': n_customers,
        'n_stores': n_stores,
        'n_weeks': n_weeks,
    }

    vocab['created_at'] = datetime.now().isoformat()

    vocab_path = output_dir / 'vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    output_files['vocab'] = vocab_path
    logger.info(f"Saved vocabulary to {vocab_path}")

    # === Summary ===
    logger.info("\n=== Tensor Cache Summary ===")
    logger.info(f"Product embeddings: {n_products} x {product_dim}")
    logger.info(f"Customer history: {n_customers} x {history_dim}")
    logger.info(f"Customer static: {n_customers} x {static_dim}")
    logger.info(f"Store features: {n_stores} x {store_dim}")
    logger.info(f"Price features: {n_weeks} weeks x {price_dim} features (indexed)")
    logger.info(f"Output directory: {output_dir}")

    return output_files


class TensorCache:
    """
    Efficient tensor cache for World Model training.

    Loads pre-computed embeddings and provides O(1) lookup by ID or index.
    """

    def __init__(self, cache_dir: Path):
        """
        Load tensor cache from directory.

        Args:
            cache_dir: Path to data/tensor_cache directory
        """
        self.cache_dir = Path(cache_dir)

        # Load vocabulary
        with open(self.cache_dir / 'vocab.json', 'r') as f:
            self.vocab = json.load(f)

        # Load tensors
        logger.info("Loading tensor cache...")
        self.product_embeddings = np.load(self.cache_dir / 'product_embeddings.npy')
        self.customer_history = np.load(self.cache_dir / 'customer_history_embeddings.npy')
        self.customer_static = np.load(self.cache_dir / 'customer_static_embeddings.npy')
        self.store_embeddings = np.load(self.cache_dir / 'store_embeddings.npy')

        # Load price features (lazy load due to size)
        self._price_df = None

        logger.info(f"Loaded tensor cache from {cache_dir}")
        logger.info(f"  Products: {self.product_embeddings.shape}")
        logger.info(f"  Customers: {self.customer_history.shape} + {self.customer_static.shape}")
        logger.info(f"  Stores: {self.store_embeddings.shape}")

    @property
    def price_features(self) -> pd.DataFrame:
        """Lazy load price features."""
        if self._price_df is None:
            logger.info("Loading price features...")
            self._price_df = pd.read_parquet(self.cache_dir / 'price_features_indexed.parquet')
            # Create multi-index for fast lookup
            self._price_df = self._price_df.set_index(['product_idx', 'store_idx', 'week_idx'])
        return self._price_df

    def get_product_embedding(self, product_id: str) -> np.ndarray:
        """Get product embedding by ID."""
        idx = self.vocab['product_to_idx'].get(product_id)
        if idx is None:
            return np.zeros(self.vocab['dimensions']['product'], dtype=np.float32)
        return self.product_embeddings[idx]

    def get_product_embeddings_batch(self, product_indices: np.ndarray) -> np.ndarray:
        """Get product embeddings for batch of indices."""
        return self.product_embeddings[product_indices]

    def get_customer_embedding(self, customer_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get customer history and static embeddings by ID."""
        idx = self.vocab['customer_to_idx'].get(customer_id)
        if idx is None:
            history_dim = self.vocab['dimensions']['customer_history']
            static_dim = self.vocab['dimensions']['customer_static']
            return np.zeros(history_dim, dtype=np.float32), np.zeros(static_dim, dtype=np.float32)
        return self.customer_history[idx], self.customer_static[idx]

    def get_customer_embeddings_batch(self, customer_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get customer embeddings for batch of indices."""
        return self.customer_history[customer_indices], self.customer_static[customer_indices]

    def get_store_embedding(self, store_id: str) -> np.ndarray:
        """Get store embedding by ID."""
        idx = self.vocab['store_to_idx'].get(store_id)
        if idx is None:
            return np.zeros(self.vocab['dimensions']['store'], dtype=np.float32)
        return self.store_embeddings[idx]

    def get_store_embeddings_batch(self, store_indices: np.ndarray) -> np.ndarray:
        """Get store embeddings for batch of indices."""
        return self.store_embeddings[store_indices]

    def get_price_features(self, product_idx: int, store_idx: int, week_idx: int) -> np.ndarray:
        """Get price features for product/store/week combination."""
        try:
            row = self.price_features.loc[(product_idx, store_idx, week_idx)]
            feature_cols = self.vocab['price_feature_cols']
            return row[feature_cols].values.astype(np.float32)
        except KeyError:
            return np.zeros(self.vocab['dimensions']['price'], dtype=np.float32)

    def product_id_to_idx(self, product_id: str) -> int:
        """Convert product ID to index."""
        return self.vocab['product_to_idx'].get(product_id, 0)  # 0 is PAD

    def customer_id_to_idx(self, customer_id: str) -> int:
        """Convert customer ID to index."""
        return self.vocab['customer_to_idx'].get(customer_id, 0)

    def store_id_to_idx(self, store_id: str) -> int:
        """Convert store ID to index."""
        return self.vocab['store_to_idx'].get(store_id, 0)

    def week_to_idx(self, week: int) -> int:
        """Convert YYYYWW week to index."""
        return self.vocab['week_to_idx'].get(str(week), 0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare tensor cache')
    parser.add_argument(
        '--project-root',
        type=str,
        default='/Users/hazymoji/Documents/DataDev/ML Projects/retail_sim',
        help='Path to project root'
    )

    args = parser.parse_args()

    output_files = prepare_tensor_cache(
        project_root=Path(args.project_root),
    )

    print("\nOutput files created:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")
