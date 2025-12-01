"""
Stage 4: Tensor Cache Builder
=============================
Pre-computes and caches static embeddings for efficient training.

Input: Training samples + processed features
Output:
  - product_embeddings.pt
  - customer_segment_embeddings.pt
  - store_embeddings.pt
  - vocab.json (ID mappings)

This avoids recomputing static embeddings during training.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings

warnings.filterwarnings('ignore')


class TensorCacheBuilder:
    """
    Builds tensor caches for static embeddings.

    Creates:
    1. Product ID -> index mapping
    2. Customer segment embeddings (for cold-start)
    3. Store embeddings
    4. Pre-computed product feature tensors
    """

    # Special tokens
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    MASK_TOKEN = '[MASK]'

    def __init__(
        self,
        embedding_dim: int = 128,
        pad_idx: int = 0,
        unk_idx: int = 1,
        mask_idx: int = 2
    ):
        """
        Parameters
        ----------
        embedding_dim : int
            Dimension for embeddings
        pad_idx : int
            Index for padding token
        unk_idx : int
            Index for unknown token
        mask_idx : int
            Index for mask token
        """
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.mask_idx = mask_idx

    def run(
        self,
        transactions_df: pd.DataFrame,
        product_features_path: Optional[Path] = None,
        customer_segments_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Build tensor caches.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions for building vocabularies
        product_features_path : Path, optional
            Path to product features parquet
        customer_segments_path : Path, optional
            Path to customer segments parquet

        Returns
        -------
        Dict containing:
            - vocab: ID mappings
            - product_features: tensor (N_products, D)
            - segment_embeddings: tensor (N_segments, D)
            - store_embeddings: tensor (N_stores, D)
        """
        print("Stage 4: Tensor Cache Building")
        print("=" * 60)

        cache = {}

        # Step 1: Build vocabularies
        print("\nStep 1: Building vocabularies...")
        vocab = self._build_vocabularies(transactions_df)
        cache['vocab'] = vocab
        self._print_vocab_stats(vocab)

        # Step 2: Initialize product embeddings
        print("\nStep 2: Initializing product embeddings...")
        product_embeddings = self._init_product_embeddings(
            vocab, product_features_path
        )
        cache['product_embeddings'] = product_embeddings
        print(f"  - Shape: {product_embeddings.shape}")

        # Step 3: Build segment embeddings (for cold-start)
        print("\nStep 3: Building segment embeddings...")
        segment_embeddings = self._build_segment_embeddings(
            customer_segments_path
        )
        cache['segment_embeddings'] = segment_embeddings
        print(f"  - Shape: {segment_embeddings.shape}")

        # Step 4: Build store embeddings
        print("\nStep 4: Building store embeddings...")
        store_embeddings = self._build_store_embeddings(vocab)
        cache['store_embeddings'] = store_embeddings
        print(f"  - Shape: {store_embeddings.shape}")

        # Step 5: Build category embeddings
        print("\nStep 5: Building category embeddings...")
        category_embeddings = self._build_category_embeddings(vocab)
        cache['category_embeddings'] = category_embeddings
        print(f"  - Shape: {category_embeddings.shape}")

        print("\n" + "=" * 60)
        print("Tensor Cache Building Complete!")

        return cache

    def _build_vocabularies(
        self,
        transactions_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Build ID -> index mappings for all entity types."""
        vocab = {}

        # Product vocabulary (with special tokens)
        products = transactions_df['PROD_CODE'].unique().tolist()
        product_to_idx = {
            self.PAD_TOKEN: self.pad_idx,
            self.UNK_TOKEN: self.unk_idx,
            self.MASK_TOKEN: self.mask_idx
        }
        for idx, prod in enumerate(products, start=3):
            product_to_idx[str(prod)] = idx

        vocab['product_to_idx'] = product_to_idx
        vocab['idx_to_product'] = {v: k for k, v in product_to_idx.items()}
        vocab['n_products'] = len(product_to_idx)

        # Store vocabulary
        stores = transactions_df['STORE_CODE'].unique().tolist()
        store_to_idx = {self.PAD_TOKEN: 0}
        for idx, store in enumerate(stores, start=1):
            store_to_idx[str(store)] = idx

        vocab['store_to_idx'] = store_to_idx
        vocab['idx_to_store'] = {v: k for k, v in store_to_idx.items()}
        vocab['n_stores'] = len(store_to_idx)

        # Customer vocabulary
        customers = transactions_df['CUST_CODE'].unique().tolist()
        customer_to_idx = {self.PAD_TOKEN: 0}
        for idx, cust in enumerate(customers, start=1):
            customer_to_idx[str(cust)] = idx

        vocab['customer_to_idx'] = customer_to_idx
        vocab['idx_to_customer'] = {v: k for k, v in customer_to_idx.items()}
        vocab['n_customers'] = len(customer_to_idx)

        # Category vocabulary (if available)
        if 'PROD_CODE_40' in transactions_df.columns:
            categories = transactions_df['PROD_CODE_40'].dropna().unique().tolist()
            category_to_idx = {self.PAD_TOKEN: 0}
            for idx, cat in enumerate(categories, start=1):
                category_to_idx[str(cat)] = idx

            vocab['category_to_idx'] = category_to_idx
            vocab['n_categories'] = len(category_to_idx)

        return vocab

    def _print_vocab_stats(self, vocab: Dict) -> None:
        """Print vocabulary statistics."""
        print(f"  - Products: {vocab['n_products']:,} (including special tokens)")
        print(f"  - Stores: {vocab['n_stores']:,}")
        print(f"  - Customers: {vocab['n_customers']:,}")
        if 'n_categories' in vocab:
            print(f"  - Categories: {vocab['n_categories']:,}")

    def _init_product_embeddings(
        self,
        vocab: Dict,
        product_features_path: Optional[Path]
    ) -> torch.Tensor:
        """
        Initialize product embeddings.

        If product features exist, use them. Otherwise, random init.
        """
        n_products = vocab['n_products']

        if product_features_path and Path(product_features_path).exists():
            # Load pre-computed features
            features_df = pd.read_parquet(product_features_path)

            # Map features to embedding matrix
            embeddings = torch.zeros(n_products, self.embedding_dim)

            # Special tokens get zero embeddings (will be learned)
            # Regular products get features (truncated/padded to embedding_dim)

            for _, row in features_df.iterrows():
                prod_id = str(row.get('product_id', row.get('PROD_CODE', '')))
                if prod_id in vocab['product_to_idx']:
                    idx = vocab['product_to_idx'][prod_id]
                    # Extract numeric features
                    feature_cols = [c for c in features_df.columns
                                  if c not in ['product_id', 'PROD_CODE']]
                    features = row[feature_cols].values.astype(np.float32)

                    # Pad/truncate to embedding_dim
                    if len(features) < self.embedding_dim:
                        features = np.pad(features,
                                        (0, self.embedding_dim - len(features)))
                    else:
                        features = features[:self.embedding_dim]

                    embeddings[idx] = torch.tensor(features)

            print(f"  - Loaded features from {product_features_path}")
        else:
            # Random initialization (Xavier)
            embeddings = torch.zeros(n_products, self.embedding_dim)
            torch.nn.init.xavier_uniform_(embeddings[3:])  # Skip special tokens
            print("  - Random initialization (no features found)")

        return embeddings

    def _build_segment_embeddings(
        self,
        customer_segments_path: Optional[Path]
    ) -> torch.Tensor:
        """
        Build segment embeddings for cold-start customers.

        Segments are based on customer clustering from feature engineering.
        """
        # Default: 10 segments + padding
        n_segments = 11

        if customer_segments_path and Path(customer_segments_path).exists():
            segments_df = pd.read_parquet(customer_segments_path)
            n_segments = segments_df['segment_id'].nunique() + 1
            print(f"  - Loaded {n_segments - 1} segments from file")
        else:
            print("  - Using default 10 segments")

        # Random initialization
        embeddings = torch.zeros(n_segments, self.embedding_dim)
        torch.nn.init.xavier_uniform_(embeddings[1:])  # Skip padding

        return embeddings

    def _build_store_embeddings(self, vocab: Dict) -> torch.Tensor:
        """Build store embeddings (random init, will be learned)."""
        n_stores = vocab['n_stores']

        embeddings = torch.zeros(n_stores, self.embedding_dim)
        torch.nn.init.xavier_uniform_(embeddings[1:])  # Skip padding

        return embeddings

    def _build_category_embeddings(self, vocab: Dict) -> torch.Tensor:
        """Build category embeddings (random init, will be learned)."""
        n_categories = vocab.get('n_categories', 100)

        embeddings = torch.zeros(n_categories, self.embedding_dim)
        torch.nn.init.xavier_uniform_(embeddings[1:])  # Skip padding

        return embeddings

    def save(self, cache: Dict[str, Any], output_dir: Path) -> None:
        """Save tensor cache to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vocab as JSON
        vocab_path = output_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            # Convert int keys to str for JSON
            vocab_save = {}
            for key, value in cache['vocab'].items():
                if isinstance(value, dict):
                    vocab_save[key] = {str(k): v for k, v in value.items()}
                else:
                    vocab_save[key] = value
            json.dump(vocab_save, f, indent=2)
        print(f"Saved vocab: {vocab_path}")

        # Save embeddings as PyTorch tensors
        for name in ['product_embeddings', 'segment_embeddings',
                     'store_embeddings', 'category_embeddings']:
            if name in cache:
                tensor_path = output_dir / f'{name}.pt'
                torch.save(cache[name], tensor_path)
                print(f"Saved {name}: {tensor_path}")

    @staticmethod
    def load(cache_dir: Path) -> Dict[str, Any]:
        """Load tensor cache from files."""
        cache_dir = Path(cache_dir)
        cache = {}

        # Load vocab
        vocab_path = cache_dir / 'vocab.json'
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                cache['vocab'] = json.load(f)

        # Load embeddings
        for name in ['product_embeddings', 'segment_embeddings',
                     'store_embeddings', 'category_embeddings']:
            tensor_path = cache_dir / f'{name}.pt'
            if tensor_path.exists():
                cache[name] = torch.load(tensor_path)

        return cache


def main():
    """Run tensor cache building."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    output_dir = project_root / 'data' / 'prepared' / 'tensor_cache'

    # Optional paths for pre-computed features
    product_features_path = project_root / 'data' / 'features' / 'product_features.parquet'
    customer_segments_path = project_root / 'data' / 'features' / 'customer_segments.parquet'

    # Load transactions
    print(f"Loading transactions from {raw_data_path}...")
    usecols = ['PROD_CODE', 'STORE_CODE', 'CUST_CODE']
    if Path(raw_data_path).exists():
        # Check if PROD_CODE_40 exists
        sample = pd.read_csv(raw_data_path, nrows=1)
        if 'PROD_CODE_40' in sample.columns:
            usecols.append('PROD_CODE_40')

    transactions_df = pd.read_csv(raw_data_path, usecols=usecols)
    print(f"  - Loaded {len(transactions_df):,} transactions")

    # Build cache
    builder = TensorCacheBuilder(embedding_dim=128)
    cache = builder.run(
        transactions_df,
        product_features_path if product_features_path.exists() else None,
        customer_segments_path if customer_segments_path.exists() else None
    )

    # Save
    print("\nSaving tensor cache...")
    builder.save(cache, output_dir)

    return cache


if __name__ == '__main__':
    main()
