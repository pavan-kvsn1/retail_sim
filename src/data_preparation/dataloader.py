"""
DataLoader for RetailSim Training
=================================
PyTorch DataLoader with:
- Temporal windowing
- Dynamic history padding
- MEM (Masked Event Modeling) masking
- Efficient batching by history length bucket

Supports train/validation/test splits with proper temporal boundaries.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import random
import warnings

warnings.filterwarnings('ignore')


class RetailSimDataset(Dataset):
    """
    PyTorch Dataset for RetailSim training samples.

    Each sample contains:
    - Customer history (sequence of baskets)
    - Target basket (products to predict)
    - Context features (store, time)
    - Metadata (cold-start flag, etc.)
    """

    def __init__(
        self,
        samples_df: pd.DataFrame,
        vocab: Dict[str, Dict],
        max_history_length: int = 50,
        max_basket_size: int = 30,
        max_target_size: int = 50,
        mask_ratio: float = 0.15,
        apply_mem_masking: bool = True
    ):
        """
        Parameters
        ----------
        samples_df : pd.DataFrame
            Training samples from Stage 3
        vocab : Dict
            Vocabulary mappings from Stage 4
        max_history_length : int
            Maximum baskets in history
        max_basket_size : int
            Maximum products per history basket
        max_target_size : int
            Maximum products in target basket
        mask_ratio : float
            Ratio of products to mask for MEM
        apply_mem_masking : bool
            Whether to apply MEM masking (False for val/test)
        """
        self.samples_df = samples_df.reset_index(drop=True)
        self.vocab = vocab
        self.max_history_length = max_history_length
        self.max_basket_size = max_basket_size
        self.max_target_size = max_target_size
        self.mask_ratio = mask_ratio
        self.apply_mem_masking = apply_mem_masking

        # Get special token indices
        self.pad_idx = vocab['product_to_idx'].get('[PAD]', 0)
        self.unk_idx = vocab['product_to_idx'].get('[UNK]', 1)
        self.mask_idx = vocab['product_to_idx'].get('[MASK]', 2)

    def __len__(self) -> int:
        return len(self.samples_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.samples_df.iloc[idx]

        # Parse history products
        history_products = row['history_products']
        if isinstance(history_products, str):
            history_products = json.loads(history_products)

        # Parse target products
        target_products = row['target_products']
        if isinstance(target_products, str):
            target_products = json.loads(target_products)

        # Convert to indices
        history_indices = self._encode_history(history_products)
        target_indices = self._encode_target(target_products)

        # Create attention mask for history (1 = real, 0 = padding)
        history_mask = (history_indices != self.pad_idx).float()

        # Create target mask (for loss computation, ignore PAD)
        target_mask = (target_indices != self.pad_idx).float()

        # Apply MEM masking if training
        if self.apply_mem_masking:
            history_indices, mem_labels = self._apply_mem_masking(
                history_indices, history_mask
            )
        else:
            mem_labels = torch.zeros_like(history_indices)

        # Context features
        store_idx = self.vocab['store_to_idx'].get(
            str(row['store_id']), 0
        )
        customer_idx = self.vocab['customer_to_idx'].get(
            str(row['customer_id']), 0
        )
        week = row['week']

        return {
            # History sequence: (max_history_length, max_basket_size)
            'history_indices': history_indices,
            'history_mask': history_mask,
            'mem_labels': mem_labels,

            # Target: (max_target_size,)
            'target_indices': target_indices,
            'target_mask': target_mask,

            # Context
            'customer_idx': torch.tensor(customer_idx, dtype=torch.long),
            'store_idx': torch.tensor(store_idx, dtype=torch.long),
            'week': torch.tensor(week, dtype=torch.long),

            # Metadata
            'is_cold_start': torch.tensor(row['is_cold_start'], dtype=torch.bool),
            'bucket': torch.tensor(row['bucket'], dtype=torch.long),
            'num_prior_baskets': torch.tensor(
                row['num_prior_baskets'], dtype=torch.long
            )
        }

    def _encode_history(
        self,
        history_products: List[List[str]]
    ) -> torch.Tensor:
        """
        Encode history as 2D tensor.

        Shape: (max_history_length, max_basket_size)
        Padding applied to both dimensions.
        """
        indices = torch.full(
            (self.max_history_length, self.max_basket_size),
            self.pad_idx,
            dtype=torch.long
        )

        # Take most recent baskets
        recent_history = history_products[-self.max_history_length:]

        for basket_idx, basket in enumerate(recent_history):
            products = basket[:self.max_basket_size]
            for prod_idx, prod in enumerate(products):
                indices[basket_idx, prod_idx] = self.vocab['product_to_idx'].get(
                    str(prod), self.unk_idx
                )

        return indices

    def _encode_target(self, target_products: List[str]) -> torch.Tensor:
        """Encode target basket as 1D tensor."""
        indices = torch.full(
            (self.max_target_size,),
            self.pad_idx,
            dtype=torch.long
        )

        products = target_products[:self.max_target_size]
        for idx, prod in enumerate(products):
            indices[idx] = self.vocab['product_to_idx'].get(
                str(prod), self.unk_idx
            )

        return indices

    def _apply_mem_masking(
        self,
        history_indices: torch.Tensor,
        history_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Masked Event Modeling (MEM) masking.

        Randomly mask 15% of products in history for self-supervised learning.
        """
        # Create copy for masking
        masked_indices = history_indices.clone()
        mem_labels = torch.full_like(history_indices, -100)  # Ignore index

        # Find real (non-PAD) positions
        real_positions = (history_indices != self.pad_idx).nonzero(as_tuple=False)

        if len(real_positions) == 0:
            return masked_indices, mem_labels

        # Select positions to mask
        n_to_mask = max(1, int(len(real_positions) * self.mask_ratio))
        mask_positions = real_positions[
            torch.randperm(len(real_positions))[:n_to_mask]
        ]

        for pos in mask_positions:
            i, j = pos[0].item(), pos[1].item()
            # Store original for loss computation
            mem_labels[i, j] = history_indices[i, j]
            # Apply masking
            masked_indices[i, j] = self.mask_idx

        return masked_indices, mem_labels


class BucketBatchSampler(Sampler):
    """
    Batch sampler that groups samples by history length bucket.

    This ensures efficient batching with minimal padding.
    """

    def __init__(
        self,
        samples_df: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.samples_df = samples_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Group indices by bucket
        self.bucket_indices = {}
        for bucket in samples_df['bucket'].unique():
            self.bucket_indices[bucket] = samples_df[
                samples_df['bucket'] == bucket
            ].index.tolist()

    def __iter__(self):
        batches = []

        for bucket, indices in self.bucket_indices.items():
            if self.shuffle:
                random.shuffle(indices)

            # Create batches for this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle batches (not within batches)
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        total = 0
        for indices in self.bucket_indices.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            total += n_batches
        return total


def create_dataloader(
    samples_path: Path,
    vocab: Dict,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    apply_mem_masking: bool = True,
    use_bucket_sampling: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for training/validation/testing.

    Parameters
    ----------
    samples_path : Path
        Path to samples parquet file
    vocab : Dict
        Vocabulary mappings
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle
    num_workers : int
        Number of data loading workers
    apply_mem_masking : bool
        Whether to apply MEM masking
    use_bucket_sampling : bool
        Whether to use bucket-based batching

    Returns
    -------
    DataLoader
    """
    # Load samples
    samples_df = pd.read_parquet(samples_path)

    # Create dataset
    dataset = RetailSimDataset(
        samples_df=samples_df,
        vocab=vocab,
        apply_mem_masking=apply_mem_masking,
        **kwargs
    )

    # Create sampler
    if use_bucket_sampling and 'bucket' in samples_df.columns:
        sampler = BucketBatchSampler(
            samples_df=samples_df,
            batch_size=batch_size,
            shuffle=shuffle
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


def create_dataloaders(
    data_dir: Path,
    vocab: Dict,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create all DataLoaders for training pipeline.

    Parameters
    ----------
    data_dir : Path
        Directory containing sample parquet files
    vocab : Dict
        Vocabulary mappings
    batch_size : int
        Batch size
    num_workers : int
        Number of workers

    Returns
    -------
    Dict with 'train', 'validation', 'test' DataLoaders
    """
    data_dir = Path(data_dir)
    loaders = {}

    # Training loaders (per bucket)
    train_files = list(data_dir.glob('train_bucket_*_samples.parquet'))
    if train_files:
        # Combine all training buckets
        train_dfs = [pd.read_parquet(f) for f in train_files]
        train_df = pd.concat(train_dfs, ignore_index=True)

        train_dataset = RetailSimDataset(
            samples_df=train_df,
            vocab=vocab,
            apply_mem_masking=True
        )

        train_sampler = BucketBatchSampler(
            samples_df=train_df,
            batch_size=batch_size,
            shuffle=True
        )

        loaders['train'] = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

    # Validation loader
    val_path = data_dir / 'validation_samples.parquet'
    if val_path.exists():
        loaders['validation'] = create_dataloader(
            val_path,
            vocab,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            apply_mem_masking=False,
            use_bucket_sampling=False
        )

    # Test loader
    test_path = data_dir / 'test_samples.parquet'
    if test_path.exists():
        loaders['test'] = create_dataloader(
            test_path,
            vocab,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            apply_mem_masking=False,
            use_bucket_sampling=False
        )

    return loaders


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    collated = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values

    return collated


if __name__ == '__main__':
    # Test dataloader
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    samples_dir = project_root / 'data' / 'prepared' / 'samples'
    cache_dir = project_root / 'data' / 'prepared' / 'tensor_cache'

    # Load vocab
    vocab_path = cache_dir / 'vocab.json'
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        print("Creating dataloaders...")
        loaders = create_dataloaders(
            samples_dir,
            vocab,
            batch_size=32,
            num_workers=0  # For testing
        )

        for name, loader in loaders.items():
            print(f"\n{name} loader:")
            print(f"  - Batches: {len(loader)}")

            # Test one batch
            batch = next(iter(loader))
            print(f"  - Batch keys: {list(batch.keys())}")
            print(f"  - History shape: {batch['history_indices'].shape}")
            print(f"  - Target shape: {batch['target_indices'].shape}")
    else:
        print(f"Vocab not found at {vocab_path}")
        print("Run stage4_tensor_cache.py first.")
