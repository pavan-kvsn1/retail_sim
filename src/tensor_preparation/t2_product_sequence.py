"""
T2: Product Sequence Tensor [256d per item]
============================================
Variable-length product sequences with special tokens and sparse representation.

Special Tokens:
- [PAD] (token_id=0): Padding for batching
- [MASK] (token_id=5001): Masked Event Modeling
- [EOS] (token_id=5002): End of sequence

Features:
- GraphSAGE product embeddings
- Positional encoding
- Sparse COO tensor format
- BERT-style masking for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import pickle
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ProductSequenceBatch:
    """Container for batched product sequence data."""
    # Product embeddings [B, S, 256]
    embeddings: np.ndarray
    # Token IDs [B, S]
    token_ids: np.ndarray
    # Attention mask [B, S] - 1 for real tokens, 0 for PAD
    attention_mask: np.ndarray
    # Sequence lengths [B]
    lengths: np.ndarray
    # For masked language modeling
    masked_positions: Optional[np.ndarray] = None
    masked_targets: Optional[np.ndarray] = None


class ProductSequenceEncoder:
    """
    Encodes product sequences for T2 tensor.

    Special token vocabulary:
    - 0: [PAD]
    - 1-5000: Product SKUs
    - 5001: [MASK]
    - 5002: [EOS]
    """

    # Special token IDs
    PAD_TOKEN_ID = 0
    MASK_TOKEN_ID = 5001
    EOS_TOKEN_ID = 5002

    def __init__(
        self,
        product_embeddings: Dict[str, np.ndarray],
        embedding_dim: int = 256,
        max_seq_len: int = 50,
        mask_prob: float = 0.15
    ):
        """
        Parameters
        ----------
        product_embeddings : Dict[str, np.ndarray]
            Pre-computed GraphSAGE embeddings
        embedding_dim : int
            Product embedding dimension (default 256)
        max_seq_len : int
            Maximum sequence length (default 50)
        mask_prob : float
            Probability of masking tokens during training
        """
        self.product_embeddings = product_embeddings
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob

        # Build product vocabulary
        self._build_vocabulary()

        # Initialize special token embeddings
        self._init_special_embeddings()

        # Pre-compute positional encodings
        self._init_positional_encoding()

    def _build_vocabulary(self):
        """Build product ID to token ID mapping."""
        products = sorted(self.product_embeddings.keys())

        # Reserve 0 for PAD, start products at 1
        self.product_to_token = {prod: i + 1 for i, prod in enumerate(products)}
        self.token_to_product = {i + 1: prod for i, prod in enumerate(products)}

        # Special tokens
        self.token_to_product[self.PAD_TOKEN_ID] = '[PAD]'
        self.token_to_product[self.MASK_TOKEN_ID] = '[MASK]'
        self.token_to_product[self.EOS_TOKEN_ID] = '[EOS]'

        self.vocab_size = len(products) + 3  # products + PAD + MASK + EOS

    def _init_special_embeddings(self):
        """Initialize embeddings for special tokens."""
        np.random.seed(42)

        # PAD: Zero vector
        self.pad_embedding = np.zeros(self.embedding_dim)

        # MASK: Learnable embedding (random init)
        self.mask_embedding = np.random.randn(self.embedding_dim) * 0.1

        # EOS: Learnable embedding (random init)
        self.eos_embedding = np.random.randn(self.embedding_dim) * 0.1

    def _init_positional_encoding(self):
        """Pre-compute sinusoidal positional encodings."""
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * (-np.log(10000.0) / self.embedding_dim))

        self.positional_encoding = np.zeros((self.max_seq_len, self.embedding_dim))
        self.positional_encoding[:, 0::2] = np.sin(position * div_term)
        self.positional_encoding[:, 1::2] = np.cos(position * div_term)

    def get_embedding(self, token_id: int) -> np.ndarray:
        """Get embedding for a single token."""
        if token_id == self.PAD_TOKEN_ID:
            return self.pad_embedding
        elif token_id == self.MASK_TOKEN_ID:
            return self.mask_embedding
        elif token_id == self.EOS_TOKEN_ID:
            return self.eos_embedding
        else:
            product_id = self.token_to_product.get(token_id)
            if product_id and product_id in self.product_embeddings:
                return self.product_embeddings[product_id]
            return np.zeros(self.embedding_dim)

    def encode_sequence(
        self,
        product_ids: List[str],
        add_eos: bool = True,
        apply_masking: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int, Optional[List], Optional[List]]:
        """
        Encode a single product sequence.

        Parameters
        ----------
        product_ids : List[str]
            List of product IDs in basket
        add_eos : bool
            Whether to add EOS token at end
        apply_masking : bool
            Whether to apply BERT-style masking

        Returns
        -------
        embeddings : np.ndarray [S, 256]
            Product embeddings with positional encoding
        token_ids : np.ndarray [S]
            Token IDs
        length : int
            Actual sequence length (before padding)
        masked_positions : List[int], optional
            Positions of masked tokens
        masked_targets : List[int], optional
            Original token IDs at masked positions
        """
        # Convert product IDs to token IDs
        token_ids = []
        for prod in product_ids:
            token_id = self.product_to_token.get(prod, 0)
            if token_id > 0:
                token_ids.append(token_id)

        # Add EOS token
        if add_eos:
            token_ids.append(self.EOS_TOKEN_ID)

        # Truncate if needed
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.EOS_TOKEN_ID]

        length = len(token_ids)

        # Apply masking for training
        masked_positions = None
        masked_targets = None
        if apply_masking and length > 1:
            token_ids, masked_positions, masked_targets = self._apply_masking(token_ids)

        # Get embeddings
        embeddings = np.array([self.get_embedding(tid) for tid in token_ids])

        # Add positional encoding
        embeddings = embeddings + self.positional_encoding[:length]

        return embeddings, np.array(token_ids), length, masked_positions, masked_targets

    def _apply_masking(
        self,
        token_ids: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Apply BERT-style masking.

        Strategy:
        - 80%: Replace with [MASK]
        - 10%: Replace with random token
        - 10%: Keep original
        """
        masked_ids = token_ids.copy()
        masked_positions = []
        masked_targets = []

        # Don't mask EOS token
        maskable_positions = [i for i, tid in enumerate(token_ids) if tid != self.EOS_TOKEN_ID]

        # Select positions to mask
        num_to_mask = max(1, int(len(maskable_positions) * self.mask_prob))
        positions_to_mask = np.random.choice(
            maskable_positions, size=min(num_to_mask, len(maskable_positions)), replace=False
        )

        for pos in positions_to_mask:
            masked_targets.append(token_ids[pos])
            masked_positions.append(pos)

            rand = np.random.random()
            if rand < 0.8:
                # Replace with [MASK]
                masked_ids[pos] = self.MASK_TOKEN_ID
            elif rand < 0.9:
                # Replace with random token (not special tokens)
                masked_ids[pos] = np.random.randint(1, self.vocab_size - 2)
            # else: keep original (10%)

        return masked_ids, masked_positions, masked_targets

    def encode_batch(
        self,
        baskets: List[List[str]],
        apply_masking: bool = False
    ) -> ProductSequenceBatch:
        """
        Encode a batch of product sequences.

        Parameters
        ----------
        baskets : List[List[str]]
            List of baskets, each basket is a list of product IDs
        apply_masking : bool
            Whether to apply masking for training

        Returns
        -------
        ProductSequenceBatch
            Batched tensor data
        """
        batch_size = len(baskets)

        # Encode all sequences
        all_embeddings = []
        all_token_ids = []
        all_lengths = []
        all_masked_positions = []
        all_masked_targets = []

        for basket in baskets:
            emb, tids, length, mask_pos, mask_tgt = self.encode_sequence(
                basket, add_eos=True, apply_masking=apply_masking
            )
            all_embeddings.append(emb)
            all_token_ids.append(tids)
            all_lengths.append(length)
            if mask_pos:
                all_masked_positions.append(mask_pos)
                all_masked_targets.append(mask_tgt)

        # Find max length in batch
        max_len = max(all_lengths)

        # Pad sequences
        padded_embeddings = np.zeros((batch_size, max_len, self.embedding_dim))
        padded_token_ids = np.zeros((batch_size, max_len), dtype=np.int32)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)

        for i, (emb, tids, length) in enumerate(zip(all_embeddings, all_token_ids, all_lengths)):
            padded_embeddings[i, :length] = emb
            padded_token_ids[i, :length] = tids
            attention_mask[i, :length] = 1

        # Handle masked positions
        masked_positions = None
        masked_targets = None
        if all_masked_positions:
            # Pad masked positions to same length
            max_masks = max(len(mp) for mp in all_masked_positions)
            masked_positions = np.zeros((batch_size, max_masks), dtype=np.int32)
            masked_targets = np.zeros((batch_size, max_masks), dtype=np.int32)

            for i, (pos, tgt) in enumerate(zip(all_masked_positions, all_masked_targets)):
                masked_positions[i, :len(pos)] = pos
                masked_targets[i, :len(tgt)] = tgt

        return ProductSequenceBatch(
            embeddings=padded_embeddings,
            token_ids=padded_token_ids,
            attention_mask=attention_mask,
            lengths=np.array(all_lengths),
            masked_positions=masked_positions,
            masked_targets=masked_targets
        )

    def decode_sequence(self, token_ids: np.ndarray) -> List[str]:
        """Convert token IDs back to product IDs."""
        products = []
        for tid in token_ids:
            if tid == self.PAD_TOKEN_ID:
                continue
            elif tid == self.EOS_TOKEN_ID:
                break
            elif tid == self.MASK_TOKEN_ID:
                products.append('[MASK]')
            else:
                prod = self.token_to_product.get(tid, f'UNK_{tid}')
                products.append(prod)
        return products


def main():
    """Test product sequence encoding."""
    project_root = Path(__file__).parent.parent.parent

    # Load product embeddings
    print("Loading product embeddings...")
    embed_path = project_root / 'data' / 'features' / 'product_embeddings.pkl'

    with open(embed_path, 'rb') as f:
        data = pickle.load(f)
        product_embeddings = data.get('embeddings', {})

    print(f"  Loaded {len(product_embeddings)} product embeddings")

    # Create encoder
    encoder = ProductSequenceEncoder(product_embeddings)
    print(f"  Vocabulary size: {encoder.vocab_size}")

    # Test single sequence encoding
    sample_products = list(product_embeddings.keys())[:6]
    print(f"\nTest basket: {sample_products}")

    emb, tids, length, _, _ = encoder.encode_sequence(sample_products)
    print(f"  Embeddings shape: {emb.shape}")
    print(f"  Token IDs: {tids}")
    print(f"  Length: {length}")

    # Test batch encoding with masking
    baskets = [
        list(product_embeddings.keys())[:6],
        list(product_embeddings.keys())[10:18],
        list(product_embeddings.keys())[20:23],
    ]

    batch = encoder.encode_batch(baskets, apply_masking=True)
    print(f"\nBatch encoding:")
    print(f"  Embeddings shape: {batch.embeddings.shape}")
    print(f"  Token IDs shape: {batch.token_ids.shape}")
    print(f"  Attention mask shape: {batch.attention_mask.shape}")
    print(f"  Lengths: {batch.lengths}")

    if batch.masked_positions is not None:
        print(f"  Masked positions shape: {batch.masked_positions.shape}")

    return encoder


if __name__ == '__main__':
    main()
