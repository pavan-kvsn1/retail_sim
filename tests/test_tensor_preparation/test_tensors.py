"""
Tests for Tensor Encoders (T1-T6)
=================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tensor_preparation.t1_customer_context import CustomerContextEncoder
from src.tensor_preparation.t2_product_sequence import ProductSequenceEncoder
from src.tensor_preparation.t3_temporal_context import TemporalContextEncoder
from src.tensor_preparation.t4_price_context import PriceContextEncoder
from src.tensor_preparation.t5_store_context import StoreContextEncoder
from src.tensor_preparation.t6_trip_context import TripContextEncoder


class TestT1CustomerContext:
    """Tests for T1: Customer Context Tensor [192d]."""

    def test_init(self):
        """Test initialization."""
        encoder = CustomerContextEncoder()
        assert encoder.output_dim == 192

    def test_output_dimension(self):
        """Test output has correct dimension."""
        encoder = CustomerContextEncoder()
        t1 = encoder.encode_customer(
            customer_id='C1',
            seg1='CT',
            seg2='DI',
            history_embedding=np.random.randn(160),
            num_trips=10
        )
        assert t1.shape == (192,)

    def test_no_nan_output(self):
        """Test output has no NaN values."""
        encoder = CustomerContextEncoder()
        t1 = encoder.encode_customer(
            customer_id='C1',
            seg1='CT',
            seg2='DI',
            history_embedding=np.random.randn(160),
            num_trips=10
        )
        assert not np.isnan(t1).any()

    def test_cold_start_handling(self):
        """Test cold-start customer handling."""
        encoder = CustomerContextEncoder()

        # New customer (1 trip)
        t1_new = encoder.encode_customer(
            customer_id='C1',
            seg1='CT',
            seg2='DI',
            history_embedding=np.random.randn(160),
            num_trips=1
        )

        # Established customer (20 trips)
        t1_established = encoder.encode_customer(
            customer_id='C2',
            seg1='CT',
            seg2='DI',
            history_embedding=np.random.randn(160),
            num_trips=20
        )

        # Both should be valid
        assert t1_new.shape == (192,)
        assert t1_established.shape == (192,)

    def test_unknown_segment(self):
        """Test handling of unknown segment codes."""
        encoder = CustomerContextEncoder()
        t1 = encoder.encode_customer(
            customer_id='C1',
            seg1='UNKNOWN',
            seg2='UNKNOWN',
            history_embedding=np.random.randn(160),
            num_trips=10
        )
        assert t1.shape == (192,)
        assert not np.isnan(t1).any()

    def test_no_history_embedding(self):
        """Test handling when no history embedding provided."""
        encoder = CustomerContextEncoder()
        t1 = encoder.encode_customer(
            customer_id='C1',
            seg1='CT',
            seg2='DI',
            history_embedding=None,
            num_trips=0
        )
        assert t1.shape == (192,)
        assert not np.isnan(t1).any()


class TestT2ProductSequence:
    """Tests for T2: Product Sequence Tensor [256d/item]."""

    def test_init(self, sample_product_embeddings):
        """Test initialization."""
        encoder = ProductSequenceEncoder(sample_product_embeddings)
        assert encoder.embedding_dim == 256

    def test_special_tokens(self, sample_product_embeddings):
        """Test special token IDs."""
        encoder = ProductSequenceEncoder(sample_product_embeddings)
        assert encoder.PAD_TOKEN_ID == 0
        assert encoder.MASK_TOKEN_ID == 5001
        assert encoder.EOS_TOKEN_ID == 5002

    def test_single_sequence(self, sample_product_embeddings):
        """Test single sequence encoding."""
        encoder = ProductSequenceEncoder(sample_product_embeddings)
        products = list(sample_product_embeddings.keys())[:5]

        emb, tids, length, _, _ = encoder.encode_sequence(products, add_eos=True)

        assert emb.shape == (6, 256)  # 5 products + EOS
        assert len(tids) == 6
        assert length == 6
        assert tids[-1] == encoder.EOS_TOKEN_ID

    def test_batch_encoding(self, sample_product_embeddings):
        """Test batch encoding."""
        encoder = ProductSequenceEncoder(sample_product_embeddings)
        products = list(sample_product_embeddings.keys())

        baskets = [
            products[:3],
            products[5:10],
            products[10:12],
        ]

        batch = encoder.encode_batch(baskets, apply_masking=False)

        assert batch.embeddings.shape[0] == 3  # Batch size
        assert batch.embeddings.shape[2] == 256  # Embedding dim
        assert len(batch.lengths) == 3

    def test_masking(self, sample_product_embeddings):
        """Test BERT-style masking."""
        encoder = ProductSequenceEncoder(sample_product_embeddings, mask_prob=0.5)
        products = list(sample_product_embeddings.keys())[:10]

        emb, tids, length, mask_pos, mask_tgt = encoder.encode_sequence(
            products, add_eos=True, apply_masking=True
        )

        # Should have some masked positions
        assert mask_pos is not None
        assert mask_tgt is not None
        assert len(mask_pos) > 0

    def test_padding(self, sample_product_embeddings):
        """Test padding in batch."""
        encoder = ProductSequenceEncoder(sample_product_embeddings)
        products = list(sample_product_embeddings.keys())

        baskets = [
            products[:3],   # 3 + EOS = 4
            products[:10],  # 10 + EOS = 11
        ]

        batch = encoder.encode_batch(baskets)

        # Both should be padded to max length
        assert batch.embeddings.shape[1] == 11
        # Attention mask should reflect actual lengths
        assert batch.attention_mask[0, 4:].sum() == 0  # Padding
        assert batch.attention_mask[1, :11].sum() == 11  # All valid

    def test_decode_sequence(self, sample_product_embeddings):
        """Test decoding token IDs back to products."""
        encoder = ProductSequenceEncoder(sample_product_embeddings)
        products = list(sample_product_embeddings.keys())[:3]

        _, tids, _, _, _ = encoder.encode_sequence(products, add_eos=True)
        decoded = encoder.decode_sequence(tids)

        assert decoded == products  # EOS is not included in decode


class TestT3TemporalContext:
    """Tests for T3: Temporal Context Tensor [64d]."""

    def test_init(self):
        """Test initialization."""
        encoder = TemporalContextEncoder()
        assert encoder.output_dim == 64

    def test_output_dimension(self):
        """Test output has correct dimension."""
        encoder = TemporalContextEncoder()
        t3 = encoder.encode_temporal(
            shop_week=200626,
            shop_weekday=3,
            shop_hour=14
        )
        assert t3.shape == (64,)

    def test_no_nan_output(self):
        """Test output has no NaN values."""
        encoder = TemporalContextEncoder()
        t3 = encoder.encode_temporal(
            shop_week=200626,
            shop_weekday=3,
            shop_hour=14
        )
        assert not np.isnan(t3).any()

    def test_holiday_encoding(self):
        """Test holiday week encoding."""
        encoder = TemporalContextEncoder()

        # Week 52 is a holiday week
        t3_holiday = encoder.encode_temporal(shop_week=200652, shop_weekday=1, shop_hour=10)

        # Week 10 is not a holiday
        t3_regular = encoder.encode_temporal(shop_week=200610, shop_weekday=1, shop_hour=10)

        # Should produce different encodings
        assert not np.allclose(t3_holiday, t3_regular)

    def test_recency_encoding(self):
        """Test recency feature encoding."""
        encoder = TemporalContextEncoder()

        # Recent visitor (last week)
        t3_recent = encoder.encode_temporal(
            shop_week=200626,
            shop_weekday=1,
            shop_hour=10,
            last_visit_week=200625
        )

        # Lapsed visitor (6 months ago)
        t3_lapsed = encoder.encode_temporal(
            shop_week=200626,
            shop_weekday=1,
            shop_hour=10,
            last_visit_week=200601
        )

        assert t3_recent.shape == (64,)
        assert t3_lapsed.shape == (64,)

    def test_batch_encoding(self, mini_transactions):
        """Test batch encoding."""
        encoder = TemporalContextEncoder()
        batch = encoder.encode_batch(mini_transactions.head(10))

        assert batch.shape == (10, 64)


class TestT4PriceContext:
    """Tests for T4: Price Context Tensor [64d/item]."""

    def test_init(self):
        """Test initialization."""
        encoder = PriceContextEncoder()
        assert encoder.output_dim == 64

    def test_single_price(self):
        """Test single price encoding."""
        encoder = PriceContextEncoder()
        t4 = encoder.encode_price(
            actual_price=1.99,
            base_price=2.49,
            category_avg_price=2.20,
            prior_price=2.19
        )
        assert t4.shape == (64,)

    def test_no_nan_output(self):
        """Test output has no NaN values."""
        encoder = PriceContextEncoder()
        t4 = encoder.encode_price(
            actual_price=1.99,
            base_price=2.49,
            category_avg_price=2.20
        )
        assert not np.isnan(t4).any()

    def test_basket_prices(self):
        """Test basket price encoding."""
        encoder = PriceContextEncoder()
        products = ['P1', 'P2', 'P3']
        price_lookup = {
            'P1': {'actual_price': 1.99, 'base_price': 2.49},
            'P2': {'actual_price': 3.50, 'base_price': 3.50},
            'P3': {'actual_price': 0.99, 'base_price': 1.29},
        }
        category_avg = {'P1': 2.20, 'P2': 3.80, 'P3': 1.10}

        result = encoder.encode_basket_prices(products, price_lookup, category_avg)

        assert result.shape == (3, 64)

    def test_batch_encoding(self):
        """Test batch encoding."""
        encoder = PriceContextEncoder()
        baskets = [['P1', 'P2'], ['P1', 'P2', 'P3'], ['P3']]
        price_lookup = {
            'P1': {'actual_price': 1.99},
            'P2': {'actual_price': 3.50},
            'P3': {'actual_price': 0.99},
        }
        category_avg = {'P1': 2.0, 'P2': 3.5, 'P3': 1.0}

        batch = encoder.encode_batch(baskets, price_lookup, category_avg, max_seq_len=4)

        assert batch.features.shape[0] == 3  # Batch size
        assert batch.features.shape[1] == 4  # Padded to max_seq_len
        assert batch.features.shape[2] == 64  # Feature dim


class TestT5StoreContext:
    """Tests for T5: Store Context Tensor [96d]."""

    def test_init(self):
        """Test initialization."""
        encoder = StoreContextEncoder()
        assert encoder.output_dim == 96

    def test_single_store(self):
        """Test single store encoding."""
        encoder = StoreContextEncoder()
        t5 = encoder.encode_store(
            store_id='STORE001',
            store_format='LS',
            store_region='E01'
        )
        assert t5.shape == (96,)

    def test_no_nan_output(self):
        """Test output has no NaN values."""
        encoder = StoreContextEncoder()
        t5 = encoder.encode_store(
            store_id='STORE001',
            store_format='LS',
            store_region='E01'
        )
        assert not np.isnan(t5).any()

    def test_operational_features(self):
        """Test operational features included."""
        encoder = StoreContextEncoder()
        t5 = encoder.encode_store(
            store_id='STORE001',
            store_format='LS',
            store_region='E01',
            operational_features={
                'store_size': 0.8,
                'traffic': 0.7,
                'competition': 0.3,
                'store_age': 0.9
            }
        )
        assert t5.shape == (96,)

    def test_unknown_format(self):
        """Test handling of unknown store format."""
        encoder = StoreContextEncoder()
        t5 = encoder.encode_store(
            store_id='STORE001',
            store_format='UNKNOWN',
            store_region='E01'
        )
        assert t5.shape == (96,)
        assert not np.isnan(t5).any()


class TestT6TripContext:
    """Tests for T6: Trip Context Tensor [48d]."""

    def test_init(self):
        """Test initialization."""
        encoder = TripContextEncoder()
        assert encoder.output_dim == 48

    def test_single_trip(self):
        """Test single trip encoding."""
        encoder = TripContextEncoder()
        t6 = encoder.encode_trip(
            mission_type='Full Shop',
            mission_focus='Fresh',
            price_sensitivity='MM',
            basket_size='L'
        )
        assert t6.shape == (48,)

    def test_no_nan_output(self):
        """Test output has no NaN values."""
        encoder = TripContextEncoder()
        t6 = encoder.encode_trip(
            mission_type='Top Up',
            mission_focus='Grocery',
            price_sensitivity='LA',
            basket_size='S'
        )
        assert not np.isnan(t6).any()

    def test_unknown_values(self):
        """Test handling of unknown values."""
        encoder = TripContextEncoder()
        t6 = encoder.encode_trip(
            mission_type='UNKNOWN',
            mission_focus='UNKNOWN',
            price_sensitivity='UNKNOWN',
            basket_size='UNKNOWN'
        )
        assert t6.shape == (48,)
        assert not np.isnan(t6).any()

    def test_batch_encoding(self, mini_transactions):
        """Test batch encoding with labels."""
        encoder = TripContextEncoder()
        tensors, labels = encoder.encode_batch(mini_transactions.head(10))

        assert tensors.shape == (10, 48)
        assert 'mission_type' in labels
        assert 'mission_focus' in labels
        assert len(labels['mission_type']) == 10

    def test_sampling(self):
        """Test mission sampling from distribution."""
        encoder = TripContextEncoder()
        patterns = {
            'mission_type_dist': {'Top Up': 0.7, 'Full Shop': 0.3},
            'mission_focus_dist': {'Fresh': 0.5, 'Grocery': 0.3, 'Mixed': 0.2},
            'mean_price_sensitivity': 0.6,
            'mean_basket_size': 0.45
        }

        mission_type, focus, sensitivity, size = encoder.sample_from_distribution(patterns)

        assert mission_type in ['Top Up', 'Full Shop']
        assert focus in ['Fresh', 'Grocery', 'Mixed']
        assert sensitivity in ['LA', 'MM', 'UM']
        assert size in ['S', 'M', 'L']
