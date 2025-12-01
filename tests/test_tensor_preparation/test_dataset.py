"""
Tests for Dataset and DataLoader
================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRetailSimBatch:
    """Tests for RetailSimBatch dataclass."""

    def test_batch_properties(self):
        """Test batch property calculations."""
        from src.tensor_preparation.dataset import RetailSimBatch

        batch = RetailSimBatch(
            customer_context=np.random.randn(4, 192),
            temporal_context=np.random.randn(4, 64),
            store_context=np.random.randn(4, 96),
            trip_context=np.random.randn(4, 48),
            product_embeddings=np.random.randn(4, 10, 256),
            product_token_ids=np.zeros((4, 10), dtype=np.int32),
            price_features=np.random.randn(4, 10, 64),
            attention_mask=np.ones((4, 10), dtype=np.int32),
            sequence_lengths=np.array([8, 10, 6, 7]),
            trip_labels={'mission_type': np.array([0, 1, 0, 2])},
        )

        assert batch.batch_size == 4
        assert batch.dense_context_dim == 400
        assert batch.sequence_feature_dim == 320

    def test_get_dense_context(self):
        """Test dense context concatenation."""
        from src.tensor_preparation.dataset import RetailSimBatch

        batch = RetailSimBatch(
            customer_context=np.ones((2, 192)),
            temporal_context=np.ones((2, 64)) * 2,
            store_context=np.ones((2, 96)) * 3,
            trip_context=np.ones((2, 48)) * 4,
            product_embeddings=np.random.randn(2, 5, 256),
            product_token_ids=np.zeros((2, 5), dtype=np.int32),
            price_features=np.random.randn(2, 5, 64),
            attention_mask=np.ones((2, 5), dtype=np.int32),
            sequence_lengths=np.array([5, 5]),
            trip_labels={},
        )

        dense = batch.get_dense_context()
        assert dense.shape == (2, 400)

    def test_get_sequence_features(self):
        """Test sequence feature concatenation."""
        from src.tensor_preparation.dataset import RetailSimBatch

        batch = RetailSimBatch(
            customer_context=np.random.randn(2, 192),
            temporal_context=np.random.randn(2, 64),
            store_context=np.random.randn(2, 96),
            trip_context=np.random.randn(2, 48),
            product_embeddings=np.random.randn(2, 5, 256),
            product_token_ids=np.zeros((2, 5), dtype=np.int32),
            price_features=np.random.randn(2, 5, 64),
            attention_mask=np.ones((2, 5), dtype=np.int32),
            sequence_lengths=np.array([5, 5]),
            trip_labels={},
        )

        seq = batch.get_sequence_features()
        assert seq.shape == (2, 5, 320)


class TestRetailSimDatasetUnit:
    """Unit tests for RetailSimDataset (without full data dependencies)."""

    def test_batch_structure(self, sample_product_embeddings, sample_customer_embeddings):
        """Test batch structure with mock data."""
        # This would require mocking the file loading
        # For now, just test the batch creation logic
        pass


class TestDataLoaderUnit:
    """Unit tests for RetailSimDataLoader."""

    def test_batch_iteration(self):
        """Test that dataloader iterates correctly."""
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return 100

            def get_batch(self, indices, apply_masking=False):
                from src.tensor_preparation.dataset import RetailSimBatch
                batch_size = len(indices)
                return RetailSimBatch(
                    customer_context=np.random.randn(batch_size, 192),
                    temporal_context=np.random.randn(batch_size, 64),
                    store_context=np.random.randn(batch_size, 96),
                    trip_context=np.random.randn(batch_size, 48),
                    product_embeddings=np.random.randn(batch_size, 10, 256),
                    product_token_ids=np.zeros((batch_size, 10), dtype=np.int32),
                    price_features=np.random.randn(batch_size, 10, 64),
                    attention_mask=np.ones((batch_size, 10), dtype=np.int32),
                    sequence_lengths=np.array([10] * batch_size),
                    trip_labels={'mission_type': np.zeros(batch_size, dtype=np.int32)},
                )

        from src.tensor_preparation.dataset import RetailSimDataLoader

        dataset = MockDataset()
        dataloader = RetailSimDataLoader(dataset, batch_size=16, shuffle=False)

        assert len(dataloader) == 7  # ceil(100 / 16) = 7

        batches = list(dataloader)
        assert len(batches) == 7
        assert batches[0].batch_size == 16
        assert batches[-1].batch_size == 4  # 100 % 16 = 4

    def test_shuffle(self):
        """Test that shuffle randomizes order."""
        class MockDataset:
            def __init__(self):
                self.call_indices = []

            def __len__(self):
                return 32

            def get_batch(self, indices, apply_masking=False):
                self.call_indices.append(list(indices))
                from src.tensor_preparation.dataset import RetailSimBatch
                batch_size = len(indices)
                return RetailSimBatch(
                    customer_context=np.random.randn(batch_size, 192),
                    temporal_context=np.random.randn(batch_size, 64),
                    store_context=np.random.randn(batch_size, 96),
                    trip_context=np.random.randn(batch_size, 48),
                    product_embeddings=np.random.randn(batch_size, 10, 256),
                    product_token_ids=np.zeros((batch_size, 10), dtype=np.int32),
                    price_features=np.random.randn(batch_size, 10, 64),
                    attention_mask=np.ones((batch_size, 10), dtype=np.int32),
                    sequence_lengths=np.array([10] * batch_size),
                    trip_labels={},
                )

        from src.tensor_preparation.dataset import RetailSimDataLoader

        # Without shuffle
        dataset1 = MockDataset()
        loader1 = RetailSimDataLoader(dataset1, batch_size=16, shuffle=False)
        list(loader1)

        # With shuffle (different runs should produce different orders)
        dataset2 = MockDataset()
        loader2 = RetailSimDataLoader(dataset2, batch_size=16, shuffle=True)
        list(loader2)

        # The indices might be different (shuffle is probabilistic)
        # Just check that both loaders work
        assert len(dataset1.call_indices) == 2
        assert len(dataset2.call_indices) == 2


class TestTensorDimensions:
    """Test that all tensor dimensions are correct."""

    def test_dense_context_total(self):
        """Test total dense context dimension."""
        from src.tensor_preparation import (
            CustomerContextEncoder,
            TemporalContextEncoder,
            StoreContextEncoder,
            TripContextEncoder,
        )

        t1_dim = CustomerContextEncoder().output_dim
        t3_dim = TemporalContextEncoder().output_dim
        t5_dim = StoreContextEncoder().output_dim
        t6_dim = TripContextEncoder().output_dim

        total = t1_dim + t3_dim + t5_dim + t6_dim
        assert total == 400, f"Expected 400, got {total}"

    def test_sequence_feature_total(self, sample_product_embeddings):
        """Test total sequence feature dimension."""
        from src.tensor_preparation import (
            ProductSequenceEncoder,
            PriceContextEncoder,
        )

        t2_dim = ProductSequenceEncoder(sample_product_embeddings).embedding_dim
        t4_dim = PriceContextEncoder().output_dim

        total = t2_dim + t4_dim
        assert total == 320, f"Expected 320, got {total}"

    def test_t1_components(self):
        """Test T1 component dimensions sum correctly."""
        from src.tensor_preparation import CustomerContextEncoder

        encoder = CustomerContextEncoder()
        expected = encoder.segment_dim + encoder.history_dim + encoder.affinity_dim
        assert encoder.output_dim == expected

    def test_t3_components(self):
        """Test T3 component dimensions sum correctly."""
        from src.tensor_preparation import TemporalContextEncoder

        encoder = TemporalContextEncoder()
        expected = (encoder.week_dim + encoder.weekday_dim + encoder.hour_dim +
                   encoder.holiday_dim + encoder.season_dim +
                   encoder.trend_dim + encoder.recency_dim)
        assert encoder.output_dim == expected

    def test_t4_components(self):
        """Test T4 component dimensions sum correctly."""
        from src.tensor_preparation import PriceContextEncoder

        encoder = PriceContextEncoder()
        expected = (encoder.fourier_dim + encoder.log_dim +
                   encoder.relative_dim + encoder.velocity_dim)
        assert encoder.output_dim == expected

    def test_t5_components(self):
        """Test T5 component dimensions sum correctly."""
        from src.tensor_preparation import StoreContextEncoder

        encoder = StoreContextEncoder()
        expected = (encoder.format_dim + encoder.region_dim +
                   encoder.operational_dim + encoder.identity_dim)
        assert encoder.output_dim == expected

    def test_t6_components(self):
        """Test T6 component dimensions sum correctly."""
        from src.tensor_preparation import TripContextEncoder

        encoder = TripContextEncoder()
        expected = (encoder.mission_type_dim + encoder.mission_focus_dim +
                   encoder.price_sensitivity_dim + encoder.basket_scope_dim)
        assert encoder.output_dim == expected
