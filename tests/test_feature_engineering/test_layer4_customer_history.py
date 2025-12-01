"""
Tests for Layer 4: Customer History Encoding
============================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering.layer4_customer_history import CustomerHistoryEncoder


class TestCustomerHistoryEncoder:
    """Test suite for CustomerHistoryEncoder."""

    def test_init(self):
        """Test initialization."""
        encoder = CustomerHistoryEncoder()
        assert encoder is not None
        assert encoder.output_dim == 160

    def test_init_custom_dim(self):
        """Test initialization with custom dimension."""
        encoder = CustomerHistoryEncoder(output_dim=128)
        assert encoder.output_dim == 128

    def test_basic_encoding(self, mini_transactions, sample_product_embeddings):
        """Test basic customer history encoding."""
        # Create mission patterns
        customers = mini_transactions['CUST_CODE'].unique()[:10]
        mission_patterns = pd.DataFrame({
            'customer_id': customers,
            'p_mission_top_up': np.random.uniform(0, 1, len(customers)),
            'p_mission_full_shop': np.random.uniform(0, 1, len(customers)),
        })

        encoder = CustomerHistoryEncoder(output_dim=64)
        embeddings = encoder.run(mini_transactions, sample_product_embeddings, mission_patterns)

        assert len(embeddings) > 0

    def test_embedding_dimension(self, mini_transactions, sample_product_embeddings):
        """Test that embeddings have correct dimension."""
        customers = mini_transactions['CUST_CODE'].unique()[:10]
        mission_patterns = pd.DataFrame({
            'customer_id': customers,
        })

        encoder = CustomerHistoryEncoder(output_dim=160)
        embeddings = encoder.run(mini_transactions, sample_product_embeddings, mission_patterns)

        for embed in embeddings.values():
            assert embed.shape == (160,)

    def test_no_nan_embeddings(self, mini_transactions, sample_product_embeddings):
        """Test that embeddings have no NaN values."""
        customers = mini_transactions['CUST_CODE'].unique()[:10]
        mission_patterns = pd.DataFrame({'customer_id': customers})

        encoder = CustomerHistoryEncoder()
        embeddings = encoder.run(mini_transactions, sample_product_embeddings, mission_patterns)

        for cust_id, embed in embeddings.items():
            assert not np.isnan(embed).any(), f"NaN in embedding for {cust_id}"

    def test_no_infinite_embeddings(self, mini_transactions, sample_product_embeddings):
        """Test that embeddings have no infinite values."""
        customers = mini_transactions['CUST_CODE'].unique()[:10]
        mission_patterns = pd.DataFrame({'customer_id': customers})

        encoder = CustomerHistoryEncoder()
        embeddings = encoder.run(mini_transactions, sample_product_embeddings, mission_patterns)

        for cust_id, embed in embeddings.items():
            assert np.isfinite(embed).all(), f"Infinite values in embedding for {cust_id}"


class TestColdStartHandling:
    """Test cold-start handling in customer encoding."""

    def test_new_customer(self, sample_product_embeddings):
        """Test encoding for customer with few trips."""
        # Customer with only 2 trips
        transactions = pd.DataFrame({
            'CUST_CODE': ['C1', 'C1'],
            'PROD_CODE': ['PRD00000000', 'PRD00000001'],
            'BASKET_ID': ['B1', 'B2'],
            'SHOP_WEEK': [1, 2],
            'SHOP_HOUR': [10, 14],
            'SHOP_WEEKDAY': [1, 3],
            'BASKET_TYPE': ['Top Up', 'Top Up'],
            'BASKET_DOMINANT_MISSION': ['Fresh', 'Fresh'],
        })

        mission_patterns = pd.DataFrame({'customer_id': ['C1']})

        encoder = CustomerHistoryEncoder(output_dim=64)
        embeddings = encoder.run(transactions, sample_product_embeddings, mission_patterns)

        assert 'C1' in embeddings
        assert embeddings['C1'].shape == (64,)

    def test_established_customer(self, sample_product_embeddings):
        """Test encoding for customer with many trips."""
        # Customer with 20 trips
        transactions = pd.DataFrame({
            'CUST_CODE': ['C1'] * 20,
            'PROD_CODE': [f'PRD{i:08d}' for i in range(20)],
            'BASKET_ID': [f'B{i}' for i in range(20)],
            'SHOP_WEEK': list(range(20)),
            'SHOP_HOUR': [10] * 20,
            'SHOP_WEEKDAY': [1] * 20,
            'BASKET_TYPE': ['Top Up'] * 20,
            'BASKET_DOMINANT_MISSION': ['Fresh'] * 20,
        })

        mission_patterns = pd.DataFrame({'customer_id': ['C1']})

        encoder = CustomerHistoryEncoder(output_dim=64)
        embeddings = encoder.run(transactions, sample_product_embeddings, mission_patterns)

        assert 'C1' in embeddings


class TestCustomerHistoryEdgeCases:
    """Test edge cases for customer history encoding."""

    def test_empty_transactions(self, sample_product_embeddings):
        """Test with empty transactions."""
        transactions = pd.DataFrame(columns=[
            'CUST_CODE', 'PROD_CODE', 'BASKET_ID', 'SHOP_WEEK',
            'SHOP_HOUR', 'SHOP_WEEKDAY', 'BASKET_TYPE', 'BASKET_DOMINANT_MISSION'
        ])
        mission_patterns = pd.DataFrame(columns=['customer_id'])

        encoder = CustomerHistoryEncoder()
        embeddings = encoder.run(transactions, sample_product_embeddings, mission_patterns)

        assert len(embeddings) == 0

    def test_single_transaction(self, sample_product_embeddings):
        """Test customer with single transaction."""
        transactions = pd.DataFrame({
            'CUST_CODE': ['C1'],
            'PROD_CODE': ['PRD00000000'],
            'BASKET_ID': ['B1'],
            'SHOP_WEEK': [1],
            'SHOP_HOUR': [10],
            'SHOP_WEEKDAY': [1],
            'BASKET_TYPE': ['Top Up'],
            'BASKET_DOMINANT_MISSION': ['Fresh'],
        })

        mission_patterns = pd.DataFrame({'customer_id': ['C1']})

        encoder = CustomerHistoryEncoder(output_dim=64)
        embeddings = encoder.run(transactions, sample_product_embeddings, mission_patterns)

        assert 'C1' in embeddings

    def test_unknown_products(self):
        """Test handling of products not in embedding dictionary."""
        transactions = pd.DataFrame({
            'CUST_CODE': ['C1', 'C1'],
            'PROD_CODE': ['UNKNOWN_PROD1', 'UNKNOWN_PROD2'],
            'BASKET_ID': ['B1', 'B2'],
            'SHOP_WEEK': [1, 2],
            'SHOP_HOUR': [10, 14],
            'SHOP_WEEKDAY': [1, 3],
            'BASKET_TYPE': ['Top Up', 'Top Up'],
            'BASKET_DOMINANT_MISSION': ['Fresh', 'Fresh'],
        })

        product_embeddings = {}  # Empty - no known products

        mission_patterns = pd.DataFrame({'customer_id': ['C1']})

        encoder = CustomerHistoryEncoder(output_dim=64)
        embeddings = encoder.run(transactions, product_embeddings, mission_patterns)

        # Should still produce valid embedding
        if 'C1' in embeddings:
            assert np.isfinite(embeddings['C1']).all()

    def test_save_and_load(self, mini_transactions, sample_product_embeddings, temp_dir):
        """Test saving and loading embeddings."""
        customers = mini_transactions['CUST_CODE'].unique()[:10]
        mission_patterns = pd.DataFrame({'customer_id': customers})

        encoder = CustomerHistoryEncoder(output_dim=64)
        embeddings = encoder.run(mini_transactions, sample_product_embeddings, mission_patterns)

        save_path = temp_dir / 'test_customer_embeddings.pkl'
        encoder.save(str(save_path))

        assert save_path.exists()

        # Load and verify
        loaded = encoder.load(str(save_path))
        assert len(loaded) == len(embeddings)


class TestCustomerHistoryDeterminism:
    """Test that encoding is deterministic."""

    def test_deterministic_embeddings(self, mini_transactions, sample_product_embeddings):
        """Test that same input produces same output."""
        customers = mini_transactions['CUST_CODE'].unique()[:5]
        mission_patterns = pd.DataFrame({'customer_id': customers})

        encoder1 = CustomerHistoryEncoder(output_dim=64)
        embeddings1 = encoder1.run(mini_transactions, sample_product_embeddings, mission_patterns)

        encoder2 = CustomerHistoryEncoder(output_dim=64)
        embeddings2 = encoder2.run(mini_transactions, sample_product_embeddings, mission_patterns)

        for cust_id in embeddings1:
            if cust_id in embeddings2:
                np.testing.assert_array_almost_equal(
                    embeddings1[cust_id],
                    embeddings2[cust_id],
                    err_msg=f"Non-deterministic embedding for {cust_id}"
                )
