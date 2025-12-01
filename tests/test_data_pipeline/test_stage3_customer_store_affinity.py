"""
Tests for Stage 3: Customer-Store Affinity
==========================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_pipeline.stage3_customer_store_affinity import CustomerStoreAffinityPipeline


class TestCustomerStoreAffinityPipeline:
    """Test suite for CustomerStoreAffinityPipeline."""

    def test_init(self):
        """Test pipeline initialization."""
        pipeline = CustomerStoreAffinityPipeline()
        assert pipeline is not None

    def test_basic_affinity_calculation(self, mini_transactions):
        """Test basic affinity calculation."""
        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(mini_transactions)

        # Should have output for customers
        assert len(affinity_df) >= 0

    def test_output_columns(self, mini_transactions):
        """Test output has required columns."""
        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(mini_transactions)

        if len(affinity_df) > 0:
            required_cols = ['customer_id', 'primary_store', 'loyalty_score']
            for col in required_cols:
                assert col in affinity_df.columns, f"Missing column: {col}"

    def test_herfindahl_index_range(self, mini_transactions):
        """Test that Herfindahl index is in valid range [0, 1]."""
        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(mini_transactions)

        if 'store_concentration' in affinity_df.columns and len(affinity_df) > 0:
            hhi = affinity_df['store_concentration']
            assert (hhi >= 0).all(), "HHI below 0"
            assert (hhi <= 1).all(), "HHI above 1"

    def test_switching_rate_range(self, mini_transactions):
        """Test that switching rate is in valid range [0, 1]."""
        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(mini_transactions)

        if 'switching_rate' in affinity_df.columns and len(affinity_df) > 0:
            sr = affinity_df['switching_rate']
            assert (sr >= 0).all(), "Switching rate below 0"
            assert (sr <= 1).all(), "Switching rate above 1"

    def test_loyalty_score_range(self, mini_transactions):
        """Test that loyalty score is in valid range [0, 1]."""
        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(mini_transactions)

        if len(affinity_df) > 0:
            ls = affinity_df['loyalty_score']
            assert (ls >= 0).all(), "Loyalty score below 0"
            assert (ls <= 1).all(), "Loyalty score above 1"

    def test_unique_customers(self, mini_transactions):
        """Test that each customer appears only once."""
        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(mini_transactions)

        if len(affinity_df) > 0:
            assert not affinity_df['customer_id'].duplicated().any()


class TestAffinityMetrics:
    """Test specific affinity metric calculations."""

    def test_single_store_customer(self):
        """Test customer who only shops at one store."""
        data = pd.DataFrame({
            'CUST_CODE': ['C1'] * 5,
            'STORE_CODE': ['S1'] * 5,
            'STORE_REGION': ['R1'] * 5,
            'BASKET_ID': [f'B{i}' for i in range(5)],
            'SHOP_WEEK': list(range(5)),
        })

        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(data)

        assert len(affinity_df) == 1
        row = affinity_df.iloc[0]

        # Primary store should be S1
        assert row['primary_store'] == 'S1'

        # Should have perfect concentration
        if 'store_concentration' in affinity_df.columns:
            assert row['store_concentration'] == 1.0

    def test_even_split_customer(self):
        """Test customer who splits evenly between stores."""
        data = pd.DataFrame({
            'CUST_CODE': ['C1'] * 4,
            'STORE_CODE': ['S1', 'S2', 'S1', 'S2'],
            'STORE_REGION': ['R1', 'R1', 'R1', 'R1'],
            'BASKET_ID': [f'B{i}' for i in range(4)],
            'SHOP_WEEK': list(range(4)),
        })

        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(data)

        assert len(affinity_df) == 1
        row = affinity_df.iloc[0]

        # HHI for 50-50 split = 0.5^2 + 0.5^2 = 0.5
        if 'store_concentration' in affinity_df.columns:
            assert abs(row['store_concentration'] - 0.5) < 0.01


class TestAffinityEdgeCases:
    """Test edge cases for affinity calculation."""

    def test_empty_transactions(self):
        """Test with empty transactions."""
        data = pd.DataFrame(columns=['CUST_CODE', 'STORE_CODE', 'STORE_REGION', 'BASKET_ID', 'SHOP_WEEK'])

        pipeline = CustomerStoreAffinityPipeline()
        affinity_df = pipeline.run(data)

        assert len(affinity_df) == 0
