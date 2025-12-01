"""
Tests for Layer 5: Store Context Features
=========================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering.layer5_store_context import StoreContextEncoder


class TestStoreContextEncoder:
    """Test suite for StoreContextEncoder."""

    def test_init(self):
        """Test initialization."""
        encoder = StoreContextEncoder()
        assert encoder is not None
        assert encoder.output_dim == 96

    def test_basic_encoding(self, mini_transactions):
        """Test basic store encoding."""
        affinity = pd.DataFrame({
            'customer_id': mini_transactions['CUST_CODE'].unique()[:10],
            'primary_store': mini_transactions['STORE_CODE'].iloc[:10],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(mini_transactions, affinity)

        assert len(result) > 0
        assert 'store_id' in result.columns

    def test_output_dimension(self, mini_transactions):
        """Test that output has correct feature dimension."""
        affinity = pd.DataFrame({
            'customer_id': mini_transactions['CUST_CODE'].unique()[:10],
            'primary_store': mini_transactions['STORE_CODE'].iloc[:10],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(mini_transactions, affinity)

        # Count feature columns
        feature_cols = [c for c in result.columns if c.startswith(('identity_', 'format_', 'region_', 'operational_'))]
        assert len(feature_cols) == 96

    def test_no_nan_features(self, mini_transactions):
        """Test that output has no NaN values."""
        affinity = pd.DataFrame({
            'customer_id': mini_transactions['CUST_CODE'].unique()[:10],
            'primary_store': mini_transactions['STORE_CODE'].iloc[:10],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(mini_transactions, affinity)

        feature_cols = [c for c in result.columns if c.startswith(('identity_', 'format_', 'region_', 'operational_'))]
        for col in feature_cols:
            assert not result[col].isna().any(), f"NaN values in {col}"

    def test_unique_stores(self, mini_transactions):
        """Test that each store appears only once."""
        affinity = pd.DataFrame({
            'customer_id': mini_transactions['CUST_CODE'].unique()[:10],
            'primary_store': mini_transactions['STORE_CODE'].iloc[:10],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(mini_transactions, affinity)

        assert not result['store_id'].duplicated().any()


class TestStoreFeatureComponents:
    """Test individual feature components."""

    def test_identity_features(self, mini_transactions):
        """Test store identity feature generation."""
        affinity = pd.DataFrame({
            'customer_id': ['C1'],
            'primary_store': ['S1'],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(mini_transactions, affinity)

        identity_cols = [c for c in result.columns if c.startswith('identity_')]
        assert len(identity_cols) == 32

    def test_format_features(self, mini_transactions):
        """Test store format feature generation."""
        affinity = pd.DataFrame({
            'customer_id': ['C1'],
            'primary_store': ['S1'],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(mini_transactions, affinity)

        format_cols = [c for c in result.columns if c.startswith('format_')]
        assert len(format_cols) == 24

    def test_region_features(self, mini_transactions):
        """Test store region feature generation."""
        affinity = pd.DataFrame({
            'customer_id': ['C1'],
            'primary_store': ['S1'],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(mini_transactions, affinity)

        region_cols = [c for c in result.columns if c.startswith('region_')]
        assert len(region_cols) == 24

    def test_operational_features(self, mini_transactions):
        """Test operational feature generation."""
        affinity = pd.DataFrame({
            'customer_id': ['C1'],
            'primary_store': ['S1'],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(mini_transactions, affinity)

        operational_cols = [c for c in result.columns if c.startswith('operational_')]
        assert len(operational_cols) == 16


class TestStoreContextEdgeCases:
    """Test edge cases for store context encoding."""

    def test_empty_transactions(self):
        """Test with empty transactions."""
        transactions = pd.DataFrame(columns=['STORE_CODE', 'STORE_FORMAT', 'STORE_REGION'])
        affinity = pd.DataFrame(columns=['customer_id', 'primary_store'])

        encoder = StoreContextEncoder()
        result = encoder.run(transactions, affinity)

        assert len(result) == 0

    def test_single_store(self):
        """Test with single store."""
        transactions = pd.DataFrame({
            'STORE_CODE': ['S1'] * 5,
            'STORE_FORMAT': ['LS'] * 5,
            'STORE_REGION': ['E01'] * 5,
            'BASKET_ID': [f'B{i}' for i in range(5)],
            'CUST_CODE': ['C1'] * 5,
        })

        affinity = pd.DataFrame({
            'customer_id': ['C1'],
            'primary_store': ['S1'],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(transactions, affinity)

        assert len(result) == 1

    def test_different_formats(self):
        """Test encoding different store formats."""
        transactions = pd.DataFrame({
            'STORE_CODE': ['S1', 'S2', 'S3'],
            'STORE_FORMAT': ['LS', 'MS', 'SS'],
            'STORE_REGION': ['E01', 'E01', 'E01'],
            'BASKET_ID': ['B1', 'B2', 'B3'],
            'CUST_CODE': ['C1', 'C2', 'C3'],
        })

        affinity = pd.DataFrame({
            'customer_id': ['C1', 'C2', 'C3'],
            'primary_store': ['S1', 'S2', 'S3'],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(transactions, affinity)

        assert len(result) == 3

        # Different formats should have different embeddings
        if len(result) >= 2:
            format_cols = [c for c in result.columns if c.startswith('format_')]
            embed1 = result.iloc[0][format_cols].values
            embed2 = result.iloc[1][format_cols].values
            # They should be different (different formats)
            assert not np.allclose(embed1, embed2)

    def test_different_regions(self):
        """Test encoding different store regions."""
        transactions = pd.DataFrame({
            'STORE_CODE': ['S1', 'S2'],
            'STORE_FORMAT': ['LS', 'LS'],
            'STORE_REGION': ['E01', 'W01'],
            'BASKET_ID': ['B1', 'B2'],
            'CUST_CODE': ['C1', 'C2'],
        })

        affinity = pd.DataFrame({
            'customer_id': ['C1', 'C2'],
            'primary_store': ['S1', 'S2'],
        })

        encoder = StoreContextEncoder()
        result = encoder.run(transactions, affinity)

        # Different regions should have different region embeddings
        if len(result) >= 2:
            region_cols = [c for c in result.columns if c.startswith('region_')]
            embed1 = result.iloc[0][region_cols].values
            embed2 = result.iloc[1][region_cols].values
            assert not np.allclose(embed1, embed2)


class TestStoreIdentityConsistency:
    """Test that store identity is consistent."""

    def test_same_store_same_identity(self):
        """Test that same store always gets same identity embedding."""
        transactions1 = pd.DataFrame({
            'STORE_CODE': ['S1'],
            'STORE_FORMAT': ['LS'],
            'STORE_REGION': ['E01'],
            'BASKET_ID': ['B1'],
            'CUST_CODE': ['C1'],
        })

        transactions2 = pd.DataFrame({
            'STORE_CODE': ['S1'],
            'STORE_FORMAT': ['LS'],
            'STORE_REGION': ['E01'],
            'BASKET_ID': ['B2'],
            'CUST_CODE': ['C2'],
        })

        affinity1 = pd.DataFrame({'customer_id': ['C1'], 'primary_store': ['S1']})
        affinity2 = pd.DataFrame({'customer_id': ['C2'], 'primary_store': ['S1']})

        encoder = StoreContextEncoder()

        result1 = encoder.run(transactions1, affinity1)
        result2 = encoder.run(transactions2, affinity2)

        # Same store should have same identity embedding
        identity_cols = [c for c in result1.columns if c.startswith('identity_')]
        embed1 = result1.iloc[0][identity_cols].values
        embed2 = result2.iloc[0][identity_cols].values

        np.testing.assert_array_almost_equal(embed1, embed2)
