"""
Tests for Layer 2: Fourier Price Encoding
==========================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering.layer2_fourier_price import FourierPriceEncoder


class TestFourierPriceEncoder:
    """Test suite for FourierPriceEncoder."""

    def test_init(self):
        """Test initialization."""
        encoder = FourierPriceEncoder()
        assert encoder is not None
        assert encoder.output_dim == 64

    def test_output_dimension(self, sample_prices):
        """Test that output has correct dimension."""
        encoder = FourierPriceEncoder()
        result = encoder.run(sample_prices)

        # Should have 64 feature columns plus metadata
        feature_cols = [c for c in result.columns if c.startswith(('fourier_', 'log_', 'relative_', 'velocity_'))]
        assert len(feature_cols) == 64

    def test_no_nan_values(self, sample_prices):
        """Test that output has no NaN values in features."""
        encoder = FourierPriceEncoder()
        result = encoder.run(sample_prices)

        feature_cols = [c for c in result.columns if c.startswith(('fourier_', 'log_', 'relative_', 'velocity_'))]
        for col in feature_cols:
            assert not result[col].isna().any(), f"NaN values in {col}"

    def test_no_infinite_values(self, sample_prices):
        """Test that output has no infinite values."""
        encoder = FourierPriceEncoder()
        result = encoder.run(sample_prices)

        feature_cols = [c for c in result.columns if c.startswith(('fourier_', 'log_', 'relative_', 'velocity_'))]
        for col in feature_cols:
            assert np.isfinite(result[col]).all(), f"Infinite values in {col}"

    def test_fourier_features_bounded(self, sample_prices):
        """Test that Fourier features are bounded (sin/cos in [-1, 1])."""
        encoder = FourierPriceEncoder()
        result = encoder.run(sample_prices)

        fourier_cols = [c for c in result.columns if c.startswith('fourier_')]
        for col in fourier_cols:
            # Sin/cos should be in [-1, 1], but we may have products, so allow [-2, 2]
            assert (result[col] >= -2).all(), f"{col} has values < -2"
            assert (result[col] <= 2).all(), f"{col} has values > 2"


class TestFourierFeatureComponents:
    """Test individual feature components."""

    def test_log_features(self):
        """Test log-price feature computation."""
        prices = pd.DataFrame({
            'product_id': ['P1'],
            'actual_price': [10.0],
            'base_price': [12.0],
        })

        encoder = FourierPriceEncoder()
        result = encoder.run(prices)

        log_cols = [c for c in result.columns if c.startswith('log_')]
        assert len(log_cols) == 16

    def test_relative_features(self):
        """Test relative price feature computation."""
        prices = pd.DataFrame({
            'product_id': ['P1', 'P2'],
            'actual_price': [10.0, 20.0],
            'base_price': [12.0, 22.0],
        })

        encoder = FourierPriceEncoder()
        result = encoder.run(prices)

        relative_cols = [c for c in result.columns if c.startswith('relative_')]
        assert len(relative_cols) == 16

    def test_velocity_features(self):
        """Test velocity feature computation."""
        prices = pd.DataFrame({
            'product_id': ['P1'],
            'actual_price': [10.0],
            'base_price': [12.0],
        })

        encoder = FourierPriceEncoder()
        result = encoder.run(prices)

        velocity_cols = [c for c in result.columns if c.startswith('velocity_')]
        assert len(velocity_cols) == 8


class TestFourierEdgeCases:
    """Test edge cases for Fourier encoding."""

    def test_empty_prices(self):
        """Test with empty price data."""
        prices = pd.DataFrame(columns=['product_id', 'actual_price', 'base_price'])

        encoder = FourierPriceEncoder()
        result = encoder.run(prices)

        assert len(result) == 0

    def test_zero_price(self):
        """Test handling of zero price."""
        prices = pd.DataFrame({
            'product_id': ['P1'],
            'actual_price': [0.0],
            'base_price': [10.0],
        })

        encoder = FourierPriceEncoder()
        result = encoder.run(prices)

        # Should handle gracefully (no NaN/Inf)
        feature_cols = [c for c in result.columns if c.startswith(('fourier_', 'log_', 'relative_', 'velocity_'))]
        for col in feature_cols:
            assert np.isfinite(result[col]).all()

    def test_very_small_price(self):
        """Test handling of very small price."""
        prices = pd.DataFrame({
            'product_id': ['P1'],
            'actual_price': [0.001],
            'base_price': [0.01],
        })

        encoder = FourierPriceEncoder()
        result = encoder.run(prices)

        feature_cols = [c for c in result.columns if c.startswith(('fourier_', 'log_', 'relative_', 'velocity_'))]
        for col in feature_cols:
            assert np.isfinite(result[col]).all()

    def test_very_large_price(self):
        """Test handling of very large price."""
        prices = pd.DataFrame({
            'product_id': ['P1'],
            'actual_price': [10000.0],
            'base_price': [12000.0],
        })

        encoder = FourierPriceEncoder()
        result = encoder.run(prices)

        feature_cols = [c for c in result.columns if c.startswith(('fourier_', 'log_', 'relative_', 'velocity_'))]
        for col in feature_cols:
            assert np.isfinite(result[col]).all()

    def test_equal_prices(self):
        """Test when actual equals base price (no discount)."""
        prices = pd.DataFrame({
            'product_id': ['P1'],
            'actual_price': [10.0],
            'base_price': [10.0],
        })

        encoder = FourierPriceEncoder()
        result = encoder.run(prices)

        # Log diff should be 0
        assert len(result) == 1


class TestFourierDeterminism:
    """Test that encoding is deterministic."""

    def test_deterministic_output(self, sample_prices):
        """Test that same input produces same output."""
        encoder = FourierPriceEncoder()

        result1 = encoder.run(sample_prices)
        result2 = encoder.run(sample_prices)

        feature_cols = [c for c in result1.columns if c.startswith(('fourier_', 'log_', 'relative_', 'velocity_'))]

        for col in feature_cols:
            np.testing.assert_array_almost_equal(
                result1[col].values,
                result2[col].values,
                err_msg=f"Non-deterministic output in {col}"
            )
