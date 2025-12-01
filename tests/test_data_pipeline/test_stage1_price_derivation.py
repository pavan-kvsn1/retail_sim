"""
Tests for Stage 1: Price Derivation Pipeline
=============================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_pipeline.stage1_price_derivation import PriceDerivationPipeline


class TestPriceDerivationPipeline:
    """Test suite for PriceDerivationPipeline."""

    def test_init(self):
        """Test pipeline initialization."""
        pipeline = PriceDerivationPipeline()
        assert pipeline is not None
        assert pipeline.rolling_window == 4
        assert pipeline.discount_threshold == 0.05

    def test_basic_price_derivation(self, mini_transactions):
        """Test basic price calculation from SPEND/QUANTITY."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        # Should have output
        assert len(prices_df) > 0

        # Check required columns
        required_cols = ['product_id', 'store_id', 'week', 'actual_price', 'base_price']
        for col in required_cols:
            assert col in prices_df.columns, f"Missing column: {col}"

    def test_no_negative_prices(self, mini_transactions):
        """Test that no negative prices are produced."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        assert (prices_df['actual_price'] >= 0).all(), "Found negative actual prices"
        assert (prices_df['base_price'] >= 0).all(), "Found negative base prices"

    def test_no_infinite_prices(self, mini_transactions):
        """Test that no infinite prices are produced."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        assert np.isfinite(prices_df['actual_price']).all(), "Found infinite actual prices"
        assert np.isfinite(prices_df['base_price']).all(), "Found infinite base prices"

    def test_discount_calculation(self, mini_transactions):
        """Test discount percentage calculation."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        if 'discount_depth' in prices_df.columns:
            # Discount depth should be capped at max (70%)
            assert (prices_df['discount_depth'] >= 0).all()
            assert (prices_df['discount_depth'] <= 0.7).all()

    def test_promo_flag(self, mini_transactions):
        """Test promotion flag detection."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        assert 'promo_flag' in prices_df.columns
        assert prices_df['promo_flag'].isin([0, 1]).all()

    def test_price_rank_in_range(self, mini_transactions):
        """Test that price rank is normalized between 0 and 1."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        if 'price_rank' in prices_df.columns:
            valid_ranks = prices_df['price_rank'].dropna()
            if len(valid_ranks) > 0:
                assert (valid_ranks >= 0).all(), "Price rank below 0"
                assert (valid_ranks <= 1).all(), "Price rank above 1"


class TestPriceDerivationOutput:
    """Test output format and structure."""

    def test_output_has_required_columns(self, mini_transactions):
        """Test output contains required columns."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        required = ['product_id', 'store_id', 'week', 'actual_price', 'base_price',
                    'discount_depth', 'promo_flag', 'imputation_method', 'quality_score']
        for col in required:
            assert col in prices_df.columns, f"Missing column: {col}"

    def test_output_types(self, mini_transactions):
        """Test output column types."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        assert pd.api.types.is_numeric_dtype(prices_df['actual_price'])
        assert pd.api.types.is_numeric_dtype(prices_df['base_price'])
        assert pd.api.types.is_numeric_dtype(prices_df['quality_score'])

    def test_quality_score_range(self, mini_transactions):
        """Test quality score is in valid range."""
        pipeline = PriceDerivationPipeline()
        prices_df = pipeline.run(mini_transactions)

        assert (prices_df['quality_score'] >= 0).all()
        assert (prices_df['quality_score'] <= 1).all()
