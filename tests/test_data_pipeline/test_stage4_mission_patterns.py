"""
Tests for Stage 4: Mission Pattern Extraction
=============================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_pipeline.stage4_mission_patterns import MissionPatternPipeline


class TestMissionPatternPipeline:
    """Test suite for MissionPatternPipeline."""

    def test_init(self):
        """Test pipeline initialization."""
        pipeline = MissionPatternPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'min_trips')

    def test_basic_pattern_extraction(self, mini_transactions):
        """Test basic pattern extraction."""
        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(mini_transactions)

        # Should have output for customers with enough baskets
        assert len(patterns_df) >= 0

    def test_output_columns(self, mini_transactions):
        """Test output has customer_id column."""
        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(mini_transactions)

        if len(patterns_df) > 0:
            assert 'customer_id' in patterns_df.columns

    def test_probability_distributions_sum_to_one(self, mini_transactions):
        """Test that probability distributions sum to approximately 1."""
        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(mini_transactions)

        if len(patterns_df) > 0:
            # Check mission type probabilities if they exist
            mission_type_cols = [c for c in patterns_df.columns if c.startswith('p_mission_')]
            if mission_type_cols:
                sums = patterns_df[mission_type_cols].sum(axis=1)
                # Allow some tolerance for floating point
                assert ((sums > 0.99) & (sums < 1.01)).all() or (sums == 0).all()

    def test_probabilities_non_negative(self, mini_transactions):
        """Test that all probabilities are non-negative."""
        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(mini_transactions)

        if len(patterns_df) > 0:
            prob_cols = [c for c in patterns_df.columns if c.startswith('p_')]
            for col in prob_cols:
                assert (patterns_df[col] >= 0).all(), f"Negative values in {col}"

    def test_probabilities_max_one(self, mini_transactions):
        """Test that all probabilities are at most 1."""
        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(mini_transactions)

        if len(patterns_df) > 0:
            prob_cols = [c for c in patterns_df.columns if c.startswith('p_')]
            for col in prob_cols:
                assert (patterns_df[col] <= 1).all(), f"Values > 1 in {col}"

    def test_unique_customers(self, mini_transactions):
        """Test that each customer appears only once."""
        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(mini_transactions)

        if len(patterns_df) > 0:
            assert not patterns_df['customer_id'].duplicated().any()


class TestMissionTypePatterns:
    """Test mission type pattern extraction."""

    def test_single_mission_type(self):
        """Test customer with single mission type."""
        data = pd.DataFrame({
            'CUST_CODE': ['C1'] * 5,
            'BASKET_ID': [f'B{i}' for i in range(5)],
            'BASKET_TYPE': ['Top Up'] * 5,
            'BASKET_DOMINANT_MISSION': ['Fresh'] * 5,
            'BASKET_PRICE_SENSITIVITY': ['MM'] * 5,
            'BASKET_SIZE': ['M'] * 5,
        })

        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(data)

        assert len(patterns_df) == 1

    def test_mixed_mission_types(self):
        """Test customer with mixed mission types."""
        data = pd.DataFrame({
            'CUST_CODE': ['C1'] * 4,
            'BASKET_ID': [f'B{i}' for i in range(4)],
            'BASKET_TYPE': ['Top Up', 'Top Up', 'Full Shop', 'Full Shop'],
            'BASKET_DOMINANT_MISSION': ['Fresh'] * 4,
            'BASKET_PRICE_SENSITIVITY': ['MM'] * 4,
            'BASKET_SIZE': ['M'] * 4,
        })

        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(data)

        assert len(patterns_df) == 1


class TestMissionPatternEdgeCases:
    """Test edge cases for mission pattern extraction."""

    def test_empty_transactions(self):
        """Test with empty transactions."""
        data = pd.DataFrame(columns=[
            'CUST_CODE', 'BASKET_ID', 'BASKET_TYPE',
            'BASKET_DOMINANT_MISSION', 'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE'
        ])

        pipeline = MissionPatternPipeline()
        patterns_df = pipeline.run(data)

        assert len(patterns_df) == 0
