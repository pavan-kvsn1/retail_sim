"""
Tests for Layer 1: Pseudo-Brand Inference
==========================================
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering.layer1_pseudo_brand import PseudoBrandInference


class TestPseudoBrandInference:
    """Test suite for PseudoBrandInference."""

    def test_init(self):
        """Test initialization."""
        inferencer = PseudoBrandInference()
        assert inferencer is not None

    def test_basic_inference(self, mini_transactions, sample_prices):
        """Test basic pseudo-brand inference."""
        # Create a simple graph
        graph = nx.Graph()
        for prod in mini_transactions['PROD_CODE'].unique()[:20]:
            graph.add_node(prod)

        inferencer = PseudoBrandInference()
        result = inferencer.run(mini_transactions, sample_prices, graph)

        assert len(result) > 0
        assert 'product_id' in result.columns
        assert 'pseudo_brand_id' in result.columns

    def test_unique_products(self, mini_transactions, sample_prices):
        """Test that each product has one pseudo-brand."""
        graph = nx.Graph()
        for prod in mini_transactions['PROD_CODE'].unique()[:20]:
            graph.add_node(prod)

        inferencer = PseudoBrandInference()
        result = inferencer.run(mini_transactions, sample_prices, graph)

        if len(result) > 0:
            assert not result['product_id'].duplicated().any()

    def test_pseudo_brand_id_format(self, mini_transactions, sample_prices):
        """Test pseudo-brand ID format."""
        graph = nx.Graph()
        for prod in mini_transactions['PROD_CODE'].unique()[:20]:
            graph.add_node(prod)

        inferencer = PseudoBrandInference()
        result = inferencer.run(mini_transactions, sample_prices, graph)

        if len(result) > 0:
            # Pseudo-brand IDs should be non-empty strings
            assert result['pseudo_brand_id'].notna().all()


class TestPseudoBrandEdgeCases:
    """Test edge cases for pseudo-brand inference."""

    def test_single_product(self):
        """Test with single product."""
        transactions = pd.DataFrame({
            'PROD_CODE': ['P1'],
            'PROD_CODE_40': ['C1'],
        })

        prices = pd.DataFrame({
            'product_id': ['P1'],
            'actual_price': [10.0],
        })

        graph = nx.Graph()
        graph.add_node('P1')

        inferencer = PseudoBrandInference()
        result = inferencer.run(transactions, prices, graph)

        assert len(result) == 1
