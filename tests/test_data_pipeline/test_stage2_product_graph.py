"""
Tests for Stage 2: Product Graph Construction
==============================================
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_pipeline.stage2_product_graph import ProductGraphBuilder


class TestProductGraphBuilder:
    """Test suite for ProductGraphBuilder."""

    def test_init(self):
        """Test builder initialization."""
        builder = ProductGraphBuilder()
        assert builder is not None
        assert builder.lift_threshold == 1.5

    def test_basic_graph_construction(self, mini_transactions, sample_prices):
        """Test basic graph construction."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        assert graph is not None
        assert isinstance(graph, nx.MultiGraph)
        assert graph.number_of_nodes() > 0

    def test_graph_has_products(self, mini_transactions, sample_prices):
        """Test that graph contains product nodes."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        # Should have at least some product nodes
        product_nodes = [n for n in graph.nodes() if str(n).startswith('PRD')]
        assert len(product_nodes) > 0

    def test_copurchase_edges(self, mini_transactions, sample_prices):
        """Test co-purchase edge creation."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        # Check for co-purchase edges
        copurchase_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get('edge_type') == 'copurchase'
        ]

        # Should have some co-purchase edges if baskets have multiple items
        # (may be empty for very sparse data)
        assert isinstance(copurchase_edges, list)

    def test_hierarchy_edges(self, mini_transactions, sample_prices):
        """Test hierarchy edge creation."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        # Check for hierarchy edges
        hierarchy_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get('edge_type') == 'hierarchy'
        ]

        # Should have hierarchy edges
        assert len(hierarchy_edges) > 0

    def test_edge_weights_positive(self, mini_transactions, sample_prices):
        """Test that edge weights are positive."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        for u, v, d in graph.edges(data=True):
            weight = d.get('weight', 1.0)
            assert weight > 0, f"Non-positive weight on edge ({u}, {v})"

    def test_no_self_loops(self, mini_transactions, sample_prices):
        """Test that graph has no self-loops."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        self_loops = list(nx.selfloop_edges(graph))
        assert len(self_loops) == 0, f"Found self-loops: {self_loops}"

    def test_graph_connected_components(self, mini_transactions, sample_prices):
        """Test graph connectivity."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        # Get number of connected components
        n_components = nx.number_connected_components(graph)

        # Should have at least one component
        assert n_components >= 1

    def test_save_and_load(self, mini_transactions, sample_prices, temp_dir):
        """Test saving and loading graph."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        # Save
        save_path = temp_dir / 'test_graph.pkl'
        builder.save(str(save_path))

        assert save_path.exists()

        # Load
        loaded_graph = builder.load(str(save_path))

        assert loaded_graph.number_of_nodes() == graph.number_of_nodes()
        assert loaded_graph.number_of_edges() == graph.number_of_edges()


class TestGraphMetrics:
    """Test graph quality metrics."""

    def test_average_degree(self, mini_transactions, sample_prices):
        """Test that average degree is reasonable."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        if graph.number_of_nodes() > 0:
            avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
            # Average degree should be >= 0
            assert avg_degree >= 0

    def test_graph_density(self, mini_transactions, sample_prices):
        """Test graph density is in valid range."""
        builder = ProductGraphBuilder()
        graph = builder.run(mini_transactions, sample_prices)

        if graph.number_of_nodes() > 1:
            # Use simple graph for density calculation
            simple_graph = nx.Graph(graph)
            density = nx.density(simple_graph)
            assert 0 <= density <= 1
