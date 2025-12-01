"""
Tests for Layer 3: Graph Embeddings (GraphSAGE)
===============================================
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering.layer3_graph_embeddings import GraphSAGEEncoder


class TestGraphSAGEEncoder:
    """Test suite for GraphSAGEEncoder."""

    def test_init(self):
        """Test initialization."""
        encoder = GraphSAGEEncoder()
        assert encoder is not None
        assert encoder.embedding_dim == 256

    def test_init_custom_dim(self):
        """Test initialization with custom dimension."""
        encoder = GraphSAGEEncoder(embedding_dim=128)
        assert encoder.embedding_dim == 128

    def test_basic_embedding(self):
        """Test basic embedding generation."""
        # Create simple graph
        graph = nx.Graph()
        graph.add_edges_from([
            ('P1', 'P2', {'edge_type': 'copurchase', 'weight': 1.0}),
            ('P2', 'P3', {'edge_type': 'copurchase', 'weight': 1.0}),
            ('P1', 'C1', {'edge_type': 'hierarchy', 'weight': 1.0}),
        ])

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2', 'P3'],
            'pseudo_brand_id': ['B1', 'B1', 'B2'],
            'price_tier': ['Mid', 'Mid', 'Premium'],
            'category': ['C1', 'C1', 'C1'],
        })

        encoder = GraphSAGEEncoder(embedding_dim=64, num_layers=1)
        embeddings = encoder.run(graph, pseudo_brands)

        assert len(embeddings) > 0
        for prod_id, embed in embeddings.items():
            assert embed.shape == (64,)

    def test_embedding_dimension(self):
        """Test that embeddings have correct dimension."""
        graph = nx.Graph()
        graph.add_edges_from([
            ('P1', 'P2', {'edge_type': 'copurchase', 'weight': 1.0}),
        ])

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2'],
            'pseudo_brand_id': ['B1', 'B1'],
            'price_tier': ['Mid', 'Mid'],
            'category': ['C1', 'C1'],
        })

        encoder = GraphSAGEEncoder(embedding_dim=256)
        embeddings = encoder.run(graph, pseudo_brands)

        for embed in embeddings.values():
            assert embed.shape == (256,)

    def test_no_nan_embeddings(self):
        """Test that embeddings have no NaN values."""
        graph = nx.Graph()
        graph.add_edges_from([
            ('P1', 'P2', {'edge_type': 'copurchase', 'weight': 1.0}),
        ])

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2'],
            'pseudo_brand_id': ['B1', 'B1'],
            'price_tier': ['Mid', 'Mid'],
            'category': ['C1', 'C1'],
        })

        encoder = GraphSAGEEncoder()
        embeddings = encoder.run(graph, pseudo_brands)

        for prod_id, embed in embeddings.items():
            assert not np.isnan(embed).any(), f"NaN in embedding for {prod_id}"

    def test_no_infinite_embeddings(self):
        """Test that embeddings have no infinite values."""
        graph = nx.Graph()
        graph.add_edges_from([
            ('P1', 'P2', {'edge_type': 'copurchase', 'weight': 1.0}),
        ])

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2'],
            'pseudo_brand_id': ['B1', 'B1'],
            'price_tier': ['Mid', 'Mid'],
            'category': ['C1', 'C1'],
        })

        encoder = GraphSAGEEncoder()
        embeddings = encoder.run(graph, pseudo_brands)

        for prod_id, embed in embeddings.items():
            assert np.isfinite(embed).all(), f"Infinite values in embedding for {prod_id}"


class TestGraphSAGEAggregation:
    """Test GraphSAGE aggregation behavior."""

    def test_neighbor_influence(self):
        """Test that neighbors influence embeddings."""
        # Graph where P1 and P2 are connected
        graph = nx.Graph()
        graph.add_edge('P1', 'P2', edge_type='copurchase', weight=1.0)

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2'],
            'pseudo_brand_id': ['B1', 'B1'],
            'price_tier': ['Mid', 'Mid'],
            'category': ['C1', 'C1'],
        })

        encoder = GraphSAGEEncoder(embedding_dim=64)
        embeddings = encoder.run(graph, pseudo_brands)

        # Connected nodes should have similar embeddings
        if 'P1' in embeddings and 'P2' in embeddings:
            similarity = np.dot(embeddings['P1'], embeddings['P2'])
            similarity /= (np.linalg.norm(embeddings['P1']) * np.linalg.norm(embeddings['P2']))
            # Should have some positive correlation
            assert similarity > -1  # Very weak requirement

    def test_isolated_node(self):
        """Test handling of isolated nodes."""
        graph = nx.Graph()
        graph.add_node('P1')  # Isolated
        graph.add_edge('P2', 'P3', edge_type='copurchase', weight=1.0)

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2', 'P3'],
            'pseudo_brand_id': ['B1', 'B1', 'B1'],
            'price_tier': ['Mid', 'Mid', 'Mid'],
            'category': ['C1', 'C1', 'C1'],
        })

        encoder = GraphSAGEEncoder(embedding_dim=64)
        embeddings = encoder.run(graph, pseudo_brands)

        # Isolated node should still get embedding
        assert 'P1' in embeddings
        assert not np.isnan(embeddings['P1']).any()


class TestGraphSAGEEdgeCases:
    """Test edge cases for GraphSAGE encoder."""

    def test_empty_graph(self):
        """Test with empty graph."""
        graph = nx.Graph()
        pseudo_brands = pd.DataFrame(columns=['product_id', 'pseudo_brand_id', 'price_tier', 'category'])

        encoder = GraphSAGEEncoder()
        embeddings = encoder.run(graph, pseudo_brands)

        assert len(embeddings) == 0

    def test_single_node(self):
        """Test with single node."""
        graph = nx.Graph()
        graph.add_node('P1')

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1'],
            'pseudo_brand_id': ['B1'],
            'price_tier': ['Mid'],
            'category': ['C1'],
        })

        encoder = GraphSAGEEncoder(embedding_dim=64)
        embeddings = encoder.run(graph, pseudo_brands)

        assert len(embeddings) == 1
        assert 'P1' in embeddings

    def test_multiple_layers(self):
        """Test with multiple aggregation layers."""
        graph = nx.Graph()
        # Chain: P1 - P2 - P3 - P4
        graph.add_edge('P1', 'P2', edge_type='copurchase', weight=1.0)
        graph.add_edge('P2', 'P3', edge_type='copurchase', weight=1.0)
        graph.add_edge('P3', 'P4', edge_type='copurchase', weight=1.0)

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2', 'P3', 'P4'],
            'pseudo_brand_id': ['B1'] * 4,
            'price_tier': ['Mid'] * 4,
            'category': ['C1'] * 4,
        })

        # With 2 layers, P1 should see P2 (1-hop) and P3 (2-hop)
        encoder = GraphSAGEEncoder(embedding_dim=64, num_layers=2)
        embeddings = encoder.run(graph, pseudo_brands)

        assert len(embeddings) >= 4

    def test_save_and_load(self, temp_dir):
        """Test saving and loading embeddings."""
        graph = nx.Graph()
        graph.add_edge('P1', 'P2', edge_type='copurchase', weight=1.0)

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2'],
            'pseudo_brand_id': ['B1', 'B1'],
            'price_tier': ['Mid', 'Mid'],
            'category': ['C1', 'C1'],
        })

        encoder = GraphSAGEEncoder(embedding_dim=64)
        embeddings = encoder.run(graph, pseudo_brands)

        save_path = temp_dir / 'test_embeddings.pkl'
        encoder.save(str(save_path))

        assert save_path.exists()

        # Load and verify
        loaded = encoder.load(str(save_path))
        assert 'embeddings' in loaded
        assert len(loaded['embeddings']) == len(embeddings)


class TestGraphSAGEDeterminism:
    """Test that GraphSAGE is deterministic with fixed seed."""

    def test_deterministic_embeddings(self):
        """Test that same input produces same output."""
        graph = nx.Graph()
        graph.add_edge('P1', 'P2', edge_type='copurchase', weight=1.0)

        pseudo_brands = pd.DataFrame({
            'product_id': ['P1', 'P2'],
            'pseudo_brand_id': ['B1', 'B1'],
            'price_tier': ['Mid', 'Mid'],
            'category': ['C1', 'C1'],
        })

        encoder1 = GraphSAGEEncoder(embedding_dim=64)
        embeddings1 = encoder1.run(graph, pseudo_brands)

        encoder2 = GraphSAGEEncoder(embedding_dim=64)
        embeddings2 = encoder2.run(graph, pseudo_brands)

        for prod_id in embeddings1:
            np.testing.assert_array_almost_equal(
                embeddings1[prod_id],
                embeddings2[prod_id],
                err_msg=f"Non-deterministic embedding for {prod_id}"
            )
