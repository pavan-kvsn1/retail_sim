"""
Layer 3: Graph Embeddings (GraphSAGE)
======================================
Learns product representations using Graph Sample and Aggregate.

Architecture:
- Layer 0: Node feature initialization (SKU + pseudo-brand + category)
- Layer 1: 1-hop neighborhood aggregation by edge type
- Layer 2: 2-hop neighborhood aggregation

Output: product_embeddings.pkl (PyTorch tensor [N, 256])
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import pickle
import warnings

warnings.filterwarnings('ignore')

# Try importing PyTorch, fall back to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using numpy-only implementation.")


class GraphSAGEEncoder:
    """
    GraphSAGE-style encoder for product graph.

    Learns product embeddings by aggregating features from:
    - Co-purchase neighbors (complementarity)
    - Substitution neighbors (competition)
    - Hierarchy neighbors (taxonomy)

    Output: 256d embedding per product
    """

    def __init__(
        self,
        initial_dim: int = 144,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_samples: int = 10,
        aggregator: str = 'mean'
    ):
        """
        Parameters
        ----------
        initial_dim : int
            Dimension of initial node features
        hidden_dim : int
            Hidden layer dimension
        output_dim : int
            Final embedding dimension
        num_samples : int
            Number of neighbors to sample per edge type
        aggregator : str
            Aggregation method: 'mean', 'max', or 'attention'
        """
        self.initial_dim = initial_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        self.aggregator = aggregator

        # Edge types
        self.edge_types = ['copurchase', 'substitution', 'hierarchy']

        # Storage for embeddings
        self.node_features = {}
        self.embeddings = {}

    def run(
        self,
        product_graph: nx.MultiGraph,
        pseudo_brands: pd.DataFrame,
        use_torch: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate graph embeddings for all products.

        Parameters
        ----------
        product_graph : nx.MultiGraph
            Product graph from Stage 2
        pseudo_brands : pd.DataFrame
            Pseudo-brand features from Layer 1
        use_torch : bool
            Whether to use PyTorch (if available)

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from product_id to 256d embedding
        """
        print("Layer 3: Graph Embeddings (GraphSAGE)")
        print("=" * 50)

        # Step 1: Initialize node features
        print("\nStep 1: Initializing node features...")
        self._initialize_node_features(product_graph, pseudo_brands)
        print(f"  - Initialized {len(self.node_features):,} nodes")

        # Step 2: Build adjacency lists by edge type
        print("\nStep 2: Building adjacency structure...")
        adj_lists = self._build_adjacency_lists(product_graph)
        for edge_type, adj in adj_lists.items():
            avg_neighbors = np.mean([len(v) for v in adj.values()]) if adj else 0
            print(f"  - {edge_type}: {len(adj):,} nodes, avg {avg_neighbors:.1f} neighbors")

        # Step 3: Run GraphSAGE layers
        print("\nStep 3: Running GraphSAGE aggregation...")
        if use_torch and TORCH_AVAILABLE:
            embeddings = self._run_graphsage_torch(adj_lists)
        else:
            embeddings = self._run_graphsage_numpy(adj_lists)

        print(f"  - Generated embeddings for {len(embeddings):,} products")

        # Store embeddings
        self.embeddings = embeddings

        print("\n" + "=" * 50)
        print("Graph Embeddings Complete!")
        print(f"  - Products with embeddings: {len(embeddings):,}")
        print(f"  - Embedding dimension: {self.output_dim}d")

        return embeddings

    def _initialize_node_features(
        self,
        product_graph: nx.MultiGraph,
        pseudo_brands: pd.DataFrame
    ) -> None:
        """
        Initialize node features from pseudo-brands and graph attributes.

        Initial features: [144d]
        - SKU embedding: [64d]
        - Pseudo-brand embedding: [32d]
        - Category embedding: [32d]
        - Price tier embedding: [16d]
        """
        # Create mappings from pseudo_brands
        product_to_brand = dict(zip(
            pseudo_brands['product_id'],
            pseudo_brands['pseudo_brand_idx']
        ))
        product_to_category = dict(zip(
            pseudo_brands['product_id'],
            pseudo_brands['sub_commodity']
        ))
        product_to_tier = dict(zip(
            pseudo_brands['product_id'],
            pseudo_brands['price_tier']
        ))
        product_to_price = dict(zip(
            pseudo_brands['product_id'],
            pseudo_brands['category_price_percentile']
        ))

        # Create vocabulary mappings
        unique_brands = pseudo_brands['pseudo_brand_idx'].unique()
        unique_categories = pseudo_brands['sub_commodity'].unique()
        tier_map = {'value': 0, 'mid': 1, 'premium': 2}

        # Random embeddings (in practice, these would be learned)
        np.random.seed(42)
        brand_embeds = {b: np.random.randn(32) * 0.1 for b in unique_brands}
        category_embeds = {c: np.random.randn(32) * 0.1 for c in unique_categories}
        tier_embeds = {t: np.random.randn(16) * 0.1 for t in range(3)}

        # Initialize features for each product node
        for node in product_graph.nodes():
            node_data = product_graph.nodes[node]

            # Only process product nodes (not category nodes)
            if node_data.get('node_type') != 'product':
                continue

            # Get attributes
            brand_idx = product_to_brand.get(node, 0)
            category = product_to_category.get(node, unique_categories[0])
            tier = product_to_tier.get(node, 'mid')
            price_pct = product_to_price.get(node, 0.5)

            # Create SKU embedding (hash-based for consistency)
            sku_hash = hash(node) % (2**31)
            np.random.seed(sku_hash)
            sku_embed = np.random.randn(64) * 0.1

            # Get other embeddings
            brand_embed = brand_embeds.get(brand_idx, np.zeros(32))
            cat_embed = category_embeds.get(category, np.zeros(32))
            tier_embed = tier_embeds.get(tier_map.get(tier, 1), np.zeros(16))

            # Concatenate all features
            features = np.concatenate([
                sku_embed,      # [64d]
                brand_embed,    # [32d]
                cat_embed,      # [32d]
                tier_embed      # [16d]
            ])

            self.node_features[node] = features

        # Handle category nodes with simple features
        for node in product_graph.nodes():
            if node not in self.node_features:
                # Category node - use zero features
                self.node_features[node] = np.zeros(self.initial_dim)

    def _build_adjacency_lists(
        self,
        product_graph: nx.MultiGraph
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """
        Build adjacency lists separated by edge type.

        Returns
        -------
        Dict[str, Dict[str, List[Tuple[str, float]]]]
            adj_lists[edge_type][node] = [(neighbor, weight), ...]
        """
        adj_lists = {edge_type: defaultdict(list) for edge_type in self.edge_types}

        for u, v, data in product_graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            weight = data.get('weight', 1.0)

            if edge_type in adj_lists:
                adj_lists[edge_type][u].append((v, weight))
                adj_lists[edge_type][v].append((u, weight))

        return adj_lists

    def _run_graphsage_numpy(
        self,
        adj_lists: Dict[str, Dict[str, List[Tuple[str, float]]]]
    ) -> Dict[str, np.ndarray]:
        """
        Run GraphSAGE using numpy (no PyTorch required).
        """
        # Get all product nodes
        product_nodes = [n for n in self.node_features.keys()
                        if self.node_features[n].sum() != 0]

        # Layer 1: Aggregate 1-hop neighbors
        print("  - Layer 1: 1-hop aggregation...")
        layer1_embeds = {}

        for node in product_nodes:
            # Aggregate from each edge type
            aggregated = []

            for edge_type in self.edge_types:
                neighbors = adj_lists[edge_type].get(node, [])

                if neighbors:
                    # Sample neighbors if too many
                    if len(neighbors) > self.num_samples:
                        sampled = np.random.choice(len(neighbors), self.num_samples, replace=False)
                        neighbors = [neighbors[i] for i in sampled]

                    # Weighted mean aggregation
                    neighbor_feats = []
                    weights = []
                    for neighbor, weight in neighbors:
                        if neighbor in self.node_features:
                            neighbor_feats.append(self.node_features[neighbor])
                            weights.append(weight)

                    if neighbor_feats:
                        weights = np.array(weights)
                        weights = weights / weights.sum()
                        agg_feat = np.average(neighbor_feats, axis=0, weights=weights)
                    else:
                        agg_feat = np.zeros(self.initial_dim)
                else:
                    agg_feat = np.zeros(self.initial_dim)

                aggregated.append(agg_feat)

            # Concatenate own features with aggregated neighbors
            own_feat = self.node_features[node]
            combined = np.concatenate([own_feat] + aggregated)  # [144 + 144*3 = 576d]

            # Linear projection + ReLU
            np.random.seed(hash(node) % (2**31) + 1)
            W1 = np.random.randn(combined.shape[0], self.hidden_dim) * 0.1
            h1 = np.maximum(0, combined @ W1)  # ReLU

            layer1_embeds[node] = h1

        # Layer 2: Aggregate from layer 1 embeddings
        print("  - Layer 2: 2-hop aggregation...")
        final_embeds = {}

        for node in product_nodes:
            # Aggregate from layer 1 embeddings
            aggregated = []

            for edge_type in self.edge_types:
                neighbors = adj_lists[edge_type].get(node, [])

                if neighbors:
                    if len(neighbors) > self.num_samples:
                        sampled = np.random.choice(len(neighbors), self.num_samples, replace=False)
                        neighbors = [neighbors[i] for i in sampled]

                    neighbor_feats = []
                    weights = []
                    for neighbor, weight in neighbors:
                        if neighbor in layer1_embeds:
                            neighbor_feats.append(layer1_embeds[neighbor])
                            weights.append(weight)

                    if neighbor_feats:
                        weights = np.array(weights)
                        weights = weights / weights.sum()
                        agg_feat = np.average(neighbor_feats, axis=0, weights=weights)
                    else:
                        agg_feat = np.zeros(self.hidden_dim)
                else:
                    agg_feat = np.zeros(self.hidden_dim)

                aggregated.append(agg_feat)

            # Concatenate
            own_feat = layer1_embeds[node]
            combined = np.concatenate([own_feat] + aggregated)  # [256 + 256*3 = 1024d]

            # Final projection
            np.random.seed(hash(node) % (2**31) + 2)
            W2 = np.random.randn(combined.shape[0], self.output_dim) * 0.1
            h2 = np.maximum(0, combined @ W2)

            # L2 normalize
            norm = np.linalg.norm(h2)
            if norm > 0:
                h2 = h2 / norm

            final_embeds[node] = h2

        return final_embeds

    def _run_graphsage_torch(
        self,
        adj_lists: Dict[str, Dict[str, List[Tuple[str, float]]]]
    ) -> Dict[str, np.ndarray]:
        """
        Run GraphSAGE using PyTorch for better performance.
        """
        if not TORCH_AVAILABLE:
            return self._run_graphsage_numpy(adj_lists)

        # Get all product nodes
        product_nodes = [n for n in self.node_features.keys()
                        if self.node_features[n].sum() != 0]

        # Convert features to tensors
        node_to_idx = {n: i for i, n in enumerate(product_nodes)}
        features = torch.tensor(
            np.array([self.node_features[n] for n in product_nodes]),
            dtype=torch.float32
        )

        # Initialize weight matrices
        torch.manual_seed(42)
        W1 = torch.randn(self.initial_dim * 4, self.hidden_dim) * 0.1
        W2 = torch.randn(self.hidden_dim * 4, self.output_dim) * 0.1

        # Layer 1
        print("  - Layer 1: 1-hop aggregation (PyTorch)...")
        layer1 = torch.zeros(len(product_nodes), self.hidden_dim)

        for i, node in enumerate(product_nodes):
            own_feat = features[i]
            aggregated = []

            for edge_type in self.edge_types:
                neighbors = adj_lists[edge_type].get(node, [])

                if neighbors:
                    if len(neighbors) > self.num_samples:
                        sampled = np.random.choice(len(neighbors), self.num_samples, replace=False)
                        neighbors = [neighbors[j] for j in sampled]

                    neighbor_feats = []
                    weights = []
                    for neighbor, weight in neighbors:
                        if neighbor in node_to_idx:
                            neighbor_feats.append(features[node_to_idx[neighbor]])
                            weights.append(weight)

                    if neighbor_feats:
                        weights = torch.tensor(weights, dtype=torch.float32)
                        weights = weights / weights.sum()
                        stacked = torch.stack(neighbor_feats)
                        agg_feat = (stacked * weights.unsqueeze(1)).sum(0)
                    else:
                        agg_feat = torch.zeros(self.initial_dim)
                else:
                    agg_feat = torch.zeros(self.initial_dim)

                aggregated.append(agg_feat)

            combined = torch.cat([own_feat] + aggregated)
            layer1[i] = F.relu(combined @ W1)

        # Layer 2
        print("  - Layer 2: 2-hop aggregation (PyTorch)...")
        layer2 = torch.zeros(len(product_nodes), self.output_dim)

        for i, node in enumerate(product_nodes):
            own_feat = layer1[i]
            aggregated = []

            for edge_type in self.edge_types:
                neighbors = adj_lists[edge_type].get(node, [])

                if neighbors:
                    if len(neighbors) > self.num_samples:
                        sampled = np.random.choice(len(neighbors), self.num_samples, replace=False)
                        neighbors = [neighbors[j] for j in sampled]

                    neighbor_feats = []
                    weights = []
                    for neighbor, weight in neighbors:
                        if neighbor in node_to_idx:
                            neighbor_feats.append(layer1[node_to_idx[neighbor]])
                            weights.append(weight)

                    if neighbor_feats:
                        weights = torch.tensor(weights, dtype=torch.float32)
                        weights = weights / weights.sum()
                        stacked = torch.stack(neighbor_feats)
                        agg_feat = (stacked * weights.unsqueeze(1)).sum(0)
                    else:
                        agg_feat = torch.zeros(self.hidden_dim)
                else:
                    agg_feat = torch.zeros(self.hidden_dim)

                aggregated.append(agg_feat)

            combined = torch.cat([own_feat] + aggregated)
            h2 = F.relu(combined @ W2)
            layer2[i] = F.normalize(h2, dim=0)

        # Convert back to dict
        return {node: layer2[i].numpy() for i, node in enumerate(product_nodes)}

    def get_embedding(self, product_id: str) -> Optional[np.ndarray]:
        """Get embedding for a single product."""
        return self.embeddings.get(product_id)

    def save(self, filepath: str) -> None:
        """Save embeddings to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'output_dim': self.output_dim
            }, f)
        print(f"Embeddings saved to: {filepath}")

    @staticmethod
    def load(filepath: str) -> Dict[str, np.ndarray]:
        """Load embeddings from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['embeddings']


def main():
    """Run graph embedding generation on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    graph_path = project_root / 'data' / 'processed' / 'product_graph.pkl'
    brands_path = project_root / 'data' / 'features' / 'pseudo_brands.parquet'
    output_path = project_root / 'data' / 'features' / 'product_embeddings.pkl'

    # Load data
    print("Loading data...")
    with open(graph_path, 'rb') as f:
        product_graph = pickle.load(f)
    print(f"  - Graph: {product_graph.number_of_nodes()} nodes, {product_graph.number_of_edges()} edges")

    pseudo_brands = pd.read_parquet(brands_path)
    print(f"  - Pseudo-brands: {len(pseudo_brands)} products")

    # Run encoder
    encoder = GraphSAGEEncoder()
    embeddings = encoder.run(product_graph, pseudo_brands, use_torch=TORCH_AVAILABLE)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(str(output_path))

    # Display sample
    print("\nSample embeddings:")
    sample_products = list(embeddings.keys())[:5]
    for prod in sample_products:
        emb = embeddings[prod]
        print(f"  {prod}: dim={len(emb)}, norm={np.linalg.norm(emb):.3f}, "
              f"mean={emb.mean():.3f}, std={emb.std():.3f}")

    return embeddings


if __name__ == '__main__':
    main()
