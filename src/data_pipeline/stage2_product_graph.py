"""
Stage 2: Product Graph Construction
====================================
Builds a heterogeneous product graph with 3 edge types:
1. Co-purchase edges (complementarity via Lift)
2. Substitution edges (competition via Jaccard + low Lift)
3. Hierarchy edges (taxonomy relationships)

Output: product_graph.pkl (NetworkX MultiGraph)
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional
import pickle
import warnings

warnings.filterwarnings('ignore')


class ProductGraphBuilder:
    """
    Constructs heterogeneous product graph with co-purchase,
    substitution, and hierarchy edges.

    Graph Properties:
    - Nodes: Products + Category nodes
    - Edges: 3 types (copurchase, substitution, hierarchy)
    - Weighted: Edge weights capture relationship strength
    """

    def __init__(self, lift_threshold: float = 1.5, min_copurchase_count: int = 50, top_k_complements: int = 15,
                 jaccard_threshold: float = 0.6, max_lift_for_substitution: float = 1.2,
        price_gap_threshold: float = 0.30
    ):
        """
        Parameters
        ----------
        lift_threshold : float
            Minimum lift score for co-purchase edges (default 1.5)
        min_copurchase_count : int
            Minimum basket co-occurrences for statistical significance
        top_k_complements : int
            Keep only top K strongest complements per product
        jaccard_threshold : float
            Minimum customer overlap for substitution (default 0.6)
        max_lift_for_substitution : float
            Maximum lift (mutual exclusivity) for substitution edges
        price_gap_threshold : float
            Maximum price difference ratio for substitutes (30%)
        """
        self.lift_threshold = lift_threshold
        self.min_copurchase_count = min_copurchase_count
        self.top_k_complements = top_k_complements
        self.jaccard_threshold = jaccard_threshold
        self.max_lift_for_substitution = max_lift_for_substitution
        self.price_gap_threshold = price_gap_threshold

        self.graph = nx.MultiGraph()

    def run(self, transactions_df: pd.DataFrame, prices_df: Optional[pd.DataFrame] = None) -> nx.MultiGraph:
        """
        Build complete product graph.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions with basket information
        prices_df : pd.DataFrame, optional
            Derived prices for price-based filtering

        Returns
        -------
        nx.MultiGraph
            Heterogeneous product graph
        """
        print("Stage 2: Product Graph Construction")
        print("=" * 50)

        # Initialize graph with product nodes
        print("\nStep 1: Creating product nodes...")
        self._create_product_nodes(transactions_df)

        # Build co-purchase edges
        print("\nStep 2: Building co-purchase edges (complementarity)...")
        copurchase_edges = self._build_copurchase_edges(transactions_df)
        print(f"  - Created {copurchase_edges:,} co-purchase edges")

        # Build substitution edges
        print("\nStep 3: Building substitution edges (competition)...")
        substitution_edges = self._build_substitution_edges(transactions_df, prices_df)
        print(f"  - Created {substitution_edges:,} substitution edges")

        # Build hierarchy edges
        print("\nStep 4: Building hierarchy edges (taxonomy)...")
        hierarchy_edges = self._build_hierarchy_edges(transactions_df)
        print(f"  - Created {hierarchy_edges:,} hierarchy edges")

        # Summary
        print("\n" + "=" * 50)
        print("Product Graph Complete!")
        print(f"  - Total nodes: {self.graph.number_of_nodes():,}")
        print(f"  - Total edges: {self.graph.number_of_edges():,}")
        print(f"  - Average degree: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.1f}")

        return self.graph

    def _create_product_nodes(self, df: pd.DataFrame) -> None:
        """Create nodes for each unique product with attributes."""
        # Get unique products with their hierarchy
        products = df.groupby('PROD_CODE').agg({
            'PROD_CODE_10': 'first',  # Sub-commodity
            'PROD_CODE_20': 'first',  # Commodity (DEP)
            'PROD_CODE_30': 'first',  # Sub-department (G)
            'PROD_CODE_40': 'first',  # Department (D)
            'QUANTITY': 'sum',
            'SPEND': 'sum'
        }).reset_index()

        # Add product nodes
        for _, row in products.iterrows():
            self.graph.add_node(
                row['PROD_CODE'],
                node_type='product',
                hierarchy_level=0,
                sub_commodity=row['PROD_CODE_10'],
                commodity=row['PROD_CODE_20'],
                sub_department=row['PROD_CODE_30'],
                department=row['PROD_CODE_40'],
                total_quantity=row['QUANTITY'],
                total_spend=row['SPEND'],
                category_path=f"{row['PROD_CODE_40']}/{row['PROD_CODE_30']}/{row['PROD_CODE_20']}/{row['PROD_CODE_10']}"
            )

        print(f"  - Added {len(products):,} product nodes")

    def _build_copurchase_edges(self, df: pd.DataFrame) -> int:
        """
        Build co-purchase edges using Lift score.

        Lift(A,B) = P(A ∩ B) / [P(A) × P(B)]
        - Lift > 1.5 indicates meaningful complementarity
        """
        # Group transactions by basket
        baskets = df.groupby('BASKET_ID')['PROD_CODE'].apply(set).reset_index()
        baskets.columns = ['basket_id', 'products']

        total_baskets = len(baskets)
        print(f"  - Total baskets: {total_baskets:,}")

        # Count single product occurrences
        product_counts = defaultdict(int)
        for products in baskets['products']:
            for p in products:
                product_counts[p] += 1

        # Count pair co-occurrences
        pair_counts = defaultdict(int)
        for products in baskets['products']:
            products_list = list(products)
            if len(products_list) < 2:
                continue
            # Limit to avoid explosion on large baskets
            products_list = products_list[:50]
            for p1, p2 in combinations(sorted(products_list), 2):
                pair_counts[(p1, p2)] += 1

        print(f"  - Computing lift for {len(pair_counts):,} product pairs...")

        # Compute lift and create edges
        edges_added = 0
        lift_scores = []

        for (p1, p2), count in pair_counts.items():
            if count < self.min_copurchase_count:
                continue

            # Calculate lift
            p_a = product_counts[p1] / total_baskets
            p_b = product_counts[p2] / total_baskets
            p_ab = count / total_baskets

            if p_a * p_b > 0:
                lift = p_ab / (p_a * p_b)
            else:
                continue

            lift_scores.append((p1, p2, lift, count))

        # Filter by lift threshold and add edges
        for p1, p2, lift, count in lift_scores:
            if lift >= self.lift_threshold:
                self.graph.add_edge(
                    p1, p2,
                    edge_type='copurchase',
                    weight=lift,
                    count=count,
                    lift=lift
                )
                edges_added += 1

        # Keep only top K complements per product
        if self.top_k_complements:
            edges_added = self._prune_to_top_k(edges_added, 'copurchase')

        # Store lift scores for substitution analysis
        self._lift_cache = {(p1, p2): lift for p1, p2, lift, _ in lift_scores}

        return edges_added

    def _build_substitution_edges(
        self,
        df: pd.DataFrame,
        prices_df: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Build substitution edges using heuristic approach:
        1. High customer overlap (Jaccard > 0.6)
        2. Low co-purchase rate (Lift < 1.2)
        3. Same category (sub-commodity)
        4. Similar price points (gap < 30%)
        """
        # Group customers by products they've purchased
        customer_products = df.groupby('CUST_CODE')['PROD_CODE'].apply(set).to_dict()

        # Build product -> customer set mapping
        product_customers = defaultdict(set)
        for cust, products in customer_products.items():
            if pd.isna(cust):
                continue
            for p in products:
                product_customers[p].add(cust)

        # Get product categories and prices
        product_info = df.groupby('PROD_CODE').agg({
            'PROD_CODE_10': 'first',  # Sub-commodity
            'SPEND': 'sum',
            'QUANTITY': 'sum'
        }).reset_index()

        product_category = dict(zip(product_info['PROD_CODE'], product_info['PROD_CODE_10']))

        # Compute average price per product
        product_info['avg_price'] = product_info['SPEND'] / product_info['QUANTITY'].clip(1)
        product_price = dict(zip(product_info['PROD_CODE'], product_info['avg_price']))

        # Group products by category
        category_products = defaultdict(list)
        for p, cat in product_category.items():
            category_products[cat].append(p)

        print(f"  - Analyzing substitution within {len(category_products):,} categories...")

        edges_added = 0

        # Within each category, find substitutes
        for category, products in category_products.items():
            if len(products) < 2:
                continue

            # Limit comparisons within large categories
            products = products[:100]

            for p1, p2 in combinations(products, 2):
                # Check if both have customer data
                customers_1 = product_customers.get(p1, set())
                customers_2 = product_customers.get(p2, set())

                if len(customers_1) < 5 or len(customers_2) < 5:
                    continue

                # Compute Jaccard similarity
                intersection = len(customers_1 & customers_2)
                union = len(customers_1 | customers_2)
                jaccard = intersection / union if union > 0 else 0

                if jaccard < self.jaccard_threshold:
                    continue

                # Check lift (should be low for substitutes)
                pair_key = tuple(sorted([p1, p2]))
                lift = self._lift_cache.get(pair_key, 0.5)  # Default low lift if not in cache

                if lift > self.max_lift_for_substitution:
                    continue

                # Check price similarity
                price_1 = product_price.get(p1, 0)
                price_2 = product_price.get(p2, 0)

                if price_1 > 0 and price_2 > 0:
                    price_gap = abs(price_1 - price_2) / ((price_1 + price_2) / 2)
                    if price_gap > self.price_gap_threshold:
                        continue
                else:
                    continue

                # Compute substitution weight: Higher Jaccard × Lower Lift = Stronger substitute
                weight = jaccard * (1 - min(lift, 1.0))

                # Add substitution edge
                self.graph.add_edge(
                    p1, p2,
                    edge_type='substitution',
                    weight=weight,
                    jaccard=jaccard,
                    lift=lift,
                    price_gap=price_gap
                )
                edges_added += 1

        return edges_added

    def _build_hierarchy_edges(self, df: pd.DataFrame) -> int:
        """
        Build hierarchy edges from product taxonomy.

        4-Level Taxonomy:
        D (Department) <- G (Sub-department) <- DEP (Commodity) <- CL (Sub-commodity) <- PRD (Product)

        Edge weights: Inverse of hierarchy level distance
        """
        # Get unique hierarchy mappings
        hierarchy = df.groupby(['PROD_CODE']).agg({
            'PROD_CODE_10': 'first',  # Sub-commodity (CL)
            'PROD_CODE_20': 'first',  # Commodity (DEP)
            'PROD_CODE_30': 'first',  # Sub-department (G)
            'PROD_CODE_40': 'first',  # Department (D)
        }).reset_index()

        edges_added = 0

        # Create category nodes if they don't exist
        for level, col in [
            (1, 'PROD_CODE_10'),
            (2, 'PROD_CODE_20'),
            (3, 'PROD_CODE_30'),
            (4, 'PROD_CODE_40')
        ]:
            unique_cats = hierarchy[col].dropna().unique()
            for cat in unique_cats:
                if cat not in self.graph:
                    self.graph.add_node(
                        cat,
                        node_type='category',
                        hierarchy_level=level
                    )

        # Add edges from products to hierarchy
        for _, row in hierarchy.iterrows():
            product = row['PROD_CODE']
            sub_commodity = row['PROD_CODE_10']
            commodity = row['PROD_CODE_20']
            sub_dept = row['PROD_CODE_30']
            department = row['PROD_CODE_40']

            # PRD -> CL (weight = 1.0)
            if pd.notna(sub_commodity):
                self.graph.add_edge(
                    product, sub_commodity,
                    edge_type='hierarchy',
                    weight=1.0,
                    hierarchy_distance=1
                )
                edges_added += 1

            # CL -> DEP (weight = 0.5)
            if pd.notna(sub_commodity) and pd.notna(commodity):
                if not self.graph.has_edge(sub_commodity, commodity):
                    self.graph.add_edge(
                        sub_commodity, commodity,
                        edge_type='hierarchy',
                        weight=0.5,
                        hierarchy_distance=2
                    )
                    edges_added += 1

            # DEP -> G (weight = 0.25)
            if pd.notna(commodity) and pd.notna(sub_dept):
                if not self.graph.has_edge(commodity, sub_dept):
                    self.graph.add_edge(
                        commodity, sub_dept,
                        edge_type='hierarchy',
                        weight=0.25,
                        hierarchy_distance=3
                    )
                    edges_added += 1

            # G -> D (weight = 0.125)
            if pd.notna(sub_dept) and pd.notna(department):
                if not self.graph.has_edge(sub_dept, department):
                    self.graph.add_edge(
                        sub_dept, department,
                        edge_type='hierarchy',
                        weight=0.125,
                        hierarchy_distance=4
                    )
                    edges_added += 1

        return edges_added

    def _prune_to_top_k(self, current_count: int, edge_type: str) -> int:
        """Keep only top K edges per node for a given edge type."""
        # Get all edges of this type
        edges_to_remove = []
        node_edges = defaultdict(list)

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('edge_type') == edge_type:
                weight = data.get('weight', 0)
                node_edges[u].append((v, key, weight))
                node_edges[v].append((u, key, weight))

        # For each node, keep only top K
        for node, edges in node_edges.items():
            if len(edges) > self.top_k_complements:
                # Sort by weight descending
                edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
                # Mark edges beyond top K for removal
                for neighbor, key, _ in edges_sorted[self.top_k_complements:]:
                    edges_to_remove.append((node, neighbor, key))

        # Remove edges
        removed = set()
        for u, v, key in edges_to_remove:
            edge_tuple = tuple(sorted([u, v])) + (key,)
            if edge_tuple not in removed:
                try:
                    self.graph.remove_edge(u, v, key)
                    removed.add(edge_tuple)
                except:
                    pass

        return current_count - len(removed)

    def get_edge_type_counts(self) -> Dict[str, int]:
        """Get count of edges by type."""
        counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            counts[edge_type] += 1
        return dict(counts)

    def save(self, filepath: str) -> None:
        """Save graph to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to: {filepath}")

    @staticmethod
    def load(filepath: str) -> nx.MultiGraph:
        """Load graph from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def load_transactions_sample(
    filepath: str,
    nrows: int = 10000
) -> pd.DataFrame:
    """Load a sample of transactions for pipeline development."""
    print(f"Loading {nrows:,} rows from transactions...")

    df = pd.read_csv(
        filepath,
        nrows=nrows,
        usecols=[
            'PROD_CODE', 'PROD_CODE_10', 'PROD_CODE_20',
            'PROD_CODE_30', 'PROD_CODE_40', 'STORE_CODE',
            'SHOP_WEEK', 'SPEND', 'QUANTITY', 'CUST_CODE',
            'BASKET_ID'
        ]
    )

    print(f"  - Loaded {len(df):,} transactions")
    print(f"  - Products: {df['PROD_CODE'].nunique():,}")
    print(f"  - Baskets: {df['BASKET_ID'].nunique():,}")
    print(f"  - Customers: {df['CUST_CODE'].nunique():,}")

    return df


def main():
    """Run product graph construction on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'
    output_path = project_root / 'data' / 'processed' / 'product_graph.pkl'

    # Load sample
    transactions_df = load_transactions_sample(str(raw_data_path), nrows=10000)

    # Build graph
    builder = ProductGraphBuilder(
        min_copurchase_count=5,  # Lower threshold for sample data
        top_k_complements=10,
        jaccard_threshold=0.2,
        max_lift_for_substitution=0.5,
        price_gap_threshold=0.5
    )
    graph = builder.run(transactions_df)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder.save(str(output_path))

    # Print edge type distribution
    print("\nEdge type distribution:")
    for edge_type, count in builder.get_edge_type_counts().items():
        print(f"  - {edge_type}: {count:,}")

    return graph


if __name__ == '__main__':
    main()
