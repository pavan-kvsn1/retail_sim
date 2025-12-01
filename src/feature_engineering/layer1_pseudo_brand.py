"""
Layer 1: Pseudo-Brand Inference
================================
Infers brand-like clusters from observable signals since LGSR lacks explicit brands.

Steps:
1. Category Grouping (by sub-commodity PROD_CODE_10)
2. Price Tier Clustering (Premium/Mid/Value)
3. Substitution Pattern Analysis (from product graph)
4. Pseudo-Brand Assignment

Output: pseudo_brands.parquet
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import hashlib
import pickle
import warnings

warnings.filterwarnings('ignore')


class PseudoBrandInference:
    """
    Infers pseudo-brands from price positioning and substitution patterns.

    Pseudo-Brand = f(Sub_Commodity, Price_Tier, Substitution_Cluster)
    """

    def __init__(
        self,
        premium_percentile: float = 0.80,
        value_percentile: float = 0.20,
        substitution_threshold: float = 0.5
    ):
        """
        Parameters
        ----------
        premium_percentile : float
            Price percentile threshold for premium tier (default 80th)
        value_percentile : float
            Price percentile threshold for value tier (default 20th)
        substitution_threshold : float
            Minimum substitution edge weight to cluster together
        """
        self.premium_percentile = premium_percentile
        self.value_percentile = value_percentile
        self.substitution_threshold = substitution_threshold

    def run(
        self,
        transactions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        product_graph: Optional[nx.MultiGraph] = None
    ) -> pd.DataFrame:
        """
        Execute pseudo-brand inference pipeline.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Raw transactions with product hierarchy
        prices_df : pd.DataFrame
            Derived prices from Stage 1
        product_graph : nx.MultiGraph, optional
            Product graph from Stage 2 for substitution analysis

        Returns
        -------
        pd.DataFrame
            Pseudo-brand features per product
        """
        print("Layer 1: Pseudo-Brand Inference")
        print("=" * 50)

        # Step 1: Extract product hierarchy and aggregate prices
        print("\nStep 1: Extracting product information...")
        product_info = self._extract_product_info(transactions_df, prices_df)
        print(f"  - Products: {len(product_info):,}")
        print(f"  - Categories (sub-commodity): {product_info['sub_commodity'].nunique():,}")

        # Step 2: Compute price tiers within each category
        print("\nStep 2: Computing price tiers...")
        product_info = self._compute_price_tiers(product_info)
        tier_dist = product_info['price_tier'].value_counts()
        print(f"  - Premium: {tier_dist.get('premium', 0):,}")
        print(f"  - Mid: {tier_dist.get('mid', 0):,}")
        print(f"  - Value: {tier_dist.get('value', 0):,}")

        # Step 3: Substitution clustering (if graph available)
        print("\nStep 3: Analyzing substitution patterns...")
        if product_graph is not None:
            product_info = self._compute_substitution_clusters(product_info, product_graph)
            print(f"  - Substitution clusters: {product_info['substitution_cluster'].nunique():,}")
        else:
            product_info['substitution_cluster'] = 0
            print("  - No graph provided, skipping substitution clustering")

        # Step 4: Assign pseudo-brand IDs
        print("\nStep 4: Assigning pseudo-brand IDs...")
        product_info = self._assign_pseudo_brands(product_info)
        print(f"  - Unique pseudo-brands: {product_info['pseudo_brand_id'].nunique():,}")

        # Step 5: Compute additional brand features
        print("\nStep 5: Computing brand features...")
        product_info = self._compute_brand_features(product_info, product_graph)

        print("\n" + "=" * 50)
        print("Pseudo-Brand Inference Complete!")
        print(f"  - Total products: {len(product_info):,}")
        print(f"  - Pseudo-brands: {product_info['pseudo_brand_id'].nunique():,}")

        return product_info

    def _extract_product_info(
        self,
        transactions_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract product hierarchy and compute average prices."""
        # Get unique products with hierarchy
        products = transactions_df.groupby('PROD_CODE').agg({
            'PROD_CODE_10': 'first',  # Sub-commodity (CL)
            'PROD_CODE_20': 'first',  # Commodity (DEP)
            'PROD_CODE_30': 'first',  # Sub-department (G)
            'PROD_CODE_40': 'first',  # Department (D)
            'QUANTITY': 'sum',
            'SPEND': 'sum'
        }).reset_index()

        products.columns = [
            'product_id', 'sub_commodity', 'commodity',
            'sub_department', 'department', 'total_quantity', 'total_spend'
        ]

        # Compute average price from transactions
        products['avg_price_txn'] = products['total_spend'] / products['total_quantity'].clip(1)

        # Merge with derived prices if available
        if prices_df is not None and len(prices_df) > 0:
            price_agg = prices_df.groupby('product_id').agg(
                mean_price=('actual_price', 'mean'),
                median_price=('actual_price', 'median'),
                price_std=('actual_price', 'std'),
                base_price=('base_price', 'mean'),
                avg_discount=('discount_depth', 'mean'),
                promo_frequency=('promo_flag', 'mean')
            ).reset_index()

            products = products.merge(price_agg, on='product_id', how='left')

            # Fill missing price data with transaction-derived prices
            products['mean_price'] = products['mean_price'].fillna(products['avg_price_txn'])
            products['median_price'] = products['median_price'].fillna(products['avg_price_txn'])
            products['price_std'] = products['price_std'].fillna(0)
            products['base_price'] = products['base_price'].fillna(products['avg_price_txn'])
            products['avg_discount'] = products['avg_discount'].fillna(0)
            products['promo_frequency'] = products['promo_frequency'].fillna(0)
        else:
            # Use transaction-derived prices
            products['mean_price'] = products['avg_price_txn']
            products['median_price'] = products['avg_price_txn']
            products['price_std'] = 0
            products['base_price'] = products['avg_price_txn']
            products['avg_discount'] = 0
            products['promo_frequency'] = 0

        return products

    def _compute_price_tiers(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price tiers (Premium/Mid/Value) within each sub-commodity.
        """
        products_df = products_df.copy()

        def assign_tier(group):
            """Assign tier based on percentile within category."""
            prices = group['mean_price']

            if len(prices) < 3:
                # Not enough products to tier
                return pd.Series(['mid'] * len(group), index=group.index)

            premium_threshold = prices.quantile(self.premium_percentile)
            value_threshold = prices.quantile(self.value_percentile)

            tiers = []
            for price in prices:
                if price >= premium_threshold:
                    tiers.append('premium')
                elif price <= value_threshold:
                    tiers.append('value')
                else:
                    tiers.append('mid')

            return pd.Series(tiers, index=group.index)

        # Compute tier within each sub-commodity
        products_df['price_tier'] = products_df.groupby('sub_commodity').apply(
            assign_tier
        ).reset_index(level=0, drop=True)

        # Compute price percentile within category
        products_df['category_price_percentile'] = products_df.groupby('sub_commodity')['mean_price'].transform(
            lambda x: x.rank(pct=True)
        )

        # Compute coefficient of variation (price stability)
        products_df['price_cv'] = products_df['price_std'] / products_df['mean_price'].clip(0.01)
        products_df['price_cv'] = products_df['price_cv'].fillna(0).clip(0, 2)

        return products_df

    def _compute_substitution_clusters(
        self,
        products_df: pd.DataFrame,
        product_graph: nx.MultiGraph
    ) -> pd.DataFrame:
        """
        Cluster products based on substitution edges in the graph.
        """
        products_df = products_df.copy()

        # Build substitution subgraph
        substitution_edges = []
        for u, v, data in product_graph.edges(data=True):
            if data.get('edge_type') == 'substitution':
                weight = data.get('weight', 0)
                if weight >= self.substitution_threshold:
                    substitution_edges.append((u, v, weight))

        if not substitution_edges:
            products_df['substitution_cluster'] = 0
            return products_df

        # Create substitution graph
        sub_graph = nx.Graph()
        for u, v, w in substitution_edges:
            sub_graph.add_edge(u, v, weight=w)

        # Find connected components as clusters
        clusters = {}
        for cluster_id, component in enumerate(nx.connected_components(sub_graph)):
            for node in component:
                clusters[node] = cluster_id

        # Map clusters to products
        products_df['substitution_cluster'] = products_df['product_id'].map(clusters).fillna(-1).astype(int)

        # Re-number clusters by category to make them category-specific
        def renumber_by_category(group):
            unique_clusters = group['substitution_cluster'].unique()
            cluster_map = {c: i for i, c in enumerate(unique_clusters)}
            return group['substitution_cluster'].map(cluster_map)

        products_df['substitution_cluster'] = products_df.groupby('sub_commodity').apply(
            renumber_by_category
        ).reset_index(level=0, drop=True)

        return products_df

    def _assign_pseudo_brands(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign pseudo-brand IDs based on (sub_commodity, price_tier, substitution_cluster).
        """
        products_df = products_df.copy()

        def create_pseudo_brand_id(row):
            """Create unique pseudo-brand ID from components."""
            components = [
                str(row['sub_commodity']),
                str(row['price_tier']),
                str(row['substitution_cluster'])
            ]
            brand_string = "_".join(components)
            # Create short hash for ID
            hash_id = hashlib.md5(brand_string.encode()).hexdigest()[:8]
            return f"{row['sub_commodity']}_{row['price_tier']}_{hash_id}"

        products_df['pseudo_brand_id'] = products_df.apply(create_pseudo_brand_id, axis=1)

        # Create numeric pseudo-brand index for embedding
        unique_brands = products_df['pseudo_brand_id'].unique()
        brand_to_idx = {brand: idx for idx, brand in enumerate(unique_brands)}
        products_df['pseudo_brand_idx'] = products_df['pseudo_brand_id'].map(brand_to_idx)

        return products_df

    def _compute_brand_features(
        self,
        products_df: pd.DataFrame,
        product_graph: Optional[nx.MultiGraph] = None
    ) -> pd.DataFrame:
        """
        Compute additional brand-level features.
        """
        products_df = products_df.copy()

        # Competitive intensity: number of direct substitutes
        if product_graph is not None:
            def count_substitutes(product_id):
                if product_id not in product_graph:
                    return 0
                count = 0
                for _, neighbor, data in product_graph.edges(product_id, data=True):
                    if data.get('edge_type') == 'substitution':
                        count += 1
                return count

            products_df['competitive_intensity'] = products_df['product_id'].apply(count_substitutes)
        else:
            products_df['competitive_intensity'] = 0

        # Market share proxy: % of category volume
        category_volume = products_df.groupby('sub_commodity')['total_quantity'].transform('sum')
        products_df['market_share_proxy'] = products_df['total_quantity'] / category_volume.clip(1)

        # Price positioning: normalized price within category
        category_mean_price = products_df.groupby('sub_commodity')['mean_price'].transform('mean')
        products_df['relative_price'] = products_df['mean_price'] / category_mean_price.clip(0.01)

        # Brand size: number of products in pseudo-brand
        brand_size = products_df.groupby('pseudo_brand_id').size()
        products_df['brand_size'] = products_df['pseudo_brand_id'].map(brand_size)

        # Select final columns
        output_columns = [
            'product_id',
            'sub_commodity', 'commodity', 'sub_department', 'department',
            'pseudo_brand_id', 'pseudo_brand_idx',
            'price_tier',
            'category_price_percentile',
            'price_cv',
            'competitive_intensity',
            'market_share_proxy',
            'relative_price',
            'brand_size',
            'mean_price', 'base_price', 'avg_discount', 'promo_frequency'
        ]

        return products_df[output_columns]


def main():
    """Run pseudo-brand inference on sample data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    transactions_path = project_root / 'raw_data' / 'transactions.csv'
    prices_path = project_root / 'data' / 'processed' / 'prices_derived.parquet'
    graph_path = project_root / 'data' / 'processed' / 'product_graph.pkl'
    output_path = project_root / 'data' / 'features' / 'pseudo_brands.parquet'

    # Load data
    print("Loading data...")
    transactions_df = pd.read_csv(
        transactions_path,
        nrows=10000,
        usecols=[
            'PROD_CODE', 'PROD_CODE_10', 'PROD_CODE_20',
            'PROD_CODE_30', 'PROD_CODE_40', 'QUANTITY', 'SPEND'
        ]
    )

    prices_df = pd.read_parquet(prices_path) if prices_path.exists() else None

    product_graph = None
    if graph_path.exists():
        with open(graph_path, 'rb') as f:
            product_graph = pickle.load(f)

    # Run inference
    inferencer = PseudoBrandInference()
    pseudo_brands = inferencer.run(transactions_df, prices_df, product_graph)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_brands.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Display sample
    print("\nSample output:")
    print(pseudo_brands.head(10).to_string())

    return pseudo_brands


if __name__ == '__main__':
    main()
