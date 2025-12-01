"""
Evaluation Script for Section 2: Data Pipeline
===============================================
Evaluates quality and correctness of data pipeline outputs.

Metrics:
- Price derivation coverage and quality
- Graph structure and connectivity
- Customer affinity distribution
- Mission pattern consistency
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import pickle
import json
from typing import Dict, Any


def evaluate_prices(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate price derivation quality.

    Returns metrics on coverage, distribution, and validity.
    """
    metrics = {}

    # Coverage
    metrics['total_records'] = len(prices_df)
    metrics['unique_products'] = prices_df['product_id'].nunique()
    metrics['unique_weeks'] = prices_df['week'].nunique()

    # Price validity
    metrics['negative_prices'] = (prices_df['actual_price'] < 0).sum()
    metrics['zero_prices'] = (prices_df['actual_price'] == 0).sum()
    metrics['nan_prices'] = prices_df['actual_price'].isna().sum()

    # Price distribution
    metrics['price_mean'] = float(prices_df['actual_price'].mean())
    metrics['price_median'] = float(prices_df['actual_price'].median())
    metrics['price_std'] = float(prices_df['actual_price'].std())
    metrics['price_min'] = float(prices_df['actual_price'].min())
    metrics['price_max'] = float(prices_df['actual_price'].max())

    # Discount analysis
    if 'discount_pct' in prices_df.columns:
        metrics['avg_discount'] = float(prices_df['discount_pct'].mean())
        metrics['products_on_discount'] = (prices_df['discount_pct'] > 0.05).sum()

    # Quality score (0-100)
    quality_score = 100
    if metrics['negative_prices'] > 0:
        quality_score -= 20
    if metrics['nan_prices'] > 0:
        quality_score -= 20
    if metrics['zero_prices'] > metrics['total_records'] * 0.1:
        quality_score -= 10

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_graph(graph: nx.Graph) -> Dict[str, Any]:
    """
    Evaluate product graph quality.

    Returns metrics on structure, connectivity, and edge types.
    """
    metrics = {}

    # Basic structure
    metrics['num_nodes'] = graph.number_of_nodes()
    metrics['num_edges'] = graph.number_of_edges()
    metrics['density'] = nx.density(graph) if graph.number_of_nodes() > 1 else 0

    # Node types
    product_nodes = [n for n in graph.nodes() if str(n).startswith('PRD') or str(n).startswith('P')]
    category_nodes = [n for n in graph.nodes() if not (str(n).startswith('PRD') or str(n).startswith('P'))]
    metrics['product_nodes'] = len(product_nodes)
    metrics['category_nodes'] = len(category_nodes)

    # Edge types
    edge_types = {}
    for u, v, d in graph.edges(data=True):
        etype = d.get('edge_type', 'unknown')
        edge_types[etype] = edge_types.get(etype, 0) + 1
    metrics['edge_types'] = edge_types

    # Connectivity
    if graph.number_of_nodes() > 0:
        metrics['num_components'] = nx.number_connected_components(graph)
        largest_cc = max(nx.connected_components(graph), key=len)
        metrics['largest_component_size'] = len(largest_cc)
        metrics['largest_component_pct'] = len(largest_cc) / graph.number_of_nodes()
    else:
        metrics['num_components'] = 0
        metrics['largest_component_size'] = 0
        metrics['largest_component_pct'] = 0

    # Degree statistics
    if graph.number_of_nodes() > 0:
        degrees = dict(graph.degree())
        metrics['avg_degree'] = np.mean(list(degrees.values()))
        metrics['max_degree'] = max(degrees.values())
        metrics['min_degree'] = min(degrees.values())
    else:
        metrics['avg_degree'] = 0
        metrics['max_degree'] = 0
        metrics['min_degree'] = 0

    # Quality score
    quality_score = 100
    if metrics['num_edges'] == 0:
        quality_score -= 30
    if metrics['largest_component_pct'] < 0.5:
        quality_score -= 20
    if metrics['avg_degree'] < 2:
        quality_score -= 10

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_affinity(affinity_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate customer-store affinity quality.

    Returns metrics on coverage and distribution.
    """
    metrics = {}

    # Coverage
    metrics['total_customers'] = len(affinity_df)
    metrics['unique_primary_stores'] = affinity_df['primary_store'].nunique() if 'primary_store' in affinity_df.columns else 0

    # Loyalty score distribution
    if 'loyalty_score' in affinity_df.columns:
        metrics['loyalty_mean'] = float(affinity_df['loyalty_score'].mean())
        metrics['loyalty_std'] = float(affinity_df['loyalty_score'].std())
        metrics['high_loyalty_pct'] = float((affinity_df['loyalty_score'] > 0.7).mean())
        metrics['low_loyalty_pct'] = float((affinity_df['loyalty_score'] < 0.3).mean())

    # Store concentration (HHI)
    if 'store_concentration' in affinity_df.columns:
        metrics['hhi_mean'] = float(affinity_df['store_concentration'].mean())
        metrics['single_store_pct'] = float((affinity_df['store_concentration'] == 1.0).mean())

    # Switching rate
    if 'switching_rate' in affinity_df.columns:
        metrics['switching_mean'] = float(affinity_df['switching_rate'].mean())
        metrics['never_switch_pct'] = float((affinity_df['switching_rate'] == 0).mean())

    # Quality score
    quality_score = 100
    if metrics['total_customers'] == 0:
        quality_score = 0
    elif 'loyalty_score' in affinity_df.columns:
        if affinity_df['loyalty_score'].isna().any():
            quality_score -= 20

    metrics['quality_score'] = quality_score

    return metrics


def evaluate_missions(missions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate mission pattern extraction quality.

    Returns metrics on coverage and distribution consistency.
    """
    metrics = {}

    # Coverage
    metrics['total_customers'] = len(missions_df)

    # Mission type distribution
    mission_cols = [c for c in missions_df.columns if c.startswith('p_mission_')]
    if mission_cols:
        for col in mission_cols:
            metrics[f'avg_{col}'] = float(missions_df[col].mean())

        # Check probabilities sum to 1
        prob_sums = missions_df[mission_cols].sum(axis=1)
        metrics['prob_sum_mean'] = float(prob_sums.mean())
        metrics['prob_sum_valid'] = float(((prob_sums > 0.99) & (prob_sums < 1.01)).mean())

    # Mission focus distribution
    focus_cols = [c for c in missions_df.columns if c.startswith('p_focus_')]
    if focus_cols:
        for col in focus_cols:
            metrics[f'avg_{col}'] = float(missions_df[col].mean())

    # Quality score
    quality_score = 100
    if metrics['total_customers'] == 0:
        quality_score = 0
    elif mission_cols and metrics.get('prob_sum_valid', 0) < 0.95:
        quality_score -= 30

    metrics['quality_score'] = quality_score

    return metrics


def run_evaluation(project_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Run complete data pipeline evaluation.

    Parameters
    ----------
    project_root : Path
        Project root directory

    Returns
    -------
    Dict containing evaluation results for each stage
    """
    results = {}
    processed_dir = project_root / 'data' / 'processed'

    print("=" * 60)
    print("Data Pipeline Evaluation")
    print("=" * 60)

    # Stage 1: Price Derivation
    print("\n--- Stage 1: Price Derivation ---")
    prices_path = processed_dir / 'prices_derived.parquet'
    if prices_path.exists():
        prices_df = pd.read_parquet(prices_path)
        results['prices'] = evaluate_prices(prices_df)
        print(f"  Records: {results['prices']['total_records']:,}")
        print(f"  Products: {results['prices']['unique_products']:,}")
        print(f"  Quality Score: {results['prices']['quality_score']}/100")
    else:
        print("  [MISSING] prices_derived.parquet")
        results['prices'] = {'quality_score': 0, 'error': 'file not found'}

    # Stage 2: Product Graph
    print("\n--- Stage 2: Product Graph ---")
    graph_path = processed_dir / 'product_graph.pkl'
    if graph_path.exists():
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        results['graph'] = evaluate_graph(graph)
        print(f"  Nodes: {results['graph']['num_nodes']:,}")
        print(f"  Edges: {results['graph']['num_edges']:,}")
        print(f"  Components: {results['graph']['num_components']}")
        print(f"  Quality Score: {results['graph']['quality_score']}/100")
    else:
        print("  [MISSING] product_graph.pkl")
        results['graph'] = {'quality_score': 0, 'error': 'file not found'}

    # Stage 3: Customer-Store Affinity
    print("\n--- Stage 3: Customer-Store Affinity ---")
    affinity_path = processed_dir / 'customer_store_affinity.parquet'
    if affinity_path.exists():
        affinity_df = pd.read_parquet(affinity_path)
        results['affinity'] = evaluate_affinity(affinity_df)
        print(f"  Customers: {results['affinity']['total_customers']:,}")
        if 'loyalty_mean' in results['affinity']:
            print(f"  Avg Loyalty: {results['affinity']['loyalty_mean']:.3f}")
        print(f"  Quality Score: {results['affinity']['quality_score']}/100")
    else:
        print("  [MISSING] customer_store_affinity.parquet")
        results['affinity'] = {'quality_score': 0, 'error': 'file not found'}

    # Stage 4: Mission Patterns
    print("\n--- Stage 4: Mission Patterns ---")
    missions_path = processed_dir / 'customer_mission_patterns.parquet'
    if missions_path.exists():
        missions_df = pd.read_parquet(missions_path)
        results['missions'] = evaluate_missions(missions_df)
        print(f"  Customers: {results['missions']['total_customers']:,}")
        print(f"  Quality Score: {results['missions']['quality_score']}/100")
    else:
        print("  [MISSING] customer_mission_patterns.parquet")
        results['missions'] = {'quality_score': 0, 'error': 'file not found'}

    # Overall score
    scores = [r['quality_score'] for r in results.values() if 'quality_score' in r]
    overall_score = np.mean(scores) if scores else 0

    print("\n" + "=" * 60)
    print(f"Overall Quality Score: {overall_score:.1f}/100")
    print("=" * 60)

    results['overall'] = {
        'quality_score': overall_score,
        'stages_evaluated': len(scores),
        'all_files_present': all('error' not in r for r in results.values())
    }

    return results


def main():
    """Run evaluation and save results."""
    project_root = Path(__file__).parent.parent
    results = run_evaluation(project_root)

    # Save results
    output_path = project_root / 'evals' / 'data_pipeline_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
