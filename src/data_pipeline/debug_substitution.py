"""
Debug script to diagnose why substitution edges are not being created.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations


def diagnose_substitution_edges(
    df: pd.DataFrame,
    jaccard_threshold: float = 0.2,
    max_lift_for_substitution: float = 0.5,
    price_gap_threshold: float = 0.5,
    verbose: bool = True
):
    """
    Diagnose why substitution edges might not be created.

    Returns detailed statistics at each filtering stage.
    """
    stats = {}

    # Step 1: Check categories
    product_info = df.groupby('PROD_CODE').agg({
        'PROD_CODE_10': 'first',
        'SPEND': 'sum',
        'QUANTITY': 'sum'
    }).reset_index()

    product_category = dict(zip(product_info['PROD_CODE'], product_info['PROD_CODE_10']))

    # Compute average price
    product_info['avg_price'] = product_info['SPEND'] / product_info['QUANTITY'].clip(1)
    product_price = dict(zip(product_info['PROD_CODE'], product_info['avg_price']))

    # Check for zero prices
    zero_price_products = [p for p, price in product_price.items() if price == 0]
    stats['zero_price_products'] = len(zero_price_products)

    if verbose:
        print("=" * 60)
        print("SUBSTITUTION EDGE DIAGNOSTIC")
        print("=" * 60)
        print(f"\nTotal products: {len(product_info)}")
        print(f"Products with zero price: {len(zero_price_products)}")
        if zero_price_products[:5]:
            print(f"  Examples: {zero_price_products[:5]}")

    # Step 2: Check categories
    category_products = defaultdict(list)
    for p, cat in product_category.items():
        if pd.notna(cat):
            category_products[cat].append(p)

    categories_with_multiple = {cat: prods for cat, prods in category_products.items() if len(prods) >= 2}
    stats['total_categories'] = len(category_products)
    stats['categories_with_2plus_products'] = len(categories_with_multiple)

    if verbose:
        print(f"\nTotal sub-commodity categories: {len(category_products)}")
        print(f"Categories with 2+ products: {len(categories_with_multiple)}")

    # Count total possible pairs within categories
    total_pairs = sum(len(prods) * (len(prods) - 1) // 2 for prods in categories_with_multiple.values())
    stats['total_possible_pairs'] = total_pairs

    if verbose:
        print(f"Total product pairs within same category: {total_pairs:,}")

    # Step 3: Build customer mappings
    customer_products = df.groupby('CUST_CODE')['PROD_CODE'].apply(set).to_dict()

    product_customers = defaultdict(set)
    for cust, products in customer_products.items():
        if pd.isna(cust):
            continue
        for p in products:
            product_customers[p].add(cust)

    # Check customer counts
    products_with_5plus_customers = [p for p, custs in product_customers.items() if len(custs) >= 5]
    stats['products_with_5plus_customers'] = len(products_with_5plus_customers)

    if verbose:
        print(f"\nProducts with 5+ customers: {len(products_with_5plus_customers)}")
        customer_counts = [len(custs) for custs in product_customers.values()]
        print(f"Customer count stats: min={min(customer_counts)}, max={max(customer_counts)}, median={np.median(customer_counts):.0f}")

    # Step 4: Analyze pairs stage by stage
    pairs_by_stage = {
        'total_in_category': 0,
        'pass_customer_filter': 0,
        'pass_jaccard': 0,
        'pass_lift': 0,
        'pass_price_nonzero': 0,
        'pass_price_gap': 0,
    }

    # Sample analysis - do detailed check on subset of categories
    sample_categories = list(categories_with_multiple.keys())[:50]

    failing_reasons = defaultdict(int)
    passing_pairs = []

    for category in sample_categories:
        products = categories_with_multiple[category][:100]  # Limit per category

        for p1, p2 in combinations(products, 2):
            pairs_by_stage['total_in_category'] += 1

            # Check customer counts
            customers_1 = product_customers.get(p1, set())
            customers_2 = product_customers.get(p2, set())

            if len(customers_1) < 5 or len(customers_2) < 5:
                failing_reasons['insufficient_customers'] += 1
                continue
            pairs_by_stage['pass_customer_filter'] += 1

            # Check Jaccard
            intersection = len(customers_1 & customers_2)
            union = len(customers_1 | customers_2)
            jaccard = intersection / union if union > 0 else 0

            if jaccard < jaccard_threshold:
                failing_reasons[f'jaccard_below_{jaccard_threshold}'] += 1
                continue
            pairs_by_stage['pass_jaccard'] += 1

            # Check lift (using 0.5 default since we don't have lift cache here)
            lift = 0.5  # Default assumption
            if lift > max_lift_for_substitution:
                failing_reasons[f'lift_above_{max_lift_for_substitution}'] += 1
                continue
            pairs_by_stage['pass_lift'] += 1

            # Check prices
            price_1 = product_price.get(p1, 0)
            price_2 = product_price.get(p2, 0)

            if price_1 <= 0 or price_2 <= 0:
                failing_reasons['zero_or_missing_price'] += 1
                continue
            pairs_by_stage['pass_price_nonzero'] += 1

            # Check price gap
            price_gap = abs(price_1 - price_2) / ((price_1 + price_2) / 2)
            if price_gap > price_gap_threshold:
                failing_reasons[f'price_gap_above_{price_gap_threshold}'] += 1
                continue
            pairs_by_stage['pass_price_gap'] += 1

            passing_pairs.append({
                'p1': p1, 'p2': p2, 'category': category,
                'jaccard': jaccard, 'price_gap': price_gap,
                'price_1': price_1, 'price_2': price_2
            })

    stats['pairs_by_stage'] = pairs_by_stage
    stats['failing_reasons'] = dict(failing_reasons)
    stats['passing_pairs_count'] = len(passing_pairs)

    if verbose:
        print(f"\n--- PAIR FILTERING STAGES (sample of {len(sample_categories)} categories) ---")
        for stage, count in pairs_by_stage.items():
            print(f"  {stage}: {count:,}")

        print(f"\n--- FAILURE REASONS ---")
        for reason, count in sorted(failing_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count:,}")

        print(f"\n--- RESULT ---")
        print(f"Passing pairs: {len(passing_pairs)}")

        if passing_pairs:
            print("\nExample passing pairs:")
            for pair in passing_pairs[:5]:
                print(f"  {pair['p1']} <-> {pair['p2']}")
                print(f"    Category: {pair['category']}, Jaccard: {pair['jaccard']:.3f}, Price gap: {pair['price_gap']:.3f}")

    return stats, passing_pairs


def suggest_fixes(stats: dict):
    """Suggest parameter adjustments based on diagnostic stats."""
    print("\n" + "=" * 60)
    print("SUGGESTED FIXES")
    print("=" * 60)

    failing_reasons = stats.get('failing_reasons', {})

    if failing_reasons.get('insufficient_customers', 0) > stats['pairs_by_stage']['total_in_category'] * 0.5:
        print("\n1. CUSTOMER COUNT ISSUE")
        print("   Many products have < 5 customers.")
        print("   Fix: Lower the min customer threshold in _build_substitution_edges")
        print("   Change line 273-274 from 'if len(customers_1) < 5' to 'if len(customers_1) < 2'")

    jaccard_failures = sum(v for k, v in failing_reasons.items() if 'jaccard' in k)
    if jaccard_failures > stats['pairs_by_stage']['pass_customer_filter'] * 0.8:
        print("\n2. JACCARD THRESHOLD TOO HIGH")
        print("   Most pairs fail Jaccard even at 0.2.")
        print("   This suggests customers rarely buy multiple products in same category.")
        print("   Fix: Lower jaccard_threshold to 0.1 or even 0.05")
        print("   OR: Change Jaccard to consider 'either bought' instead of strict intersection")

    price_failures = failing_reasons.get('zero_or_missing_price', 0)
    if price_failures > stats['pairs_by_stage']['pass_jaccard'] * 0.3:
        print("\n3. ZERO/MISSING PRICE ISSUE")
        print(f"   {price_failures} pairs rejected due to zero prices.")
        print("   Fix: Impute missing prices before graph construction")
        print("   OR: Skip price check for products without price data (allow them through)")

    price_gap_failures = sum(v for k, v in failing_reasons.items() if 'price_gap' in k)
    if price_gap_failures > stats['pairs_by_stage']['pass_price_nonzero'] * 0.5:
        print("\n4. PRICE GAP THRESHOLD TOO STRICT")
        print(f"   {price_gap_failures} pairs rejected due to price difference.")
        print("   Fix: Increase price_gap_threshold to 0.7 or 1.0")
        print("   OR: Use relative price tier matching instead of absolute gap")


if __name__ == '__main__':
    from pathlib import Path

    # Load data
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / 'raw_data' / 'transactions.csv'

    if raw_data_path.exists():
        print(f"Loading data from {raw_data_path}...")
        df = pd.read_csv(
            raw_data_path,
            nrows=50000,  # Sample
            usecols=['PROD_CODE', 'PROD_CODE_10', 'CUST_CODE', 'SPEND', 'QUANTITY']
        )

        stats, pairs = diagnose_substitution_edges(
            df,
            jaccard_threshold=0.2,
            max_lift_for_substitution=0.5,
            price_gap_threshold=0.5
        )

        suggest_fixes(stats)
    else:
        print(f"Data file not found: {raw_data_path}")
        print("Please provide the path to your transactions data.")
