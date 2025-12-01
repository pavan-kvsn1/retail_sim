"""
Sample Transactions Data
========================
Creates a smaller sample of transactions.csv with the most active customers.

Uses chunked reading to handle large files without running out of memory.

Usage:
    python scripts/sample_transactions.py --top-customers 100000
    python scripts/sample_transactions.py --top-customers 50000 --output sampled_50k.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import Counter
import time


def count_customer_activity(input_path: Path, chunk_size: int = 500_000) -> Counter:
    """
    Count transactions per customer using chunked reading.

    Parameters
    ----------
    input_path : Path
        Path to transactions.csv
    chunk_size : int
        Rows to read per chunk

    Returns
    -------
    Counter
        Customer ID -> transaction count
    """
    print(f"Counting customer activity from {input_path}...")
    print(f"  Using chunk size: {chunk_size:,}")

    customer_counts = Counter()
    total_rows = 0
    chunk_num = 0

    for chunk in pd.read_csv(input_path, usecols=['CUST_CODE'], chunksize=chunk_size):
        chunk_num += 1
        total_rows += len(chunk)
        customer_counts.update(chunk['CUST_CODE'].values)

        if chunk_num % 10 == 0:
            print(f"  Processed {total_rows:,} rows, {len(customer_counts):,} unique customers...")

    print(f"  Total: {total_rows:,} rows, {len(customer_counts):,} unique customers")
    return customer_counts


def get_top_customers(customer_counts: Counter, top_n: int) -> set:
    """
    Get the top N most active customers.

    Parameters
    ----------
    customer_counts : Counter
        Customer ID -> transaction count
    top_n : int
        Number of top customers to select

    Returns
    -------
    set
        Set of top customer IDs
    """
    top_customers = set(cust for cust, _ in customer_counts.most_common(top_n))

    # Print statistics
    top_counts = [customer_counts[c] for c in top_customers]
    print(f"\nTop {top_n:,} customers:")
    print(f"  Min transactions: {min(top_counts):,}")
    print(f"  Max transactions: {max(top_counts):,}")
    print(f"  Median transactions: {sorted(top_counts)[len(top_counts)//2]:,}")

    return top_customers


def extract_customer_transactions(
    input_path: Path,
    output_path: Path,
    target_customers: set,
    chunk_size: int = 500_000
) -> int:
    """
    Extract all transactions for target customers.

    Parameters
    ----------
    input_path : Path
        Path to input transactions.csv
    output_path : Path
        Path to output sampled file
    target_customers : set
        Set of customer IDs to extract
    chunk_size : int
        Rows to read per chunk

    Returns
    -------
    int
        Number of transactions extracted
    """
    print(f"\nExtracting transactions for {len(target_customers):,} customers...")

    total_extracted = 0
    chunk_num = 0
    first_chunk = True

    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        chunk_num += 1

        # Filter to target customers
        filtered = chunk[chunk['CUST_CODE'].isin(target_customers)]

        if len(filtered) > 0:
            # Write to output (append mode after first chunk)
            filtered.to_csv(
                output_path,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            first_chunk = False
            total_extracted += len(filtered)

        if chunk_num % 10 == 0:
            print(f"  Chunk {chunk_num}: extracted {total_extracted:,} transactions so far...")

    print(f"  Total extracted: {total_extracted:,} transactions")
    return total_extracted


def main():
    parser = argparse.ArgumentParser(description='Sample transactions with top active customers')
    parser.add_argument(
        '--top-customers',
        type=int,
        default=75_000,
        help='Number of top customers to include (default: 75000)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input transactions.csv path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output sampled CSV path'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500_000,
        help='Chunk size for reading (default: 500000)'
    )

    args = parser.parse_args()

    # Set paths
    project_root = Path(__file__).parent.parent
    input_path = Path(args.input) if args.input else project_root / 'raw_data' / 'transactions.csv'

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / 'raw_data' / f'transactions_top{args.top_customers // 1000}k.csv'

    print("=" * 60)
    print("Transaction Sampling Script")
    print("=" * 60)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"Top customers: {args.top_customers:,}")

    start_time = time.time()

    # Step 1: Count customer activity
    customer_counts = count_customer_activity(input_path, args.chunk_size)

    # Step 2: Get top customers
    top_customers = get_top_customers(customer_counts, args.top_customers)

    # Step 3: Extract transactions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_extracted = extract_customer_transactions(
        input_path,
        output_path,
        top_customers,
        args.chunk_size
    )

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SAMPLING COMPLETE")
    print("=" * 60)
    print(f"\nOutput file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e9:.2f} GB")
    print(f"Total transactions: {total_extracted:,}")
    print(f"Total customers: {len(top_customers):,}")
    print(f"Time elapsed: {elapsed:.1f}s")

    # Estimate compression ratio
    original_customers = len(customer_counts)
    print(f"\nCompression ratio:")
    print(f"  Customers: {len(top_customers):,} / {original_customers:,} ({100*len(top_customers)/original_customers:.1f}%)")


if __name__ == '__main__':
    main()
