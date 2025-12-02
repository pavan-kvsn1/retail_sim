"""
Extract basket attributes from transactions.csv into a small parquet file.
This avoids uploading the 30GB transactions file to RunPod.

Usage:
    python scripts/extract_basket_attributes.py

Output:
    data/prepared/basket_attributes.parquet (~50-100 MB)
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    project_root = Path(__file__).parent.parent
    transactions_path = project_root / 'raw_data' / 'transactions.csv'
    output_path = project_root / 'data' / 'prepared' / 'basket_attributes.parquet'

    logger.info(f"Reading transactions from {transactions_path}...")

    # Only read the columns we need
    df = pd.read_csv(
        transactions_path,
        usecols=['BASKET_ID', 'BASKET_TYPE', 'BASKET_DOMINANT_MISSION',
                 'BASKET_PRICE_SENSITIVITY', 'BASKET_SIZE']
    )

    logger.info(f"Loaded {len(df):,} transaction rows")

    # Get unique basket attributes (one row per basket)
    basket_attrs = df.groupby('BASKET_ID').first().reset_index()

    logger.info(f"Extracted {len(basket_attrs):,} unique baskets")

    # Save as parquet (much smaller than CSV)
    basket_attrs.to_parquet(output_path, index=False)

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved to {output_path} ({file_size:.1f} MB)")

if __name__ == '__main__':
    main()
