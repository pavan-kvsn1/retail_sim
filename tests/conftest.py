"""
Pytest Configuration and Fixtures
==================================
Shared fixtures for RetailSim tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_transactions(project_root):
    """Load sample transactions for testing."""
    transactions_path = project_root / 'raw_data' / 'transactions.csv'
    if transactions_path.exists():
        return pd.read_csv(transactions_path, nrows=1000)
    else:
        # Generate synthetic data for CI environments
        return generate_synthetic_transactions(1000)


@pytest.fixture(scope="session")
def mini_transactions():
    """Generate minimal synthetic transactions for unit tests."""
    return generate_synthetic_transactions(100)


def generate_synthetic_transactions(n_rows: int) -> pd.DataFrame:
    """Generate synthetic transaction data for testing."""
    np.random.seed(42)

    n_customers = max(10, n_rows // 10)
    n_products = max(50, n_rows // 5)
    n_stores = 3
    n_baskets = max(20, n_rows // 5)

    # Generate IDs
    customer_ids = [f'CUST{i:010d}' for i in range(n_customers)]
    product_ids = [f'PRD{i:08d}' for i in range(n_products)]
    store_ids = [f'STORE{i:05d}' for i in range(n_stores)]
    basket_ids = [f'BASKET{i:010d}' for i in range(n_baskets)]

    # Generate category hierarchy
    prod_code_10 = [f'D{i % 5:02d}' for i in range(n_products)]
    prod_code_20 = [f'C{i % 20:02d}' for i in range(n_products)]
    prod_code_30 = [f'SC{i % 50:02d}' for i in range(n_products)]
    prod_code_40 = [f'SSC{i % 100:03d}' for i in range(n_products)]

    data = {
        'CUST_CODE': np.random.choice(customer_ids, n_rows),
        'PROD_CODE': np.random.choice(product_ids, n_rows),
        'STORE_CODE': np.random.choice(store_ids, n_rows),
        'BASKET_ID': np.random.choice(basket_ids, n_rows),
        'SHOP_WEEK': np.random.randint(200601, 200652, n_rows),
        'SHOP_WEEKDAY': np.random.randint(1, 8, n_rows),
        'SHOP_HOUR': np.random.randint(8, 22, n_rows),
        'SPEND': np.random.uniform(0.5, 50.0, n_rows).round(2),
        'QUANTITY': np.random.randint(1, 5, n_rows),
        'STORE_FORMAT': np.random.choice(['LS', 'MS', 'SS'], n_rows),
        'STORE_REGION': np.random.choice(['E01', 'E02', 'W01', 'W02'], n_rows),
        'BASKET_TYPE': np.random.choice(['Top Up', 'Full Shop', 'Small Shop', 'Emergency'], n_rows),
        'BASKET_DOMINANT_MISSION': np.random.choice(['Fresh', 'Grocery', 'Mixed', 'Nonfood'], n_rows),
        'BASKET_PRICE_SENSITIVITY': np.random.choice(['LA', 'MM', 'UM'], n_rows),
        'BASKET_SIZE': np.random.choice(['S', 'M', 'L'], n_rows),
    }

    # Add category codes based on product
    df = pd.DataFrame(data)
    product_to_idx = {p: i for i, p in enumerate(product_ids)}
    df['PROD_CODE_10'] = df['PROD_CODE'].map(lambda x: prod_code_10[product_to_idx.get(x, 0)])
    df['PROD_CODE_20'] = df['PROD_CODE'].map(lambda x: prod_code_20[product_to_idx.get(x, 0)])
    df['PROD_CODE_30'] = df['PROD_CODE'].map(lambda x: prod_code_30[product_to_idx.get(x, 0)])
    df['PROD_CODE_40'] = df['PROD_CODE'].map(lambda x: prod_code_40[product_to_idx.get(x, 0)])

    return df


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture(scope="session")
def sample_prices(sample_transactions):
    """Generate sample price data."""
    products = sample_transactions['PROD_CODE'].unique()
    weeks = sample_transactions['SHOP_WEEK'].unique()

    prices = []
    for prod in products[:100]:  # Limit for speed
        base_price = np.random.uniform(1, 20)
        for week in weeks[:10]:
            actual_price = base_price * np.random.uniform(0.8, 1.0)
            prices.append({
                'product_id': prod,
                'week': week,
                'actual_price': round(actual_price, 2),
                'base_price': round(base_price, 2),
                'discount_pct': round((base_price - actual_price) / base_price, 3),
                'price_rank': np.random.uniform(0, 1)
            })

    return pd.DataFrame(prices)


@pytest.fixture(scope="session")
def sample_product_embeddings():
    """Generate sample product embeddings."""
    np.random.seed(42)
    n_products = 100
    embedding_dim = 256

    return {
        f'PRD{i:08d}': np.random.randn(embedding_dim).astype(np.float32)
        for i in range(n_products)
    }


@pytest.fixture(scope="session")
def sample_customer_embeddings():
    """Generate sample customer embeddings DataFrame."""
    np.random.seed(42)
    n_customers = 50
    embedding_dim = 160

    rows = []
    for i in range(n_customers):
        row = {
            'customer_id': f'CUST{i:010d}',
            'total_trips': np.random.randint(1, 20)
        }
        embed = np.random.randn(embedding_dim).astype(np.float32)
        for j, val in enumerate(embed):
            row[f'embed_{j}'] = val
        rows.append(row)

    return pd.DataFrame(rows)
