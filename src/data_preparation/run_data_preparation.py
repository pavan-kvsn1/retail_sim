"""
Run Data Preparation Pipeline
=============================
Orchestrates all 4 stages of data preparation for RetailSim training.

Usage:
    python run_data_preparation.py [--skip-stage N] [--only-stage N]

Stages:
    1. Temporal Metadata - Split assignments and flags
    2. Customer Histories - Per-split history extraction
    3. Training Samples - Bucketed by history length
    4. Tensor Cache - Static embedding caching
"""

import argparse
import pandas as pd
import time
from pathlib import Path
from typing import Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preparation.stage1_temporal_metadata import TemporalMetadataCreator
from src.data_preparation.stage2_customer_histories import CustomerHistoryExtractor
from src.data_preparation.stage3_training_samples import TrainingSampleCreator
from src.data_preparation.stage4_tensor_cache import TensorCacheBuilder


class DataPreparationPipeline:
    """
    Orchestrates the full data preparation pipeline.

    Stages are run sequentially, with outputs from earlier
    stages feeding into later stages.
    """

    def __init__(
        self,
        raw_data_path: Path,
        output_dir: Path,
        train_end_week: int = 80,
        val_end_week: int = 95
    ):
        """
        Parameters
        ----------
        raw_data_path : Path
            Path to raw transactions CSV
        output_dir : Path
            Base output directory for prepared data
        train_end_week : int
            Last week of training set
        val_end_week : int
            Last week of validation set
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.train_end_week = train_end_week
        self.val_end_week = val_end_week

        # Output paths
        self.metadata_path = self.output_dir / 'temporal_metadata.parquet'
        self.histories_path = self.output_dir / 'customer_histories.parquet'
        self.samples_dir = self.output_dir / 'samples'
        self.cache_dir = self.output_dir / 'tensor_cache'

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        skip_stages: Optional[list] = None,
        only_stage: Optional[int] = None
    ):
        """
        Run the full pipeline.

        Parameters
        ----------
        skip_stages : list, optional
            List of stage numbers to skip
        only_stage : int, optional
            If provided, only run this stage
        """
        skip_stages = skip_stages or []

        print("\n" + "=" * 70)
        print("RETAILSIM DATA PREPARATION PIPELINE")
        print("=" * 70)
        print(f"\nRaw data: {self.raw_data_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Train weeks: 1-{self.train_end_week}")
        print(f"Validation weeks: {self.train_end_week + 1}-{self.val_end_week}")
        print(f"Test weeks: {self.val_end_week + 1}+")

        total_start = time.time()

        # Load raw transactions once
        print("\n" + "-" * 70)
        print("Loading raw transactions...")
        transactions_df = self._load_transactions()
        print(f"Loaded {len(transactions_df):,} transactions")

        # Stage 1: Temporal Metadata
        if self._should_run_stage(1, skip_stages, only_stage):
            print("\n" + "-" * 70)
            start = time.time()
            self._run_stage1(transactions_df)
            print(f"\nStage 1 completed in {time.time() - start:.1f}s")

        # Stage 2: Customer Histories
        if self._should_run_stage(2, skip_stages, only_stage):
            print("\n" + "-" * 70)
            start = time.time()
            self._run_stage2(transactions_df)
            print(f"\nStage 2 completed in {time.time() - start:.1f}s")

        # Stage 3: Training Samples
        if self._should_run_stage(3, skip_stages, only_stage):
            print("\n" + "-" * 70)
            start = time.time()
            self._run_stage3(transactions_df)
            print(f"\nStage 3 completed in {time.time() - start:.1f}s")

        # Stage 4: Tensor Cache
        if self._should_run_stage(4, skip_stages, only_stage):
            print("\n" + "-" * 70)
            start = time.time()
            self._run_stage4(transactions_df)
            print(f"\nStage 4 completed in {time.time() - start:.1f}s")

        total_time = time.time() - total_start
        print("\n" + "=" * 70)
        print(f"PIPELINE COMPLETE - Total time: {total_time:.1f}s")
        print("=" * 70)

        self._print_summary()

    def _should_run_stage(
        self,
        stage: int,
        skip_stages: list,
        only_stage: Optional[int]
    ) -> bool:
        """Check if a stage should be run."""
        if only_stage is not None:
            return stage == only_stage
        return stage not in skip_stages

    def _load_transactions(self) -> pd.DataFrame:
        """Load raw transactions with required columns."""
        usecols = [
            'BASKET_ID', 'CUST_CODE', 'STORE_CODE', 'SHOP_WEEK',
            'PROD_CODE', 'SPEND', 'QUANTITY'
        ]

        # Check for optional columns
        sample = pd.read_csv(self.raw_data_path, nrows=1)
        if 'PROD_CODE_40' in sample.columns:
            usecols.append('PROD_CODE_40')

        return pd.read_csv(self.raw_data_path, usecols=usecols)

    def _run_stage1(self, transactions_df: pd.DataFrame):
        """Run Stage 1: Temporal Metadata Creation."""
        creator = TemporalMetadataCreator(
            train_end_week=self.train_end_week,
            val_end_week=self.val_end_week
        )
        metadata_df = creator.run(transactions_df)
        creator.save(metadata_df, self.metadata_path)

    def _run_stage2(self, transactions_df: pd.DataFrame):
        """Run Stage 2: Customer History Extraction."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Stage 1 output not found: {self.metadata_path}\n"
                "Run Stage 1 first."
            )

        metadata_df = pd.read_parquet(self.metadata_path)
        extractor = CustomerHistoryExtractor()
        history_df = extractor.run(transactions_df, metadata_df)
        extractor.save(history_df, self.histories_path)

    def _run_stage3(self, transactions_df: pd.DataFrame):
        """Run Stage 3: Training Sample Creation."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Stage 1 output not found: {self.metadata_path}"
            )
        if not self.histories_path.exists():
            raise FileNotFoundError(
                f"Stage 2 output not found: {self.histories_path}"
            )

        metadata_df = pd.read_parquet(self.metadata_path)
        history_df = pd.read_parquet(self.histories_path)

        creator = TrainingSampleCreator()
        split_samples = creator.run(history_df, metadata_df, transactions_df)

        print("\nSaving samples...")
        creator.save(split_samples, self.samples_dir)

    def _run_stage4(self, transactions_df: pd.DataFrame):
        """Run Stage 4: Tensor Cache Building."""
        # Optional feature paths
        features_dir = project_root / 'data' / 'features'
        product_features_path = features_dir / 'product_features.parquet'
        customer_segments_path = features_dir / 'customer_segments.parquet'

        builder = TensorCacheBuilder(embedding_dim=128)
        cache = builder.run(
            transactions_df,
            product_features_path if product_features_path.exists() else None,
            customer_segments_path if customer_segments_path.exists() else None
        )

        print("\nSaving tensor cache...")
        builder.save(cache, self.cache_dir)

    def _print_summary(self):
        """Print summary of outputs."""
        print("\nOutput Files:")

        # Check each output
        outputs = [
            ('Temporal Metadata', self.metadata_path),
            ('Customer Histories', self.histories_path),
        ]

        for name, path in outputs:
            if path.exists():
                size = path.stat().st_size / 1e6
                print(f"  ✓ {name}: {path} ({size:.1f} MB)")
            else:
                print(f"  ✗ {name}: {path} (not created)")

        # Samples directory
        if self.samples_dir.exists():
            sample_files = list(self.samples_dir.glob('*.parquet'))
            total_size = sum(f.stat().st_size for f in sample_files) / 1e6
            print(f"  ✓ Training Samples: {len(sample_files)} files ({total_size:.1f} MB)")
        else:
            print(f"  ✗ Training Samples: {self.samples_dir} (not created)")

        # Cache directory
        if self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob('*'))
            total_size = sum(f.stat().st_size for f in cache_files) / 1e6
            print(f"  ✓ Tensor Cache: {len(cache_files)} files ({total_size:.1f} MB)")
        else:
            print(f"  ✗ Tensor Cache: {self.cache_dir} (not created)")


def main():
    parser = argparse.ArgumentParser(
        description='Run RetailSim data preparation pipeline'
    )
    parser.add_argument(
        '--raw-data',
        type=str,
        default=str(project_root / 'raw_data' / 'transactions.csv'),
        help='Path to raw transactions CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(project_root / 'data' / 'prepared'),
        help='Output directory for prepared data'
    )
    parser.add_argument(
        '--skip-stage',
        type=int,
        nargs='+',
        default=[],
        help='Stage(s) to skip (1-4)'
    )
    parser.add_argument(
        '--only-stage',
        type=int,
        default=None,
        help='Only run this stage (1-4)'
    )
    parser.add_argument(
        '--train-end-week',
        type=int,
        default=80,
        help='Last week of training set'
    )
    parser.add_argument(
        '--val-end-week',
        type=int,
        default=95,
        help='Last week of validation set'
    )

    args = parser.parse_args()

    pipeline = DataPreparationPipeline(
        raw_data_path=args.raw_data,
        output_dir=args.output_dir,
        train_end_week=args.train_end_week,
        val_end_week=args.val_end_week
    )

    pipeline.run(
        skip_stages=args.skip_stage,
        only_stage=args.only_stage
    )


if __name__ == '__main__':
    main()
