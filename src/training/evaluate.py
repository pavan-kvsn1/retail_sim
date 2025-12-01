"""
Evaluation Script for World Model.

Computes metrics on validation/test sets:
- Masked Product Prediction: Accuracy, Precision@K, Recall@K, MRR
- Auxiliary Tasks: Accuracy for basket size, price sensitivity, mission type
- Per-bucket analysis for cold-start evaluation
- Day/hour level granularity for temporal analysis

Reference: RetailSim_Data_Pipeline_and_World_Model_Design.md Section 6
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

from model import WorldModel, WorldModelConfig
from dataset import WorldModelDataset, WorldModelDataLoader, EvaluationDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Masked product prediction
    accuracy: float = 0.0
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank

    # Auxiliary tasks
    basket_size_accuracy: float = 0.0
    price_sensitivity_accuracy: float = 0.0
    mission_type_accuracy: float = 0.0
    mission_focus_accuracy: float = 0.0

    # Counts
    n_samples: int = 0
    n_masked_tokens: int = 0


class Evaluator:
    """Evaluator for World Model."""

    def __init__(
        self,
        model: WorldModel,
        device: torch.device,
        k_values: List[int] = [1, 5, 10]
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained WorldModel
            device: Device for computation
            k_values: K values for Precision@K and Recall@K
        """
        self.model = model
        self.device = device
        self.k_values = k_values
        self.model.eval()

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        """Convert WorldModelBatch to tensors on device."""
        return {
            'dense_context': torch.from_numpy(batch.get_dense_context()).float().to(self.device),
            'product_embeddings': torch.from_numpy(batch.product_embeddings).float().to(self.device),
            'price_features': torch.from_numpy(batch.price_features).float().to(self.device),
            'attention_mask': torch.from_numpy(batch.attention_mask).float().to(self.device),
            'product_ids': torch.from_numpy(batch.product_token_ids).long().to(self.device),
            'masked_positions': torch.from_numpy(batch.masked_positions).long().to(self.device) if batch.masked_positions is not None else None,
            'masked_targets': torch.from_numpy(batch.masked_targets).long().to(self.device) if batch.masked_targets is not None else None,
            'auxiliary_labels': {
                k: torch.from_numpy(v).long().to(self.device)
                for k, v in batch.auxiliary_labels.items()
            } if batch.auxiliary_labels else {}
        }

    @torch.no_grad()
    def evaluate_batch(
        self,
        batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate single batch.

        Returns:
            Dict of metric name -> value
        """
        # Forward pass
        masked_logits, aux_logits, _ = self.model(
            dense_context=batch_data['dense_context'],
            product_embeddings=batch_data['product_embeddings'],
            price_features=batch_data['price_features'],
            attention_mask=batch_data['attention_mask'],
            masked_positions=batch_data['masked_positions']
        )

        metrics = {}

        # Masked product prediction metrics
        if batch_data['masked_targets'] is not None and batch_data['masked_positions'] is not None:
            targets = batch_data['masked_targets']  # [B, M]
            valid_mask = targets > 0  # Filter out padding in masked positions

            if valid_mask.sum() > 0:
                # Get predictions
                probs = torch.softmax(masked_logits, dim=-1)  # [B, M, V]

                # Top-K predictions
                top_k_preds = torch.topk(probs, max(self.k_values), dim=-1).indices  # [B, M, K]

                # Accuracy (top-1)
                top1_preds = top_k_preds[:, :, 0]  # [B, M]
                correct = (top1_preds == targets) & valid_mask
                metrics['accuracy'] = correct.sum().item() / valid_mask.sum().item()

                # Precision@K
                for k in self.k_values:
                    top_k = top_k_preds[:, :, :k]  # [B, M, K]
                    hits = (top_k == targets.unsqueeze(-1)).any(dim=-1) & valid_mask
                    metrics[f'precision_at_{k}'] = hits.sum().item() / valid_mask.sum().item()

                # MRR (Mean Reciprocal Rank)
                ranks = (top_k_preds == targets.unsqueeze(-1)).float().argmax(dim=-1) + 1
                # If not in top-K, rank is inf
                not_found = ~(top_k_preds == targets.unsqueeze(-1)).any(dim=-1)
                reciprocal_ranks = torch.where(not_found, torch.zeros_like(ranks.float()), 1.0 / ranks.float())
                reciprocal_ranks = reciprocal_ranks * valid_mask.float()
                metrics['mrr'] = reciprocal_ranks.sum().item() / valid_mask.sum().item()

                metrics['n_masked_tokens'] = valid_mask.sum().item()

        # Auxiliary task metrics
        aux_labels = batch_data['auxiliary_labels']

        for task in ['basket_size', 'price_sensitivity', 'mission_type', 'mission_focus']:
            if task in aux_logits and task in aux_labels:
                preds = aux_logits[task].argmax(dim=-1)
                labels = aux_labels[task]
                valid = labels > 0  # Ignore 0 (unknown)
                if valid.sum() > 0:
                    correct = (preds == labels) & valid
                    metrics[f'{task}_accuracy'] = correct.sum().item() / valid.sum().item()

        metrics['n_samples'] = batch_data['dense_context'].shape[0]

        return metrics

    def evaluate_dataset(
        self,
        dataset: WorldModelDataset,
        batch_size: int = 64,
        max_batches: Optional[int] = None
    ) -> EvaluationMetrics:
        """
        Evaluate on full dataset.

        Args:
            dataset: WorldModelDataset (validation or test)
            batch_size: Batch size for evaluation
            max_batches: Limit number of batches (for quick eval)

        Returns:
            EvaluationMetrics with aggregated results
        """
        dataloader = WorldModelDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            apply_masking=True,
            bucket_batching=False
        )

        # Aggregate metrics
        total_metrics = defaultdict(float)
        n_batches = 0

        for batch in dataloader:
            batch_data = self._prepare_batch(batch)
            batch_metrics = self.evaluate_batch(batch_data)

            for k, v in batch_metrics.items():
                total_metrics[k] += v

            n_batches += 1

            if max_batches and n_batches >= max_batches:
                break

            if n_batches % 100 == 0:
                logger.info(f"Evaluated {n_batches} batches...")

        # Average metrics
        results = EvaluationMetrics()

        if n_batches > 0:
            results.accuracy = total_metrics['accuracy'] / n_batches
            results.precision_at_1 = total_metrics.get('precision_at_1', 0) / n_batches
            results.precision_at_5 = total_metrics.get('precision_at_5', 0) / n_batches
            results.precision_at_10 = total_metrics.get('precision_at_10', 0) / n_batches
            results.mrr = total_metrics['mrr'] / n_batches

            results.basket_size_accuracy = total_metrics.get('basket_size_accuracy', 0) / n_batches
            results.price_sensitivity_accuracy = total_metrics.get('price_sensitivity_accuracy', 0) / n_batches
            results.mission_type_accuracy = total_metrics.get('mission_type_accuracy', 0) / n_batches
            results.mission_focus_accuracy = total_metrics.get('mission_focus_accuracy', 0) / n_batches

            results.n_samples = int(total_metrics['n_samples'])
            results.n_masked_tokens = int(total_metrics.get('n_masked_tokens', 0))

        return results

    def evaluate_by_bucket(
        self,
        dataset: WorldModelDataset,
        batch_size: int = 64
    ) -> Dict[int, EvaluationMetrics]:
        """
        Evaluate per history bucket for cold-start analysis.

        Buckets:
            1: 1-25 weeks history
            2: 26-50 weeks history
            3: 51-75 weeks history
            4: 76-100 weeks history
            5: 101+ weeks history

        Returns:
            Dict of bucket -> EvaluationMetrics
        """
        bucket_metrics = {}

        for bucket, indices in dataset.bucket_indices.items():
            logger.info(f"Evaluating bucket {bucket} ({len(indices):,} samples)...")

            # Create subset dataloader
            bucket_samples = dataset.samples.iloc[indices]

            # Evaluate subset
            total_metrics = defaultdict(float)
            n_batches = 0

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size].tolist()
                batch = dataset.get_batch(batch_indices, apply_masking=True)
                batch_data = self._prepare_batch(batch)

                batch_metrics = self.evaluate_batch(batch_data)

                for k, v in batch_metrics.items():
                    total_metrics[k] += v

                n_batches += 1

            # Average
            results = EvaluationMetrics()
            if n_batches > 0:
                results.accuracy = total_metrics['accuracy'] / n_batches
                results.precision_at_5 = total_metrics.get('precision_at_5', 0) / n_batches
                results.precision_at_10 = total_metrics.get('precision_at_10', 0) / n_batches
                results.mrr = total_metrics['mrr'] / n_batches
                results.n_samples = int(total_metrics['n_samples'])

            bucket_metrics[bucket] = results

        return bucket_metrics

    def evaluate_by_time(
        self,
        dataset: WorldModelDataset,
        group_by: str = 'week',
        batch_size: int = 64
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate grouped by time period.

        Args:
            dataset: WorldModelDataset
            group_by: 'week', 'date', or 'hour'
            batch_size: Batch size

        Returns:
            Dict of time_key -> EvaluationMetrics
        """
        eval_loader = EvaluationDataLoader(
            dataset,
            batch_size=batch_size,
            group_by=group_by
        )

        group_metrics = {}

        for group_key, batch in eval_loader:
            batch_data = self._prepare_batch(batch)
            batch_metrics = self.evaluate_batch(batch_data)

            if group_key not in group_metrics:
                group_metrics[group_key] = defaultdict(float)
                group_metrics[group_key]['n_batches'] = 0

            for k, v in batch_metrics.items():
                group_metrics[group_key][k] += v
            group_metrics[group_key]['n_batches'] += 1

        # Convert to EvaluationMetrics
        results = {}
        for group_key, metrics in group_metrics.items():
            n = metrics['n_batches']
            em = EvaluationMetrics()
            em.accuracy = metrics['accuracy'] / n
            em.precision_at_5 = metrics.get('precision_at_5', 0) / n
            em.mrr = metrics['mrr'] / n
            em.n_samples = int(metrics['n_samples'])
            results[str(group_key)] = em

        return results


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> WorldModel:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config_dict = checkpoint.get('config', {})
    config = WorldModelConfig(
        n_products=config_dict.get('n_products', 5003),
        d_model=config_dict.get('d_model', 512),
        mamba_num_layers=config_dict.get('mamba_layers', 4),
        decoder_num_layers=config_dict.get('decoder_layers', 2)
    )

    model = WorldModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model parameters: {model.num_parameters:,}")

    return model


def run_evaluation(
    checkpoint_path: str,
    project_root: str,
    split: str = 'test',
    output_path: Optional[str] = None,
    batch_size: int = 64,
    max_batches: Optional[int] = None,
    by_bucket: bool = True,
    by_week: bool = True
) -> Dict:
    """
    Run full evaluation and save results.

    Args:
        checkpoint_path: Path to model checkpoint
        project_root: Project root directory
        split: 'validation' or 'test'
        output_path: Path to save results JSON
        batch_size: Evaluation batch size
        max_batches: Limit batches (for quick eval)
        by_bucket: Include per-bucket analysis
        by_week: Include per-week analysis

    Returns:
        Dict of all evaluation results
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    project_root = Path(project_root)

    logger.info(f"Running evaluation on {device}")

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Load dataset
    logger.info(f"Loading {split} dataset...")
    dataset = WorldModelDataset(
        project_root,
        split=split,
        max_seq_len=50,
        load_transactions=True
    )
    logger.info(f"Dataset size: {len(dataset):,}")

    # Create evaluator
    evaluator = Evaluator(model, device)

    results = {
        'checkpoint': checkpoint_path,
        'split': split,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'n_samples': len(dataset)
    }

    # Overall evaluation
    logger.info("Running overall evaluation...")
    overall_metrics = evaluator.evaluate_dataset(dataset, batch_size, max_batches)
    results['overall'] = asdict(overall_metrics)

    logger.info(f"\nOverall Results:")
    logger.info(f"  Accuracy: {overall_metrics.accuracy:.4f}")
    logger.info(f"  Precision@5: {overall_metrics.precision_at_5:.4f}")
    logger.info(f"  Precision@10: {overall_metrics.precision_at_10:.4f}")
    logger.info(f"  MRR: {overall_metrics.mrr:.4f}")
    logger.info(f"  Basket Size Acc: {overall_metrics.basket_size_accuracy:.4f}")
    logger.info(f"  Price Sens Acc: {overall_metrics.price_sensitivity_accuracy:.4f}")

    # Per-bucket evaluation
    if by_bucket:
        logger.info("\nRunning per-bucket evaluation...")
        bucket_metrics = evaluator.evaluate_by_bucket(dataset, batch_size)
        results['by_bucket'] = {
            str(k): asdict(v) for k, v in bucket_metrics.items()
        }

        logger.info("\nPer-Bucket Results:")
        for bucket, metrics in sorted(bucket_metrics.items()):
            logger.info(f"  Bucket {bucket}: Acc={metrics.accuracy:.4f}, P@5={metrics.precision_at_5:.4f}")

    # Per-week evaluation
    if by_week:
        logger.info("\nRunning per-week evaluation...")
        week_metrics = evaluator.evaluate_by_time(dataset, 'week', batch_size)
        results['by_week'] = {
            str(k): asdict(v) for k, v in week_metrics.items()
        }

        # Summary stats
        accuracies = [m.accuracy for m in week_metrics.values()]
        logger.info(f"\nPer-Week Summary:")
        logger.info(f"  Min accuracy: {min(accuracies):.4f}")
        logger.info(f"  Max accuracy: {max(accuracies):.4f}")
        logger.info(f"  Mean accuracy: {np.mean(accuracies):.4f}")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

    return results


def main():
    """Main evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate World Model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--project-root', type=str,
                        default='/Users/hazymoji/Documents/DataDev/ML Projects/retail_sim')
    parser.add_argument('--split', type=str, default='test',
                        choices=['validation', 'test'])
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Limit batches for quick evaluation')
    parser.add_argument('--no-bucket', action='store_true',
                        help='Skip per-bucket evaluation')
    parser.add_argument('--no-week', action='store_true',
                        help='Skip per-week evaluation')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = f'results/eval_{args.split}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    run_evaluation(
        checkpoint_path=args.checkpoint,
        project_root=args.project_root,
        split=args.split,
        output_path=args.output,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        by_bucket=not args.no_bucket,
        by_week=not args.no_week
    )


if __name__ == '__main__':
    main()
