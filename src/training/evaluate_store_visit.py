"""
Evaluation Script for Store Visit Prediction Model.

Evaluates a trained model on test/validation data and produces:
- Overall metrics (accuracy, top-k accuracy, MRR)
- Per-store breakdown
- Confusion analysis
- Customer segment analysis

Usage:
    python -m src.training.evaluate_store_visit --checkpoint models/store_visit/best_model.pt
    python -m src.training.evaluate_store_visit --checkpoint models/store_visit/best_model.pt --split test
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.training.model_store_visit import StoreVisitPredictor, StoreVisitModelConfig
from src.training.dataset_store_visit import StoreVisitDataset, StoreVisitDataLoader
from src.training.losses_store_visit import StoreVisitLoss, StoreVisitMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StoreVisitEvaluator:
    """Evaluator for Store Visit Prediction model."""

    def __init__(
        self,
        checkpoint_path: Path,
        project_root: Path,
        split: str = 'validation',
        batch_size: int = 128,
        device: str = 'auto',
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.project_root = Path(project_root)
        self.split = split
        self.batch_size = batch_size

        # Device setup
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load checkpoint and model
        self._load_model()

        # Load dataset
        self._load_dataset()

    def _load_model(self):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Reconstruct config
        config_dict = checkpoint['config']
        self.config = StoreVisitModelConfig(**config_dict)

        # Create and load model
        self.model = StoreVisitPredictor(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Store vocabulary from checkpoint
        self.vocabulary = checkpoint.get('vocabulary', {})

        logger.info(f"Loaded model with {self.config.num_stores} stores")
        logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

    def _load_dataset(self):
        """Load evaluation dataset."""
        logger.info(f"Loading {self.split} dataset...")

        self.dataset = StoreVisitDataset(
            self.project_root,
            split=self.split,
            include_basket=self.config.use_basket_summary,
            vocabulary=self.vocabulary,
        )

        self.loader = StoreVisitDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        logger.info(f"Loaded {len(self.dataset):,} samples")

    def _to_device(self, batch) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        inputs = {
            'customer_context': torch.from_numpy(batch.customer_context).to(self.device),
            'temporal_context': torch.from_numpy(batch.temporal_context).to(self.device),
            'previous_store_idx': torch.from_numpy(batch.previous_store_idx).to(self.device),
        }

        if batch.previous_basket_embeddings is not None:
            inputs['previous_basket_embeddings'] = torch.from_numpy(
                batch.previous_basket_embeddings
            ).to(self.device)
            inputs['previous_basket_mask'] = torch.from_numpy(
                batch.previous_basket_mask
            ).to(self.device)

        targets = torch.from_numpy(batch.target_store_idx).to(self.device)

        return inputs, targets

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run full evaluation."""
        logger.info("Running evaluation...")

        all_logits = []
        all_targets = []
        all_predictions = []
        all_previous_stores = []

        for batch in tqdm(self.loader, desc="Evaluating"):
            inputs, targets = self._to_device(batch)

            outputs = self.model(**inputs)
            logits = outputs['store_logits']
            preds = logits.argmax(dim=-1)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_predictions.append(preds.cpu())
            all_previous_stores.append(torch.from_numpy(batch.previous_store_idx))

        # Concatenate
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_previous_stores = torch.cat(all_previous_stores, dim=0)

        # Compute overall metrics
        overall_metrics = StoreVisitMetrics.compute_all(all_logits, all_targets)

        # Compute loss
        loss_fn = StoreVisitLoss(self.config.num_stores, label_smoothing=0.0)
        overall_metrics['loss'] = loss_fn(all_logits, all_targets).item()

        # Per-store metrics
        per_store_metrics = self._compute_per_store_metrics(
            all_predictions.numpy(),
            all_targets.numpy(),
        )

        # Store transition analysis
        transition_metrics = self._compute_transition_metrics(
            all_predictions.numpy(),
            all_targets.numpy(),
            all_previous_stores.numpy(),
        )

        # Confusion analysis
        confusion_analysis = self._compute_confusion_analysis(
            all_predictions.numpy(),
            all_targets.numpy(),
            top_k=10,
        )

        return {
            'overall': overall_metrics,
            'per_store': per_store_metrics,
            'transitions': transition_metrics,
            'confusion': confusion_analysis,
            'num_samples': len(all_targets),
            'split': self.split,
        }

    def _compute_per_store_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict:
        """Compute per-store accuracy and counts."""
        store_correct = defaultdict(int)
        store_total = defaultdict(int)
        store_predicted = defaultdict(int)

        for pred, target in zip(predictions, targets):
            store_total[int(target)] += 1
            store_predicted[int(pred)] += 1
            if pred == target:
                store_correct[int(target)] += 1

        # Compute per-store accuracy
        per_store_acc = {}
        for store_idx in store_total:
            total = store_total[store_idx]
            correct = store_correct[store_idx]
            per_store_acc[store_idx] = {
                'accuracy': correct / total if total > 0 else 0.0,
                'total_samples': total,
                'correct': correct,
                'predicted_count': store_predicted.get(store_idx, 0),
            }

        # Summary stats
        accuracies = [v['accuracy'] for v in per_store_acc.values()]
        totals = [v['total_samples'] for v in per_store_acc.values()]

        return {
            'per_store': per_store_acc,
            'mean_per_store_accuracy': np.mean(accuracies),
            'median_per_store_accuracy': np.median(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'stores_with_zero_accuracy': sum(1 for a in accuracies if a == 0),
            'stores_evaluated': len(per_store_acc),
        }

    def _compute_transition_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        previous_stores: np.ndarray,
    ) -> Dict:
        """Analyze store transition patterns."""
        # Same store vs different store
        same_store_mask = previous_stores == targets
        diff_store_mask = ~same_store_mask

        same_store_correct = (predictions[same_store_mask] == targets[same_store_mask]).sum()
        same_store_total = same_store_mask.sum()

        diff_store_correct = (predictions[diff_store_mask] == targets[diff_store_mask]).sum()
        diff_store_total = diff_store_mask.sum()

        return {
            'same_store': {
                'accuracy': same_store_correct / same_store_total if same_store_total > 0 else 0,
                'total': int(same_store_total),
                'correct': int(same_store_correct),
                'percentage_of_data': same_store_total / len(targets) * 100,
            },
            'different_store': {
                'accuracy': diff_store_correct / diff_store_total if diff_store_total > 0 else 0,
                'total': int(diff_store_total),
                'correct': int(diff_store_correct),
                'percentage_of_data': diff_store_total / len(targets) * 100,
            },
        }

    def _compute_confusion_analysis(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        top_k: int = 10,
    ) -> Dict:
        """Analyze most common confusions."""
        # Count confusion pairs (predicted, actual)
        confusion_counts = defaultdict(int)
        for pred, target in zip(predictions, targets):
            if pred != target:
                confusion_counts[(int(pred), int(target))] += 1

        # Get top confusions
        sorted_confusions = sorted(
            confusion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Map to store IDs if vocabulary available
        idx_to_store = self.vocabulary.get('idx_to_store', {})

        top_confusions = []
        for (pred_idx, target_idx), count in sorted_confusions:
            pred_store = idx_to_store.get(str(pred_idx), pred_idx)
            target_store = idx_to_store.get(str(target_idx), target_idx)
            top_confusions.append({
                'predicted': pred_store,
                'actual': target_store,
                'predicted_idx': pred_idx,
                'actual_idx': target_idx,
                'count': count,
            })

        return {
            'top_confusions': top_confusions,
            'total_errors': int((predictions != targets).sum()),
            'error_rate': float((predictions != targets).mean()),
        }

    def print_results(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "=" * 60)
        print(f"STORE VISIT PREDICTION EVALUATION ({results['split']})")
        print("=" * 60)

        print(f"\nSamples evaluated: {results['num_samples']:,}")

        print("\n--- Overall Metrics ---")
        overall = results['overall']
        print(f"  Loss:           {overall['loss']:.4f}")
        print(f"  Accuracy:       {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
        print(f"  Top-3 Accuracy: {overall['top_3_accuracy']:.4f} ({overall['top_3_accuracy']*100:.2f}%)")
        print(f"  Top-5 Accuracy: {overall['top_5_accuracy']:.4f} ({overall['top_5_accuracy']*100:.2f}%)")
        print(f"  Top-10 Accuracy:{overall['top_10_accuracy']:.4f} ({overall['top_10_accuracy']*100:.2f}%)")
        print(f"  MRR:            {overall['mrr']:.4f}")

        print("\n--- Store Transition Analysis ---")
        trans = results['transitions']
        print(f"  Same store visits:      {trans['same_store']['percentage_of_data']:.1f}% of data")
        print(f"    Accuracy:             {trans['same_store']['accuracy']:.4f}")
        print(f"  Different store visits: {trans['different_store']['percentage_of_data']:.1f}% of data")
        print(f"    Accuracy:             {trans['different_store']['accuracy']:.4f}")

        print("\n--- Per-Store Summary ---")
        per_store = results['per_store']
        print(f"  Stores evaluated:       {per_store['stores_evaluated']}")
        print(f"  Mean per-store acc:     {per_store['mean_per_store_accuracy']:.4f}")
        print(f"  Median per-store acc:   {per_store['median_per_store_accuracy']:.4f}")
        print(f"  Min accuracy:           {per_store['min_accuracy']:.4f}")
        print(f"  Max accuracy:           {per_store['max_accuracy']:.4f}")
        print(f"  Stores with 0% acc:     {per_store['stores_with_zero_accuracy']}")

        print("\n--- Top Confusions ---")
        for conf in results['confusion']['top_confusions'][:5]:
            print(f"  Predicted {conf['predicted']} instead of {conf['actual']}: {conf['count']:,} times")

        print("\n" + "=" * 60)

    def save_results(self, results: Dict, output_path: Path):
        """Save results to JSON."""
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        results_clean = convert(results)

        with open(output_path, 'w') as f:
            json.dump(results_clean, f, indent=2)

        logger.info(f"Results saved to {output_path}")


@torch.no_grad()
def analyze_predictions(
    evaluator: StoreVisitEvaluator,
    num_samples: int = 10,
) -> List[Dict]:
    """Analyze individual predictions for debugging."""
    examples = []

    for batch in evaluator.loader:
        if len(examples) >= num_samples:
            break

        inputs, targets = evaluator._to_device(batch)
        outputs = evaluator.model(**inputs)

        probs = outputs['store_probs']
        top5_probs, top5_indices = probs.topk(5, dim=-1)

        for i in range(min(batch.batch_size, num_samples - len(examples))):
            idx_to_store = evaluator.vocabulary.get('idx_to_store', {})

            target_idx = targets[i].item()
            prev_idx = batch.previous_store_idx[i]

            example = {
                'customer_id': batch.customer_ids[i] if batch.customer_ids is not None else None,
                'previous_store': idx_to_store.get(str(prev_idx), prev_idx),
                'target_store': idx_to_store.get(str(target_idx), target_idx),
                'top5_predictions': [
                    {
                        'store': idx_to_store.get(str(top5_indices[i, j].item()), top5_indices[i, j].item()),
                        'probability': top5_probs[i, j].item(),
                    }
                    for j in range(5)
                ],
                'correct': top5_indices[i, 0].item() == target_idx,
                'target_in_top5': target_idx in top5_indices[i].tolist(),
            }
            examples.append(example)

    return examples


def main():
    parser = argparse.ArgumentParser(description='Evaluate Store Visit Prediction Model')

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--examples', type=int, default=0, help='Number of example predictions to show')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    # Create evaluator
    evaluator = StoreVisitEvaluator(
        checkpoint_path=Path(args.checkpoint),
        project_root=project_root,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Print results
    evaluator.print_results(results)

    # Save if output specified
    if args.output:
        evaluator.save_results(results, Path(args.output))
    else:
        # Default output path
        output_path = Path(args.checkpoint).parent / f'eval_{args.split}.json'
        evaluator.save_results(results, output_path)

    # Show example predictions
    if args.examples > 0:
        print(f"\n--- Example Predictions ({args.examples}) ---")
        examples = analyze_predictions(evaluator, args.examples)
        for i, ex in enumerate(examples):
            print(f"\nExample {i+1}:")
            print(f"  Previous store: {ex['previous_store']}")
            print(f"  Target store:   {ex['target_store']}")
            print(f"  Correct:        {'Yes' if ex['correct'] else 'No'}")
            print(f"  Top 5 predictions:")
            for pred in ex['top5_predictions']:
                marker = " <--" if pred['store'] == ex['target_store'] else ""
                print(f"    {pred['store']}: {pred['probability']:.4f}{marker}")


if __name__ == '__main__':
    main()
