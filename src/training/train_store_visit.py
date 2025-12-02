"""
Training Script for Store Visit Prediction.

Stage 1 of the two-stage world model:
1. Store Visit Prediction (this script) - WHERE will customer shop?
2. Next Basket Prediction - WHAT will they buy at that store?

Usage:
    python -m src.training.train_store_visit --epochs 10 --batch-size 128

For multi-GPU (RunPod):
    python -m src.training.train_store_visit --epochs 10 --batch-size 256
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional
import json

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.training.model_store_visit import StoreVisitPredictor, StoreVisitModelConfig
from src.training.dataset_store_visit import StoreVisitDataset, StoreVisitDataLoader
from src.training.losses_store_visit import StoreVisitLoss, FocalLoss, StoreVisitMetrics, compute_store_class_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StoreVisitTrainer:
    """Trainer for Store Visit Prediction model."""

    def __init__(
        self,
        project_root: Path,
        config: StoreVisitModelConfig,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
        use_class_weights: bool = True,
        class_weight_smoothing: float = 0.3,
        include_basket: bool = True,
        device: str = 'auto',
        loss_type: str = 'cross_entropy',
        focal_gamma: float = 2.0,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.01,
        eval_steps: int = 0,
    ):
        self.project_root = Path(project_root)
        self.config = config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.use_class_weights = use_class_weights
        self.class_weight_smoothing = class_weight_smoothing
        self.include_basket = include_basket
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.eval_steps = eval_steps  # 0 = only at epoch end

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

        # Load datasets
        self._load_datasets()

        # Initialize model
        self._init_model()

        # Initialize loss and optimizer
        self._init_training()

        # Tracking
        self.global_step = 0
        self.best_val_accuracy = 0.0

    def _load_datasets(self):
        """Load train and validation datasets."""
        logger.info("Loading datasets...")

        self.train_dataset = StoreVisitDataset(
            self.project_root,
            split='train',
            include_basket=self.include_basket,
        )

        # Share vocabulary with validation set
        vocabulary = self.train_dataset.get_vocabulary()

        self.val_dataset = StoreVisitDataset(
            self.project_root,
            split='validation',
            include_basket=self.include_basket,
            vocabulary=vocabulary,
        )

        # Update config with actual num_stores
        self.config.num_stores = self.train_dataset.num_stores
        logger.info(f"Number of stores: {self.config.num_stores}")

        # Create data loaders
        self.train_loader = StoreVisitDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.val_loader = StoreVisitDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        logger.info(f"Train: {len(self.train_dataset):,} samples, {len(self.train_loader)} batches")
        logger.info(f"Val: {len(self.val_dataset):,} samples, {len(self.val_loader)} batches")

    def _init_model(self):
        """Initialize the model."""
        self.model = StoreVisitPredictor(self.config)
        self.model.to(self.device)

        # Multi-GPU support
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

    def _init_training(self):
        """Initialize loss, optimizer, and scheduler."""
        # Compute class weights if enabled
        class_weights = None
        if self.use_class_weights:
            store_counts = self.train_dataset.samples['target_store_id'].value_counts().to_dict()
            # Map store_id to idx
            store_idx_counts = {}
            for store_id, count in store_counts.items():
                idx = self.train_dataset.store_to_idx.get(store_id, 0)
                store_idx_counts[idx] = count

            class_weights = compute_store_class_weights(
                store_idx_counts,
                self.config.num_stores,
                smoothing=self.class_weight_smoothing,
            ).to(self.device)
            logger.info(f"Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")

        # Loss function
        if self.loss_type == 'focal':
            self.loss_fn = FocalLoss(
                gamma=self.focal_gamma,
                alpha=class_weights,
                label_smoothing=self.label_smoothing,
            ).to(self.device)
            logger.info(f"Using Focal Loss (gamma={self.focal_gamma})")
        else:
            self.loss_fn = StoreVisitLoss(
                num_stores=self.config.num_stores,
                class_weights=class_weights,
                label_smoothing=self.label_smoothing,
            ).to(self.device)
            logger.info("Using Cross-Entropy Loss")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler (will be set per epoch)
        self.scheduler = None

    def _to_device(self, batch):
        """Move batch tensors to device."""
        customer_context = torch.from_numpy(batch.customer_context).to(self.device)
        temporal_context = torch.from_numpy(batch.temporal_context).to(self.device)
        previous_store_idx = torch.from_numpy(batch.previous_store_idx).to(self.device)
        target_store_idx = torch.from_numpy(batch.target_store_idx).to(self.device)

        # Optional basket
        if batch.previous_basket_embeddings is not None:
            basket_embeddings = torch.from_numpy(batch.previous_basket_embeddings).to(self.device)
            basket_mask = torch.from_numpy(batch.previous_basket_mask).to(self.device)
        else:
            basket_embeddings = None
            basket_mask = None

        return {
            'customer_context': customer_context,
            'temporal_context': temporal_context,
            'previous_store_idx': previous_store_idx,
            'previous_basket_embeddings': basket_embeddings,
            'previous_basket_mask': basket_mask,
            'target_store_idx': target_store_idx,
        }

    def train_epoch(self, epoch: int, checkpoint_dir: Optional[Path] = None) -> Dict[str, float]:
        """Train for one epoch with optional mid-epoch validation."""
        self.model.train()
        self.train_loader.reset()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        start_time = time.time()
        log_interval = 100
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            self.global_step += 1

            # Move to device
            inputs = self._to_device(batch)
            targets = inputs.pop('target_store_idx')

            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs['store_logits']

            # Compute loss
            loss = self.loss_fn(logits, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item() * batch.batch_size
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch.batch_size

            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate

            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed
                avg_loss = total_loss / total_samples
                accuracy = total_correct / total_samples

                logger.info(
                    f"Epoch {epoch} | Step {batch_idx + 1}/{num_batches} | "
                    f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
                    f"LR: {current_lr:.2e} | Speed: {samples_per_sec:.0f} samples/sec"
                )

            # Mid-epoch validation (if eval_steps > 0)
            if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                val_metrics = self.validate()
                logger.info(
                    f">>> Mid-epoch validation (step {self.global_step}) | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Top-5: {val_metrics['top_5_accuracy']:.4f} | "
                    f"MRR: {val_metrics['mrr']:.4f}"
                )

                # Save best model during mid-epoch validation
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    if checkpoint_dir:
                        best_path = checkpoint_dir / 'best_model.pt'
                        self.save_checkpoint(best_path, epoch, {
                            'train_loss': total_loss / total_samples,
                            'train_accuracy': total_correct / total_samples,
                            **val_metrics
                        })
                        logger.info(f"New best accuracy: {self.best_val_accuracy:.4f}")

                self.model.train()  # Back to training mode

        # Epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.val_loader.reset()

        all_logits = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0

        for batch in self.val_loader:
            inputs = self._to_device(batch)
            targets = inputs.pop('target_store_idx')

            outputs = self.model(**inputs)
            logits = outputs['store_logits']

            # Compute loss on device (where loss_fn buffers live)
            loss = self.loss_fn(logits, targets)
            total_loss += loss.item() * batch.batch_size
            total_samples += batch.batch_size

            # Move to CPU for metric computation
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        # Concatenate all (on CPU)
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute all metrics (on CPU)
        metrics = StoreVisitMetrics.compute_all(all_logits, all_targets)
        metrics['val_loss'] = total_loss / total_samples

        return metrics

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        # Handle DataParallel wrapper
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'metrics': metrics,
            'vocabulary': self.train_dataset.get_vocabulary(),
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Handle DataParallel wrapper
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def _create_scheduler(self, num_epochs: int):
        """Create warmup + cosine annealing scheduler."""
        total_steps = num_epochs * len(self.train_loader)
        warmup_steps = int(total_steps * self.warmup_ratio)
        min_lr = self.learning_rate * self.min_lr_ratio

        def lr_lambda(current_step: int) -> float:
            """Linear warmup then cosine annealing."""
            if current_step < warmup_steps:
                # Linear warmup: 0 -> 1
                return float(current_step) / float(max(1, warmup_steps))

            # Cosine annealing: 1 -> min_lr_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

        scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"Scheduler: {warmup_steps:,} warmup steps, "
                    f"{total_steps - warmup_steps:,} cosine decay steps, "
                    f"min_lr={min_lr:.2e}")
        return scheduler

    def train(
        self,
        num_epochs: int,
        eval_every: int = 1,
        save_every: int = 1,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Main training loop."""
        if checkpoint_dir is None:
            checkpoint_dir = self.project_root / 'models' / 'store_visit'
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup scheduler with warmup + cosine annealing
        total_steps = num_epochs * len(self.train_loader)
        self.scheduler = self._create_scheduler(num_epochs)

        logger.info(f"Starting training for {num_epochs} epochs ({total_steps:,} steps)")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train (pass checkpoint_dir for mid-epoch best model saving)
            train_metrics = self.train_epoch(epoch, checkpoint_dir=checkpoint_dir)

            # Validate
            if epoch % eval_every == 0:
                val_metrics = self.validate()
                logger.info(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Top-3: {val_metrics['top_3_accuracy']:.4f} | "
                    f"Top-5: {val_metrics['top_5_accuracy']:.4f} | "
                    f"MRR: {val_metrics['mrr']:.4f}"
                )

                # Save best model
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    best_path = checkpoint_dir / 'best_model.pt'
                    self.save_checkpoint(best_path, epoch, {**train_metrics, **val_metrics})
                    logger.info(f"New best accuracy: {self.best_val_accuracy:.4f}")

            # Save periodic checkpoint
            if epoch % save_every == 0:
                ckpt_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                self.save_checkpoint(ckpt_path, epoch, train_metrics)

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        # Save final model
        final_path = checkpoint_dir / 'final_model.pt'
        self.save_checkpoint(final_path, num_epochs, train_metrics)

        logger.info(f"Training complete. Best validation accuracy: {self.best_val_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Store Visit Prediction Model')

    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Fraction of total steps for linear warmup')
    parser.add_argument('--min-lr-ratio', type=float, default=0.01,
                        help='Minimum LR as fraction of max LR for cosine annealing')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of MLP layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--no-basket', action='store_true', help='Disable basket summary')

    # Loss
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal'],
                        help='Loss function: cross_entropy or focal')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma (focusing parameter). Higher = more focus on hard examples')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable class weights')
    parser.add_argument('--class-weight-smoothing', type=float, default=0.3, help='Class weight smoothing')

    # Checkpointing & Evaluation
    parser.add_argument('--eval-every', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--eval-steps', type=int, default=0,
                        help='Validate every N steps (0 = only at epoch end). '
                             'Set to half of steps/epoch for mid-epoch validation.')
    parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')

    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/mps/cpu)')

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Project root
    project_root = Path(__file__).parent.parent.parent

    # Model config
    config = StoreVisitModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_basket_summary=not args.no_basket,
    )

    # Create trainer
    trainer = StoreVisitTrainer(
        project_root=project_root,
        config=config,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_class_weights=not args.no_class_weights,
        class_weight_smoothing=args.class_weight_smoothing,
        include_basket=not args.no_basket,
        device=args.device,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        eval_steps=args.eval_steps,
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(Path(args.resume))

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    # Train
    trainer.train(
        num_epochs=args.epochs,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == '__main__':
    main()
