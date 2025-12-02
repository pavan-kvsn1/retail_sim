"""
Training Script for Next-Basket Prediction World Model.

Usage:
    # First, generate next-basket samples
    python src/data_preparation/stage4_next_basket_samples.py

    # Then train
    python src/training/train_next_basket.py --epochs 20 --batch-size 64

    # With MPS (Apple Silicon)
    python src/training/train_next_basket.py --device mps --batch-size 64

Key differences from masked prediction:
- Multi-label BCE loss instead of masked token cross-entropy
- Metrics: Precision@k, Recall@k, F1@k instead of accuracy
- No masking - full input basket visible
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, LambdaLR
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import logging
import argparse
import time
import json

# Support both direct execution and package imports
try:
    # When run as script
    from dataset_next_basket import NextBasketDataset, NextBasketDataLoader
    from model_next_basket import NextBasketWorldModel, NextBasketModelConfig
    from losses_next_basket import NextBasketLoss, NextBasketLossConfig, NextBasketMetrics
except ImportError:
    # When imported as package (for tests)
    from .dataset_next_basket import NextBasketDataset, NextBasketDataLoader
    from .model_next_basket import NextBasketWorldModel, NextBasketModelConfig
    from .losses_next_basket import NextBasketLoss, NextBasketLossConfig, NextBasketMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    batch_size: int = 64
    num_workers: int = 0  # DataLoader workers

    # Training
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Learning rate schedule
    scheduler_type: str = 'cosine'  # 'onecycle', 'cosine', or 'cosine_restarts'
    min_lr_ratio: float = 0.01  # Minimum LR as fraction of max LR
    cosine_t0: int = 5  # Epochs per restart cycle (for cosine_restarts)

    # Model
    hidden_dim: int = 512
    encoder_layers: int = 4
    decoder_layers: int = 2
    dropout: float = 0.1

    # Evaluation
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 100

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / 'checkpoints')
    save_every_n_epochs: int = 1
    save_every_n_steps: int = 10000  # Also save checkpoint every N steps

    # Device
    device: str = 'auto'
    gradient_checkpointing: bool = False


class NextBasketTrainer:
    """Trainer for next-basket prediction model."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.project_root = Path(config.project_root)

        # Setup device
        if config.device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(config.device)

        # Load datasets
        self._load_datasets()

        # Create model
        self._create_model()

        # Setup training
        self._setup_training()

        # Tracking
        self.global_step = 0
        self.best_f1 = 0.0
        self.last_val_metrics = {}

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _load_datasets(self):
        """Load train and validation datasets."""
        logger.info("Loading training dataset...")
        self.train_dataset = NextBasketDataset(
            self.project_root,
            split='train',
        )

        # Get vocabulary from training set to share with validation
        train_vocabulary = self.train_dataset.get_vocabulary()

        logger.info("Loading validation dataset...")
        self.val_dataset = NextBasketDataset(
            self.project_root,
            split='validation',
            vocabulary=train_vocabulary,  # Use same vocabulary as training
        )

        # Create dataloaders
        self.train_loader = NextBasketDataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self.val_loader = NextBasketDataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        logger.info(f"Train samples: {len(self.train_dataset):,}")
        logger.info(f"Validation samples: {len(self.val_dataset):,}")
        logger.info(f"Vocabulary size: {self.train_dataset.vocab_size}")

    def _create_model(self):
        """Create model."""
        model_config = NextBasketModelConfig(
            vocab_size=self.train_dataset.vocab_size,
            hidden_dim=self.config.hidden_dim,
            encoder_layers=self.config.encoder_layers,
            decoder_layers=self.config.decoder_layers,
            dropout=self.config.dropout,
        )

        self.model = NextBasketWorldModel(model_config)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        self.model.to(self.device)

    def _setup_training(self):
        """Setup optimizer, scheduler, loss."""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == 'onecycle':
            # OneCycleLR: warmup + annealing built-in
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                total_steps=total_steps,
                pct_start=self.config.warmup_ratio,
            )
            logger.info(f"Using OneCycleLR scheduler: {total_steps:,} total steps")

        elif self.config.scheduler_type == 'cosine_restarts':
            # Cosine Annealing with Warm Restarts (good for finetuning)
            # T_0 is in epochs, T_mult doubles the cycle each time
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.cosine_t0 * len(self.train_loader),  # Convert epochs to steps
                T_mult=2,  # Double cycle length each restart
                eta_min=self.config.lr * self.config.min_lr_ratio,
            )
            logger.info(f"Using CosineAnnealingWarmRestarts: T_0={self.config.cosine_t0} epochs, "
                        f"min_lr={self.config.lr * self.config.min_lr_ratio:.2e}")

        else:  # Default: 'cosine' - linear warmup + cosine decay
            min_lr = self.config.lr * self.config.min_lr_ratio

            def lr_lambda(current_step: int) -> float:
                """Linear warmup then cosine annealing."""
                if current_step < warmup_steps:
                    # Linear warmup: 0 -> 1
                    return float(current_step) / float(max(1, warmup_steps))

                # Cosine annealing: 1 -> min_lr_ratio
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine_decay

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            logger.info(f"Using Cosine scheduler: {warmup_steps:,} warmup steps, "
                        f"{total_steps - warmup_steps:,} decay steps, min_lr={min_lr:.2e}")

        # Loss
        self.loss_fn = NextBasketLoss()

        # Checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """Run training loop."""
        logger.info("Starting training...")

        # Initial validation
        logger.info("Running initial validation...")
        val_metrics = self.validate()
        self._log_validation(val_metrics)

        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.epochs}")

            self._train_epoch(epoch)

            # Save checkpoint
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

        logger.info("\nTraining complete!")
        logger.info(f"Best F1@10: {self.best_f1:.4f}")

    def _train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        self.train_loader.reset()

        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        batch_times = []

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            # Move to device
            input_embeddings = torch.tensor(batch.input_embeddings, device=self.device)
            input_price_features = torch.tensor(batch.input_price_features, device=self.device)
            input_attention_mask = torch.tensor(batch.input_attention_mask, device=self.device)
            customer_context = torch.tensor(batch.customer_context, device=self.device)
            temporal_context = torch.tensor(batch.temporal_context, device=self.device)
            store_context = torch.tensor(batch.store_context, device=self.device)
            trip_context = torch.tensor(batch.trip_context, device=self.device)
            targets = torch.tensor(batch.target_products, device=self.device)

            auxiliary_labels = {
                k: torch.tensor(v, device=self.device, dtype=torch.long)
                for k, v in batch.auxiliary_labels.items()
            }

            # Forward
            outputs = self.model(
                input_embeddings,
                input_price_features,
                input_attention_mask,
                customer_context,
                temporal_context,
                store_context,
                trip_context,
            )

            # Loss
            loss, loss_dict = self.loss_fn(outputs, targets, auxiliary_labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Track
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Log
            if self.global_step % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                samples_per_sec = self.config.batch_size / np.mean(batch_times[-100:])

                logger.info(
                    f"Step {self.global_step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Product: {loss_dict['product'].item():.4f} | "
                    f"Speed: {samples_per_sec:.1f} samples/s"
                )

            # Validate
            if self.global_step % self.config.eval_every_n_steps == 0:
                val_metrics = self.validate()
                self._log_validation(val_metrics)
                self.model.train()

            # Step-based checkpointing
            if self.config.save_every_n_steps > 0 and self.global_step % self.config.save_every_n_steps == 0:
                self._save_checkpoint(f'step_{self.global_step}')

        # End of epoch
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        for batch in self.val_loader:
            # Move to device
            input_embeddings = torch.tensor(batch.input_embeddings, device=self.device)
            input_price_features = torch.tensor(batch.input_price_features, device=self.device)
            input_attention_mask = torch.tensor(batch.input_attention_mask, device=self.device)
            customer_context = torch.tensor(batch.customer_context, device=self.device)
            temporal_context = torch.tensor(batch.temporal_context, device=self.device)
            store_context = torch.tensor(batch.store_context, device=self.device)
            trip_context = torch.tensor(batch.trip_context, device=self.device)
            targets = torch.tensor(batch.target_products, device=self.device)

            auxiliary_labels = {
                k: torch.tensor(v, device=self.device, dtype=torch.long)
                for k, v in batch.auxiliary_labels.items()
            }

            # Forward
            outputs = self.model(
                input_embeddings,
                input_price_features,
                input_attention_mask,
                customer_context,
                temporal_context,
                store_context,
                trip_context,
            )

            # Loss
            loss, _ = self.loss_fn(outputs, targets, auxiliary_labels)
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions for metrics
            probs = torch.sigmoid(outputs['product_logits'])
            all_predictions.append(probs.cpu())
            all_targets.append(targets.cpu())

        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = NextBasketMetrics.compute_all(all_predictions, all_targets)
        metrics['loss'] = total_loss / num_batches

        self.last_val_metrics = metrics
        return metrics

    def _log_validation(self, metrics: Dict[str, float]):
        """Log validation metrics."""
        f1_10 = metrics.get('f1@10', 0)

        logger.info(
            f">>> Validation | "
            f"Loss: {metrics['loss']:.4f} | "
            f"P@10: {metrics['precision@10']:.4f} | "
            f"R@10: {metrics['recall@10']:.4f} | "
            f"F1@10: {f1_10:.4f} | "
            f"HR@10: {metrics['hit_rate@10']:.4f}"
        )

        # Save best model
        if f1_10 > self.best_f1:
            self.best_f1 = f1_10
            self._save_checkpoint('best')
            logger.info(f"New best F1@10: {f1_10:.4f}")

    def _save_checkpoint(self, identifier):
        """Save model checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / f'next_basket_{identifier}.pt'

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_f1': self.best_f1,
            'config': {
                'vocab_size': self.train_dataset.vocab_size,
                'hidden_dim': self.config.hidden_dim,
                'encoder_layers': self.config.encoder_layers,
                'decoder_layers': self.config.decoder_layers,
            },
            'metrics': self.last_val_metrics,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Next-Basket World Model')

    # Data
    parser.add_argument('--project-root', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'onecycle', 'cosine_restarts'],
                       help='LR scheduler: cosine (warmup+decay), onecycle, cosine_restarts')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                       help='Fraction of total steps for warmup')
    parser.add_argument('--min-lr-ratio', type=float, default=0.01,
                       help='Minimum LR as fraction of max LR')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--encoder-layers', type=int, default=4)
    parser.add_argument('--decoder-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Evaluation
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--save-every-steps', type=int, default=10000,
                       help='Save checkpoint every N steps (0 to disable)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--gradient-checkpointing', action='store_true')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        eval_every_n_steps=args.eval_every,
        log_every_n_steps=args.log_every,
        save_every_n_steps=args.save_every_steps,
        device=args.device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.project_root:
        config.project_root = Path(args.project_root)

    # Create trainer
    trainer = NextBasketTrainer(config)

    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.best_f1 = checkpoint.get('best_f1', 0.0)
        logger.info(f"Resumed from {args.resume}")

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
