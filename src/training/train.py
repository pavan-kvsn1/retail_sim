"""
Training Script for World Model.

Implements three-phase training:
1. Warm-up (epochs 1-3): Focus on primary task, low LR
2. Main Training (epochs 4-15): Full multi-task learning
3. Fine-tuning (epochs 16-20): Refinement with harder masking

Features:
- Bucket-based batching for efficient padding
- Gradient accumulation for large effective batch size
- Mixed precision training (optional)
- Validation-based early stopping
- Checkpoint saving

Reference: RetailSim_Data_Pipeline_and_World_Model_Design.md Section 5.6
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import math
from pathlib import Path
import json
import logging
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from .model import WorldModel, WorldModelConfig
from .losses import WorldModelLoss
from .dataset import WorldModelDataset, WorldModelDataLoader, EvaluationDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Paths
    project_root: str = '/Users/hazymoji/Documents/DataDev/ML Projects/retail_sim'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    # Model
    n_products: int = 5003
    d_model: int = 512
    mamba_layers: int = 4
    decoder_layers: int = 2

    # Training
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 20
    warmup_epochs: int = 3
    finetune_epochs: int = 5

    # Learning rate schedule
    warmup_steps: Optional[int] = None  # If None, uses warmup_epochs
    use_cosine_schedule: bool = True
    min_lr_ratio: float = 0.01  # Minimum LR as fraction of max LR

    # Masking
    mask_prob_train: float = 0.15
    mask_prob_finetune: float = 0.20
    max_seq_len: int = 50

    # Validation
    eval_every_n_steps: int = 500  # Validate every 500 steps
    save_every_n_steps: int = 2000
    early_stopping_patience: int = 3

    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    mixed_precision: bool = True
    num_workers: int = 4
    gradient_checkpointing: bool = False

    def get_phase(self, epoch: int) -> str:
        """Get training phase for current epoch (1-indexed)."""
        if epoch <= self.warmup_epochs:
            return 'warmup'
        elif epoch > self.num_epochs - self.finetune_epochs:
            return 'finetune'
        else:
            return 'main'

    def get_learning_rate(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        phase = self.get_phase(epoch)
        if phase == 'warmup':
            return 1e-5
        elif phase == 'finetune':
            return 1e-5
        else:
            return self.learning_rate

    def get_mask_prob(self, epoch: int) -> float:
        """Get mask probability for current epoch."""
        if self.get_phase(epoch) == 'finetune':
            return self.mask_prob_finetune
        return self.mask_prob_train


class Trainer:
    """World Model trainer with multi-phase training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.project_root = Path(config.project_root)

        # Create directories
        self.checkpoint_dir = self.project_root / config.checkpoint_dir
        self.log_dir = self.project_root / config.log_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self._init_model()

        # Initialize loss
        self.criterion = WorldModelLoss(
            n_products=config.n_products,
            focal_gamma=2.0,
            contrastive_temperature=0.07
        )

        # Initialize datasets
        self._init_datasets()

        # Initialize optimizer (will be reset per phase)
        self._init_optimizer()

        # Scheduler will be initialized in train() when we know total steps
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.last_val_loss = float('inf')
        self.last_train_loss = float('inf')
        self.patience_counter = 0
        self.training_log = []

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision and 'cuda' in config.device else None

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {self.model.num_parameters:,}")

    def _init_model(self):
        """Initialize World Model."""
        model_config = WorldModelConfig(
            n_products=self.config.n_products,
            d_model=self.config.d_model,
            mamba_num_layers=self.config.mamba_layers,
            decoder_num_layers=self.config.decoder_layers,
            max_basket_len=self.config.max_seq_len
        )
        self.model = WorldModel(model_config).to(self.device)

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    def _init_datasets(self):
        """Initialize train and validation datasets."""
        logger.info("Loading training dataset...")
        self.train_dataset = WorldModelDataset(
            self.project_root,
            split='train',
            max_seq_len=self.config.max_seq_len,
            mask_prob=self.config.mask_prob_train,
            load_transactions=True
        )

        logger.info("Loading validation dataset...")
        self.val_dataset = WorldModelDataset(
            self.project_root,
            split='validation',
            max_seq_len=self.config.max_seq_len,
            mask_prob=self.config.mask_prob_train,
            load_transactions=True
        )

        logger.info(f"Train samples: {len(self.train_dataset):,}")
        logger.info(f"Validation samples: {len(self.val_dataset):,}")

        if len(self.val_dataset) == 0:
            logger.warning("WARNING: Validation dataset is empty! Validation will not work properly.")

    def _init_optimizer(self, learning_rate: Optional[float] = None):
        """Initialize optimizer with optional LR override."""
        lr = learning_rate or self.config.learning_rate
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay
        )

    def _init_scheduler(self, total_steps: int):
        """Initialize learning rate scheduler with warmup + cosine annealing."""
        # Calculate warmup steps
        if self.config.warmup_steps is not None:
            warmup_steps = self.config.warmup_steps
        else:
            steps_per_epoch = len(self.train_loader)
            warmup_steps = self.config.warmup_epochs * steps_per_epoch

        min_lr = self.config.learning_rate * self.config.min_lr_ratio

        def lr_lambda(current_step: int) -> float:
            """
            Linear warmup then cosine annealing.

            Returns multiplier for base learning rate.
            """
            if current_step < warmup_steps:
                # Linear warmup: 0 -> 1
                return float(current_step) / float(max(1, warmup_steps))

            if not self.config.use_cosine_schedule:
                # Constant LR after warmup
                return 1.0

            # Cosine annealing: 1 -> min_lr_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale between min_lr_ratio and 1.0
            return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine_decay

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"Scheduler initialized: {warmup_steps:,} warmup steps, "
                    f"{total_steps - warmup_steps:,} cosine annealing steps")

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

    def train_step(
        self,
        batch_data: Dict[str, torch.Tensor],
        phase: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step."""
        self.model.train()

        # Forward pass
        use_amp = self.scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            masked_logits, aux_logits, encoder_output = self.model(
                dense_context=batch_data['dense_context'],
                product_embeddings=batch_data['product_embeddings'],
                price_features=batch_data['price_features'],
                attention_mask=batch_data['attention_mask'],
                masked_positions=batch_data['masked_positions']
            )

            # Create masked mask (which positions have valid targets)
            masked_mask = (batch_data['masked_targets'] > 0).float() if batch_data['masked_targets'] is not None else None

            # Compute loss
            loss, loss_dict = self.criterion(
                masked_logits=masked_logits,
                masked_targets=batch_data['masked_targets'],
                masked_mask=masked_mask,
                product_embeddings=encoder_output,
                product_ids=batch_data['product_ids'],
                attention_mask=batch_data['attention_mask'],
                auxiliary_logits=aux_logits,
                auxiliary_labels=batch_data['auxiliary_labels'],
                phase=phase
            )

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss, {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        self.model.eval()

        val_loader = WorldModelDataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            apply_masking=True,
            bucket_batching=True
        )

        total_losses = {'total': 0.0}  # Initialize with default
        n_batches = 0
        debug_logged = False

        for batch in val_loader:
            batch_data = self._prepare_batch(batch)

            # Debug: Check if masking is applied (only log once)
            if not debug_logged:
                has_masked_pos = batch_data['masked_positions'] is not None
                has_masked_tgt = batch_data['masked_targets'] is not None
                if has_masked_pos and has_masked_tgt:
                    n_masked = (batch_data['masked_targets'] > 0).sum().item()
                    logger.info(f"Validation debug: masked_positions={has_masked_pos}, masked_targets={has_masked_tgt}, n_valid_targets={n_masked}")
                else:
                    logger.warning(f"Validation debug: MISSING masked_positions={has_masked_pos}, masked_targets={has_masked_tgt}")
                debug_logged = True

            masked_logits, aux_logits, encoder_output = self.model(
                dense_context=batch_data['dense_context'],
                product_embeddings=batch_data['product_embeddings'],
                price_features=batch_data['price_features'],
                attention_mask=batch_data['attention_mask'],
                masked_positions=batch_data['masked_positions']
            )

            masked_mask = (batch_data['masked_targets'] > 0).float() if batch_data['masked_targets'] is not None else None

            _, loss_dict = self.criterion(
                masked_logits=masked_logits,
                masked_targets=batch_data['masked_targets'],
                masked_mask=masked_mask,
                product_embeddings=encoder_output,
                product_ids=batch_data['product_ids'],
                attention_mask=batch_data['attention_mask'],
                auxiliary_logits=aux_logits,
                auxiliary_labels=batch_data['auxiliary_labels'],
                phase='main'
            )

            # Debug: Log first batch losses
            if n_batches == 0:
                loss_summary = {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
                logger.info(f"Validation debug - first batch losses: {loss_summary}")

            for k, v in loss_dict.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item() if torch.is_tensor(v) else v

            n_batches += 1

            # Limit validation batches for speed
            if n_batches >= 100:
                break

        # Handle edge case of no batches
        if n_batches == 0:
            logger.warning("Validation: No batches processed! Check validation dataset.")
            return {'total': float('inf')}

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        logger.debug(f"Validation completed: {n_batches} batches, loss={avg_losses.get('total', 0):.4f}")
        return avg_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }

        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint['epoch']

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        phase = self.config.get_phase(epoch)
        mask_prob = self.config.get_mask_prob(epoch)

        # Get current LR from scheduler (for logging only)
        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate

        # Update dataset mask probability
        self.train_dataset.mask_prob = mask_prob

        # Update loss weights for phase
        self.criterion.set_phase(phase)

        logger.info(f"Epoch {epoch}/{self.config.num_epochs} - Phase: {phase}, LR: {current_lr:.2e}, Mask: {mask_prob}")

        # Create dataloader
        train_loader = WorldModelDataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            apply_masking=True,
            bucket_batching=True
        )

        epoch_losses = {}
        n_steps = 0
        start_time = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch_data = self._prepare_batch(batch)

            loss, loss_dict = self.train_step(batch_data, phase)

            # Accumulate losses for logging
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v

            n_steps += 1

            # Optimizer step after gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer_step()
                self.global_step += 1

                # Step scheduler (per step, not per epoch)
                if self.scheduler is not None:
                    self.scheduler.step()

                # Logging
                if self.global_step % 100 == 0:
                    avg_loss = epoch_losses['total'] / n_steps
                    self.last_train_loss = avg_loss
                    elapsed = time.time() - start_time
                    samples_per_sec = (n_steps * self.config.batch_size) / elapsed

                    # Include last validation loss for reference
                    if self.last_val_loss < float('inf'):
                        gap = self.last_val_loss - avg_loss
                        gap_str = f"+{gap:.4f}" if gap > 0 else f"{gap:.4f}"
                        logger.info(
                            f"Step {self.global_step} | Train: {avg_loss:.4f} | "
                            f"Val: {self.last_val_loss:.4f} (gap: {gap_str}) | "
                            f"Speed: {samples_per_sec:.1f} samples/s"
                        )
                    else:
                        logger.info(
                            f"Step {self.global_step} | Train: {avg_loss:.4f} | "
                            f"Speed: {samples_per_sec:.1f} samples/s"
                        )

                # Validation
                if self.global_step % self.config.eval_every_n_steps == 0:
                    logger.info(f"Running validation at step {self.global_step}...")
                    val_metrics = self.validate()
                    val_loss = val_metrics.get('total', float('inf'))
                    self.last_val_loss = val_loss
                    gap = val_loss - self.last_train_loss
                    gap_str = f"+{gap:.4f}" if gap > 0 else f"{gap:.4f}"

                    logger.info(
                        f">>> Validation | Train: {self.last_train_loss:.4f} | "
                        f"Val: {val_loss:.4f} | Gap: {gap_str} | "
                        f"Best: {self.best_val_loss:.4f}"
                    )

                    # Early stopping check
                    if val_metrics['total'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['total']
                        self.patience_counter = 0
                        self.save_checkpoint(epoch, is_best=True)
                    else:
                        self.patience_counter += 1

                # Checkpoint
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(epoch)

        # Average epoch losses
        epoch_losses = {k: v / n_steps for k, v in epoch_losses.items()}
        epoch_losses['epoch'] = epoch
        epoch_losses['phase'] = phase
        epoch_losses['time'] = time.time() - start_time

        return epoch_losses

    def train(self, resume_from: Optional[str] = None):
        """Full training loop."""
        start_epoch = 1

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            logger.info(f"Resuming from epoch {start_epoch}")

        logger.info("Starting training...")

        # Calculate total steps and initialize scheduler
        # Create a temporary dataloader to get step count
        temp_loader = WorldModelDataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            apply_masking=False,
            bucket_batching=True
        )
        steps_per_epoch = len(temp_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs
        self.train_loader = temp_loader  # Store for scheduler init
        self._init_scheduler(total_steps)
        logger.info(f"Total training steps: {total_steps:,} ({steps_per_epoch:,} per epoch)")

        # Initial validation before training starts
        logger.info("Running initial validation...")
        val_metrics = self.validate()
        self.last_val_loss = val_metrics.get('total', float('inf'))
        self.best_val_loss = self.last_val_loss
        logger.info(f"Initial validation loss: {self.last_val_loss:.4f}")

        for epoch in range(start_epoch, self.config.num_epochs + 1):
            epoch_metrics = self.train_epoch(epoch)

            # End of epoch validation
            val_metrics = self.validate()
            self.last_val_loss = val_metrics['total']
            gap = val_metrics['total'] - epoch_metrics['total']
            gap_str = f"+{gap:.4f}" if gap > 0 else f"{gap:.4f}"

            # Log epoch results with both train and val
            logger.info(
                f"{'='*60}\n"
                f"Epoch {epoch}/{self.config.num_epochs} Complete [{epoch_metrics['phase'].upper()}]\n"
                f"  Train Loss: {epoch_metrics['total']:.4f}\n"
                f"  Val Loss:   {val_metrics['total']:.4f} (gap: {gap_str})\n"
                f"  Best Val:   {self.best_val_loss:.4f}\n"
                f"  Time:       {epoch_metrics['time']:.1f}s\n"
                f"{'='*60}"
            )

            # Add val metrics to epoch log
            epoch_metrics['val_loss'] = val_metrics['total']
            epoch_metrics['val_gap'] = gap
            self.training_log.append(epoch_metrics)

            # Check for best model
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model! Val Loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch)

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (patience exhausted)")
                break

        # Save training log
        log_path = self.log_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        logger.info(f"Training log saved to {log_path}")

        logger.info("Training complete!")


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train World Model')
    parser.add_argument('--project-root', type=str,
                        default='/Users/hazymoji/Documents/DataDev/ML Projects/retail_sim')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to reduce memory usage')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--eval-every', type=int, default=500,
                        help='Run validation every N steps (default: 500)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    config = TrainingConfig(
        project_root=args.project_root,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        mixed_precision=(device == 'cuda'),
        gradient_checkpointing=args.gradient_checkpointing,
        num_workers=args.num_workers,
        max_grad_norm=args.max_grad_norm,
        eval_every_n_steps=args.eval_every
    )

    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
