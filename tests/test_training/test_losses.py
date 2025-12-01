"""
Unit tests for World Model loss functions.

Tests:
- FocalLoss: Correct focal weighting, masking, class imbalance handling
- ContrastiveLoss: Positive/negative sampling, temperature scaling
- AuxiliaryLoss: Multi-task cross-entropy
- WorldModelLoss: Combined loss with phase-specific weights
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.losses import (
    FocalLoss,
    ContrastiveLoss,
    AuxiliaryLoss,
    WorldModelLoss,
)


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_focal_loss_initialization(self):
        """Test FocalLoss initializes correctly."""
        loss_fn = FocalLoss(gamma=2.0)
        assert loss_fn.gamma == 2.0
        assert loss_fn.reduction == 'mean'

    def test_focal_loss_basic(self):
        """Test basic focal loss computation."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 100)
        targets = torch.randint(0, 100, (8,))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
        assert not torch.isnan(loss)

    def test_focal_loss_sequence(self):
        """Test focal loss with sequence input."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 10, 100)  # [B, S, C]
        targets = torch.randint(0, 100, (4, 10))  # [B, S]

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_focal_loss_with_mask(self):
        """Test focal loss with masking."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 10, 100)
        targets = torch.randint(0, 100, (4, 10))
        mask = torch.zeros(4, 10)
        mask[:, :5] = 1  # Only first 5 positions valid

        loss = loss_fn(logits, targets, mask)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_focal_loss_ignore_index(self):
        """Test focal loss ignores specified index."""
        loss_fn = FocalLoss(gamma=2.0, ignore_index=-100)
        logits = torch.randn(4, 100)
        targets = torch.tensor([10, -100, 20, -100])  # Two ignored

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss)

    def test_focal_vs_ce_loss(self):
        """Test focal loss reduces to CE when gamma=0."""
        focal_loss = FocalLoss(gamma=0.0, reduction='mean')
        ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        logits = torch.randn(8, 100)
        targets = torch.randint(0, 100, (8,))

        focal = focal_loss(logits, targets)
        ce = ce_loss(logits, targets)

        # Should be approximately equal when gamma=0
        assert torch.allclose(focal, ce, rtol=1e-4)

    def test_focal_loss_hard_examples(self):
        """Test focal loss up-weights hard examples."""
        loss_fn = FocalLoss(gamma=2.0, reduction='none')

        # Create easy example (high confidence)
        easy_logits = torch.tensor([[10.0, -10.0]])
        easy_target = torch.tensor([0])

        # Create hard example (low confidence)
        hard_logits = torch.tensor([[0.1, 0.0]])
        hard_target = torch.tensor([0])

        easy_loss = loss_fn(easy_logits, easy_target)
        hard_loss = loss_fn(hard_logits, hard_target)

        # Hard example should have higher loss
        assert hard_loss.item() > easy_loss.item()


class TestContrastiveLoss:
    """Tests for ContrastiveLoss."""

    def test_contrastive_loss_initialization(self):
        """Test ContrastiveLoss initializes correctly."""
        loss_fn = ContrastiveLoss(temperature=0.07)
        assert loss_fn.temperature == 0.07

    def test_contrastive_loss_basic(self):
        """Test basic contrastive loss computation."""
        loss_fn = ContrastiveLoss()

        embeddings = torch.randn(4, 10, 256)  # [B, S, D]
        ids = torch.randint(1, 100, (4, 10))
        mask = torch.ones(4, 10)

        loss = loss_fn(embeddings, ids, mask)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_contrastive_loss_with_padding(self):
        """Test contrastive loss handles padding correctly."""
        loss_fn = ContrastiveLoss()

        embeddings = torch.randn(4, 10, 256)
        ids = torch.randint(1, 100, (4, 10))
        mask = torch.zeros(4, 10)
        mask[:, :5] = 1  # Only first 5 valid

        loss = loss_fn(embeddings, ids, mask)

        assert not torch.isnan(loss)

    def test_contrastive_loss_temperature(self):
        """Test temperature affects loss magnitude."""
        low_temp = ContrastiveLoss(temperature=0.01)
        high_temp = ContrastiveLoss(temperature=1.0)

        embeddings = torch.randn(4, 10, 256)
        ids = torch.randint(1, 100, (4, 10))
        mask = torch.ones(4, 10)

        loss_low = low_temp(embeddings, ids, mask)
        loss_high = high_temp(embeddings, ids, mask)

        # Lower temperature should give higher loss
        assert loss_low.item() > loss_high.item()


class TestAuxiliaryLoss:
    """Tests for AuxiliaryLoss."""

    def test_auxiliary_loss_initialization(self):
        """Test AuxiliaryLoss initializes correctly."""
        loss_fn = AuxiliaryLoss()
        assert loss_fn is not None

    def test_auxiliary_loss_all_tasks(self):
        """Test auxiliary loss computes all tasks."""
        loss_fn = AuxiliaryLoss()

        logits = {
            'basket_size': torch.randn(8, 4),
            'price_sensitivity': torch.randn(8, 4),
            'mission_type': torch.randn(8, 5),
            'mission_focus': torch.randn(8, 6),
        }
        labels = {
            'basket_size': torch.randint(1, 4, (8,)),
            'price_sensitivity': torch.randint(1, 4, (8,)),
            'mission_type': torch.randint(1, 5, (8,)),
            'mission_focus': torch.randint(1, 6, (8,)),
        }

        losses = loss_fn(logits, labels)

        assert 'basket_size' in losses
        assert 'price_sensitivity' in losses
        assert 'mission_type' in losses
        assert 'mission_focus' in losses

    def test_auxiliary_loss_partial_tasks(self):
        """Test auxiliary loss handles partial task sets."""
        loss_fn = AuxiliaryLoss()

        logits = {'basket_size': torch.randn(8, 4)}
        labels = {'basket_size': torch.randint(1, 4, (8,))}

        losses = loss_fn(logits, labels)

        assert 'basket_size' in losses
        assert losses['basket_size'].item() > 0


class TestWorldModelLoss:
    """Tests for WorldModelLoss."""

    def test_world_model_loss_initialization(self):
        """Test WorldModelLoss initializes correctly."""
        loss_fn = WorldModelLoss()
        assert loss_fn.w_focal == 0.60
        assert loss_fn.w_contrastive == 0.20

    def test_world_model_loss_warmup_phase(self):
        """Test loss in warmup phase (focal only)."""
        loss_fn = WorldModelLoss()

        masked_logits = torch.randn(4, 3, 5003)
        masked_targets = torch.randint(1, 5000, (4, 3))
        masked_mask = torch.ones(4, 3)
        embeddings = torch.randn(4, 10, 256)
        ids = torch.randint(1, 100, (4, 10))
        mask = torch.ones(4, 10)

        total, loss_dict = loss_fn(
            masked_logits, masked_targets, masked_mask,
            embeddings, ids, mask,
            phase='warmup'
        )

        # In warmup, total should equal focal
        assert torch.allclose(total, loss_dict['focal'])
        # Contrastive should be 0
        assert loss_dict['contrastive'].item() == 0.0

    def test_world_model_loss_main_phase(self):
        """Test loss in main phase (all components)."""
        loss_fn = WorldModelLoss()

        masked_logits = torch.randn(4, 3, 5003)
        masked_targets = torch.randint(1, 5000, (4, 3))
        masked_mask = torch.ones(4, 3)
        embeddings = torch.randn(4, 10, 256)
        ids = torch.randint(1, 100, (4, 10))
        mask = torch.ones(4, 10)
        aux_logits = {
            'basket_size': torch.randn(4, 4),
            'price_sensitivity': torch.randn(4, 4),
            'mission_type': torch.randn(4, 5),
            'mission_focus': torch.randn(4, 6),
        }
        aux_labels = {
            'basket_size': torch.randint(1, 4, (4,)),
            'price_sensitivity': torch.randint(1, 4, (4,)),
            'mission_type': torch.randint(1, 5, (4,)),
            'mission_focus': torch.randint(1, 6, (4,)),
        }

        total, loss_dict = loss_fn(
            masked_logits, masked_targets, masked_mask,
            embeddings, ids, mask,
            auxiliary_logits=aux_logits,
            auxiliary_labels=aux_labels,
            phase='main'
        )

        assert 'total' in loss_dict
        assert 'focal' in loss_dict
        assert 'contrastive' in loss_dict
        assert total.item() > 0

    def test_world_model_loss_set_phase(self):
        """Test phase-specific weight updates."""
        loss_fn = WorldModelLoss()

        loss_fn.set_phase('warmup')
        assert loss_fn.w_focal == 1.0
        assert loss_fn.w_contrastive == 0.0

        loss_fn.set_phase('main')
        assert loss_fn.w_focal == 0.60
        assert loss_fn.w_contrastive == 0.20

    def test_world_model_loss_gradient_flow(self):
        """Test gradients flow correctly through loss."""
        loss_fn = WorldModelLoss()

        masked_logits = torch.randn(4, 3, 5003, requires_grad=True)
        masked_targets = torch.randint(1, 5000, (4, 3))
        masked_mask = torch.ones(4, 3)
        embeddings = torch.randn(4, 10, 256, requires_grad=True)
        ids = torch.randint(1, 100, (4, 10))
        mask = torch.ones(4, 10)

        total, _ = loss_fn(
            masked_logits, masked_targets, masked_mask,
            embeddings, ids, mask,
            phase='main'
        )

        total.backward()

        assert masked_logits.grad is not None
        assert embeddings.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
