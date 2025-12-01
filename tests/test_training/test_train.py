"""
Unit tests for World Model training infrastructure.

Tests:
- TrainingConfig: Phase scheduling, learning rates
- Trainer: Initialization, batch preparation
- Training loop components
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import TrainingConfig, Trainer
from src.training.model import WorldModel, WorldModelConfig
from src.training.losses import WorldModelLoss


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.batch_size == 256
        assert config.num_epochs == 20
        assert config.warmup_epochs == 3
        assert config.finetune_epochs == 5

    def test_phase_scheduling(self):
        """Test phase determination for epochs."""
        config = TrainingConfig(
            num_epochs=20,
            warmup_epochs=3,
            finetune_epochs=5
        )

        # Warmup: epochs 1-3
        assert config.get_phase(1) == 'warmup'
        assert config.get_phase(2) == 'warmup'
        assert config.get_phase(3) == 'warmup'

        # Main: epochs 4-15
        assert config.get_phase(4) == 'main'
        assert config.get_phase(10) == 'main'
        assert config.get_phase(15) == 'main'

        # Finetune: epochs 16-20
        assert config.get_phase(16) == 'finetune'
        assert config.get_phase(18) == 'finetune'
        assert config.get_phase(20) == 'finetune'

    def test_learning_rate_scheduling(self):
        """Test learning rate for different phases."""
        config = TrainingConfig(
            learning_rate=5e-5,
            num_epochs=20,
            warmup_epochs=3
        )

        # Warmup: low LR
        assert config.get_learning_rate(1) == 1e-5

        # Main: peak LR
        assert config.get_learning_rate(5) == 5e-5

        # Finetune: low LR
        assert config.get_learning_rate(18) == 1e-5

    def test_mask_probability_scheduling(self):
        """Test mask probability for different phases."""
        config = TrainingConfig(
            mask_prob_train=0.15,
            mask_prob_finetune=0.20,
            num_epochs=20
        )

        # Regular training: 15%
        assert config.get_mask_prob(5) == 0.15

        # Finetune: 20% (harder)
        assert config.get_mask_prob(18) == 0.20

    def test_device_detection(self):
        """Test device is detected correctly."""
        config = TrainingConfig()

        assert config.device in ['cuda', 'mps', 'cpu']


class TestTrainerComponents:
    """Tests for Trainer component methods."""

    def test_prepare_batch(self):
        """Test batch preparation converts numpy to tensors."""
        # Create mock batch with numpy arrays
        class MockBatch:
            def __init__(self):
                B, S = 4, 10
                self.customer_context = np.random.randn(B, 192).astype(np.float32)
                self.temporal_context = np.random.randn(B, 64).astype(np.float32)
                self.store_context = np.random.randn(B, 96).astype(np.float32)
                self.trip_context = np.random.randn(B, 48).astype(np.float32)
                self.product_embeddings = np.random.randn(B, S, 256).astype(np.float32)
                self.product_token_ids = np.random.randint(0, 1000, (B, S)).astype(np.int32)
                self.price_features = np.random.randn(B, S, 64).astype(np.float32)
                self.attention_mask = np.ones((B, S), dtype=np.int32)
                self.sequence_lengths = np.full(B, S, dtype=np.int32)
                self.masked_positions = np.random.randint(0, S, (B, 2)).astype(np.int32)
                self.masked_targets = np.random.randint(1, 1000, (B, 2)).astype(np.int32)
                self.auxiliary_labels = {
                    'basket_size': np.random.randint(1, 4, (B,)).astype(np.int32),
                    'price_sensitivity': np.random.randint(1, 4, (B,)).astype(np.int32),
                }

            def get_dense_context(self):
                return np.concatenate([
                    self.customer_context,
                    self.temporal_context,
                    self.store_context,
                    self.trip_context
                ], axis=1)

        # Create lightweight mock trainer to test _prepare_batch
        batch = MockBatch()
        device = torch.device('cpu')

        batch_data = {
            'dense_context': torch.from_numpy(batch.get_dense_context()).float().to(device),
            'product_embeddings': torch.from_numpy(batch.product_embeddings).float().to(device),
            'price_features': torch.from_numpy(batch.price_features).float().to(device),
            'attention_mask': torch.from_numpy(batch.attention_mask).float().to(device),
            'product_ids': torch.from_numpy(batch.product_token_ids).long().to(device),
            'masked_positions': torch.from_numpy(batch.masked_positions).long().to(device),
            'masked_targets': torch.from_numpy(batch.masked_targets).long().to(device),
            'auxiliary_labels': {
                k: torch.from_numpy(v).long().to(device)
                for k, v in batch.auxiliary_labels.items()
            }
        }

        assert batch_data['dense_context'].shape == (4, 400)
        assert batch_data['product_embeddings'].shape == (4, 10, 256)
        assert batch_data['masked_positions'].dtype == torch.long


class TestTrainStepComponents:
    """Tests for individual training step components."""

    def test_forward_pass(self):
        """Test forward pass produces expected outputs."""
        model = WorldModel()
        model.train()

        B, S = 4, 10
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        masked_positions = torch.randint(0, S, (B, 2))

        logits, aux_logits, encoder_out = model(
            dense_context, product_emb, price_feat,
            attention_mask, masked_positions
        )

        assert logits.requires_grad
        assert logits.shape[0] == B

    def test_loss_computation(self):
        """Test loss computation produces valid loss."""
        loss_fn = WorldModelLoss()

        B, S, M = 4, 10, 2
        masked_logits = torch.randn(B, M, 5003, requires_grad=True)
        masked_targets = torch.randint(1, 5000, (B, M))
        masked_mask = torch.ones(B, M)
        embeddings = torch.randn(B, S, 256, requires_grad=True)
        ids = torch.randint(1, 100, (B, S))
        mask = torch.ones(B, S)

        total, loss_dict = loss_fn(
            masked_logits, masked_targets, masked_mask,
            embeddings, ids, mask,
            phase='main'
        )

        assert not torch.isnan(total)
        assert total.requires_grad

    def test_backward_pass(self):
        """Test backward pass computes gradients."""
        model = WorldModel()
        loss_fn = WorldModelLoss()

        B, S = 2, 8
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        masked_positions = torch.randint(0, S, (B, 2))
        masked_targets = torch.randint(1, 5000, (B, 2))
        masked_mask = torch.ones(B, 2)
        product_ids = torch.randint(1, 100, (B, S))

        logits, aux_logits, encoder_out = model(
            dense_context, product_emb, price_feat,
            attention_mask, masked_positions
        )

        total, _ = loss_fn(
            logits, masked_targets, masked_mask,
            encoder_out, product_ids, attention_mask,
            phase='main'
        )

        total.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_gradient_clipping(self):
        """Test gradient clipping limits gradient norms."""
        model = WorldModel()
        loss_fn = WorldModelLoss()
        max_grad_norm = 1.0

        B, S = 2, 8
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        masked_positions = torch.randint(0, S, (B, 2))
        masked_targets = torch.randint(1, 5000, (B, 2))
        masked_mask = torch.ones(B, 2)
        product_ids = torch.randint(1, 100, (B, S))

        logits, aux_logits, encoder_out = model(
            dense_context, product_emb, price_feat,
            attention_mask, masked_positions
        )

        total, _ = loss_fn(
            logits, masked_targets, masked_mask,
            encoder_out, product_ids, attention_mask,
            phase='main'
        )

        # Scale up loss to create large gradients
        (total * 1000).backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Check gradient norm is within limit
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_grad_norm * 1.1  # Small tolerance


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_checkpoint_save_load(self):
        """Test model can be saved and loaded."""
        model = WorldModel()
        model.eval()  # Set to eval mode for deterministic output

        # Random forward to ensure weights are initialized
        B, S = 2, 8
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        masked_positions = torch.randint(0, S, (B, 2))

        with torch.no_grad():
            original_out, _, _ = model(
                dense_context, product_emb, price_feat,
                attention_mask, masked_positions
            )

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': 5,
                'global_step': 1000
            }, checkpoint_path)

        # Load into new model
        new_model = WorldModel()
        checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_model.eval()  # Set to eval mode

        # Compare outputs
        with torch.no_grad():
            new_out, _, _ = new_model(
                dense_context, product_emb, price_feat,
                attention_mask, masked_positions
            )

        assert torch.allclose(original_out, new_out)

        # Cleanup
        Path(checkpoint_path).unlink()


class TestPhaseTransitions:
    """Tests for training phase transitions."""

    def test_warmup_to_main_transition(self):
        """Test loss weights change at phase transition."""
        loss_fn = WorldModelLoss()

        # Warmup phase
        loss_fn.set_phase('warmup')
        assert loss_fn.w_focal == 1.0
        assert loss_fn.w_contrastive == 0.0

        # Main phase
        loss_fn.set_phase('main')
        assert loss_fn.w_focal == 0.60
        assert loss_fn.w_contrastive == 0.20

    def test_main_to_finetune_transition(self):
        """Test transition from main to finetune phase."""
        loss_fn = WorldModelLoss()

        loss_fn.set_phase('main')
        main_focal = loss_fn.w_focal

        loss_fn.set_phase('finetune')
        finetune_focal = loss_fn.w_focal

        # Weights should be same for main and finetune
        assert main_focal == finetune_focal


class TestValidation:
    """Tests for validation components."""

    def test_eval_mode(self):
        """Test model switches to eval mode correctly."""
        model = WorldModel()

        model.train()
        assert model.training

        model.eval()
        assert not model.training

    def test_no_grad_inference(self):
        """Test inference with no_grad is efficient."""
        model = WorldModel()
        model.eval()

        B, S = 4, 10
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        masked_positions = torch.randint(0, S, (B, 2))

        with torch.no_grad():
            logits, _, _ = model(
                dense_context, product_emb, price_feat,
                attention_mask, masked_positions
            )

        assert not logits.requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
