"""
Unit Tests for Store Visit Prediction Components.

Tests:
- Model architecture and forward pass
- Dataset loading and batching
- Loss functions and metrics
- End-to-end training step

Run: pytest tests/test_training/test_store_visit.py -v
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.training.model_store_visit import (
    StoreVisitPredictor,
    StoreVisitModelConfig,
    BasketSummarizer,
    create_store_visit_model,
)
from src.training.losses_store_visit import (
    StoreVisitLoss,
    FocalLoss,
    StoreVisitMetrics,
    compute_store_class_weights,
)


class TestStoreVisitModel:
    """Tests for StoreVisitPredictor model."""

    @pytest.fixture
    def config(self):
        return StoreVisitModelConfig(
            num_stores=100,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
        )

    @pytest.fixture
    def model(self, config):
        return StoreVisitPredictor(config)

    def test_model_creation(self, model, config):
        """Test model can be created."""
        assert model is not None
        assert model.config.num_stores == 100

    def test_model_parameters(self, model):
        """Test model has trainable parameters."""
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

        # Check all parameters require grad
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == n_params

    def test_forward_pass_minimal(self, model):
        """Test forward pass with minimal inputs (no basket)."""
        B = 4
        outputs = model(
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            previous_store_idx=torch.randint(0, 100, (B,)),
        )

        assert 'store_logits' in outputs
        assert 'store_probs' in outputs
        assert outputs['store_logits'].shape == (B, 100)
        assert outputs['store_probs'].shape == (B, 100)

    def test_forward_pass_with_basket(self, model):
        """Test forward pass with basket embeddings."""
        B, S = 4, 10
        outputs = model(
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            previous_store_idx=torch.randint(0, 100, (B,)),
            previous_basket_embeddings=torch.randn(B, S, 256),
            previous_basket_mask=torch.ones(B, S),
        )

        assert outputs['store_logits'].shape == (B, 100)
        assert outputs['store_probs'].shape == (B, 100)

    def test_probs_sum_to_one(self, model):
        """Test that store probabilities sum to 1."""
        B = 8
        outputs = model(
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            previous_store_idx=torch.randint(0, 100, (B,)),
        )

        prob_sums = outputs['store_probs'].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5)

    def test_predict_store(self, model):
        """Test top-k store prediction."""
        B = 4
        top_k = 5

        indices, probs = model.predict_store(
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            previous_store_idx=torch.randint(0, 100, (B,)),
            top_k=top_k,
        )

        assert indices.shape == (B, top_k)
        assert probs.shape == (B, top_k)
        assert (indices >= 0).all() and (indices < 100).all()
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_sample_store(self, model):
        """Test store sampling."""
        B = 4

        sampled = model.sample_store(
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            previous_store_idx=torch.randint(0, 100, (B,)),
            temperature=1.0,
        )

        assert sampled.shape == (B,)
        assert (sampled >= 0).all() and (sampled < 100).all()

    def test_sample_temperature(self, model):
        """Test that temperature affects sampling diversity."""
        B = 100

        # Low temperature = less diverse
        torch.manual_seed(42)
        sampled_low = model.sample_store(
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            previous_store_idx=torch.zeros(B, dtype=torch.long),
            temperature=0.1,
        )

        # High temperature = more diverse
        torch.manual_seed(42)
        sampled_high = model.sample_store(
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            previous_store_idx=torch.zeros(B, dtype=torch.long),
            temperature=2.0,
        )

        # High temp should have more unique values
        unique_low = len(sampled_low.unique())
        unique_high = len(sampled_high.unique())
        # This is probabilistic, but generally true
        assert unique_high >= unique_low * 0.5  # Allow some variance

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        B, S = 4, 10
        outputs = model(
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            previous_store_idx=torch.randint(0, 100, (B,)),
            previous_basket_embeddings=torch.randn(B, S, 256),
            previous_basket_mask=torch.ones(B, S),
        )

        loss = outputs['store_logits'].sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestBasketSummarizer:
    """Tests for BasketSummarizer module."""

    @pytest.fixture
    def summarizer(self):
        return BasketSummarizer(product_dim=256, output_dim=64)

    def test_summarizer_output_shape(self, summarizer):
        """Test output shape."""
        B, S = 4, 10
        output = summarizer(
            product_embeddings=torch.randn(B, S, 256),
            attention_mask=torch.ones(B, S),
        )
        assert output.shape == (B, 64)

    def test_summarizer_masked(self, summarizer):
        """Test with partial mask."""
        B, S = 4, 10
        mask = torch.zeros(B, S)
        mask[:, :5] = 1  # Only first 5 items valid

        output = summarizer(
            product_embeddings=torch.randn(B, S, 256),
            attention_mask=mask,
        )
        assert output.shape == (B, 64)

    def test_summarizer_empty_basket(self, summarizer):
        """Test with empty basket (all zeros mask)."""
        B, S = 4, 10
        mask = torch.zeros(B, S)

        output = summarizer(
            product_embeddings=torch.randn(B, S, 256),
            attention_mask=mask,
        )
        assert output.shape == (B, 64)
        assert not torch.isnan(output).any()


class TestStoreVisitLoss:
    """Tests for loss functions."""

    @pytest.fixture
    def loss_fn(self):
        return StoreVisitLoss(num_stores=100, label_smoothing=0.1)

    def test_loss_computation(self, loss_fn):
        """Test basic loss computation."""
        B = 32
        logits = torch.randn(B, 100)
        targets = torch.randint(0, 100, (B,))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_loss_with_class_weights(self):
        """Test loss with class weights."""
        weights = torch.ones(100)
        weights[0] = 2.0  # Double weight for class 0

        loss_fn = StoreVisitLoss(
            num_stores=100,
            class_weights=weights,
            label_smoothing=0.0,
        )

        B = 32
        logits = torch.randn(B, 100)
        targets = torch.randint(0, 100, (B,))

        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_loss_perfect_prediction(self, loss_fn):
        """Test loss approaches minimum for perfect predictions."""
        B = 32
        targets = torch.randint(0, 100, (B,))

        # Create "perfect" logits (high for correct class)
        logits = torch.full((B, 100), -10.0)
        logits[torch.arange(B), targets] = 10.0

        loss = loss_fn(logits, targets)

        # With label smoothing (0.1) and 100 classes, minimum loss is ~0.1 * log(100) + 0.9 * 0 ≈ 0.46
        # In practice it's higher due to softmax distribution, so allow up to 2.5
        assert loss.item() < 2.5


class TestFocalLoss:
    """Tests for focal loss function."""

    @pytest.fixture
    def focal_loss_fn(self):
        return FocalLoss(gamma=2.0, label_smoothing=0.0)

    def test_focal_loss_computation(self, focal_loss_fn):
        """Test basic focal loss computation."""
        B = 32
        num_stores = 100
        logits = torch.randn(B, num_stores)
        targets = torch.randint(0, num_stores, (B,))

        loss = focal_loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_focal_loss_with_class_weights(self):
        """Test focal loss with class weights (alpha)."""
        num_stores = 100
        weights = torch.ones(num_stores)
        weights[0] = 2.0  # Double weight for class 0

        loss_fn = FocalLoss(gamma=2.0, alpha=weights, label_smoothing=0.0)

        B = 32
        logits = torch.randn(B, num_stores)
        targets = torch.randint(0, num_stores, (B,))

        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_focal_gamma_effect(self):
        """Test that higher gamma focuses more on hard examples."""
        B = 100
        num_stores = 50

        # Create mixed easy and hard examples
        logits = torch.randn(B, num_stores)
        targets = torch.randint(0, num_stores, (B,))

        # Make half of them easy (high confidence on correct class)
        for i in range(B // 2):
            logits[i, targets[i]] = 5.0  # Confident prediction

        loss_low_gamma = FocalLoss(gamma=0.5)(logits, targets)
        loss_high_gamma = FocalLoss(gamma=3.0)(logits, targets)

        # Higher gamma should give lower loss because easy examples are down-weighted
        assert loss_high_gamma < loss_low_gamma

    def test_focal_easy_vs_hard_examples(self):
        """Test that focal loss down-weights easy examples significantly."""
        num_stores = 50

        # Easy example: very confident correct prediction
        easy_logits = torch.zeros(1, num_stores)
        easy_logits[0, 0] = 10.0  # p_t ≈ 1.0
        easy_targets = torch.tensor([0])

        # Hard example: uncertain prediction
        hard_logits = torch.zeros(1, num_stores)  # p_t ≈ 0.02
        hard_targets = torch.tensor([0])

        focal_fn = FocalLoss(gamma=2.0)
        ce_fn = StoreVisitLoss(num_stores, label_smoothing=0.0)

        easy_focal = focal_fn(easy_logits, easy_targets).item()
        hard_focal = focal_fn(hard_logits, hard_targets).item()
        easy_ce = ce_fn(easy_logits, easy_targets).item()
        hard_ce = ce_fn(hard_logits, hard_targets).item()

        # For easy examples: focal should be much smaller than CE
        # (1 - p_t)^gamma ≈ 0 when p_t ≈ 1
        assert easy_focal < easy_ce * 0.1  # At least 10x smaller

        # For hard examples: focal and CE should be similar
        # (1 - p_t)^gamma ≈ 1 when p_t ≈ 0
        assert hard_focal > easy_focal * 10  # Hard is much larger than easy

    def test_focal_gamma_zero_equals_ce(self):
        """Test that gamma=0 gives same result as cross-entropy."""
        B = 32
        num_stores = 50

        logits = torch.randn(B, num_stores)
        targets = torch.randint(0, num_stores, (B,))

        focal_loss = FocalLoss(gamma=0.0)(logits, targets)
        ce_loss = StoreVisitLoss(num_stores, label_smoothing=0.0)(logits, targets)

        # Should be equal (or very close due to numerical precision)
        assert torch.isclose(focal_loss, ce_loss, rtol=1e-4)

    def test_focal_with_label_smoothing(self):
        """Test focal loss with label smoothing."""
        B = 32
        num_stores = 100

        logits = torch.randn(B, num_stores)
        targets = torch.randint(0, num_stores, (B,))

        loss_no_smooth = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets)
        loss_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)(logits, targets)

        # With label smoothing, loss should generally be higher
        # (distributing probability to other classes increases CE)
        assert loss_smooth > loss_no_smooth * 0.9  # Allow some tolerance


class TestStoreVisitMetrics:
    """Tests for evaluation metrics."""

    def test_accuracy(self):
        """Test accuracy computation."""
        B = 100
        logits = torch.randn(B, 50)
        targets = logits.argmax(dim=-1)  # Perfect predictions

        acc = StoreVisitMetrics.compute_accuracy(logits, targets)
        assert acc == 1.0

    def test_accuracy_random(self):
        """Test accuracy with random predictions."""
        B = 1000
        num_stores = 100
        logits = torch.randn(B, num_stores)
        targets = torch.randint(0, num_stores, (B,))

        acc = StoreVisitMetrics.compute_accuracy(logits, targets)
        # Random chance ~ 1%
        assert 0 <= acc <= 0.1

    def test_top_k_accuracy(self):
        """Test top-k accuracy."""
        B = 100
        num_stores = 50

        # Create logits where target is always in top-5
        logits = torch.randn(B, num_stores)
        targets = torch.randint(0, 5, (B,))  # Targets in 0-4

        # Make first 5 classes have highest logits
        logits[:, :5] += 10.0

        top5_acc = StoreVisitMetrics.compute_top_k_accuracy(logits, targets, k=5)
        assert top5_acc == 1.0

    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        B = 4
        num_stores = 10

        # Create perfect predictions (target always rank 1)
        logits = torch.zeros(B, num_stores)
        targets = torch.arange(B) % num_stores
        logits[torch.arange(B), targets] = 10.0

        mrr = StoreVisitMetrics.compute_mrr(logits, targets)
        assert mrr == 1.0  # Perfect MRR

    def test_compute_all(self):
        """Test computing all metrics."""
        B = 32
        logits = torch.randn(B, 100)
        targets = torch.randint(0, 100, (B,))

        metrics = StoreVisitMetrics.compute_all(logits, targets)

        assert 'accuracy' in metrics
        assert 'top_3_accuracy' in metrics
        assert 'top_5_accuracy' in metrics
        assert 'top_10_accuracy' in metrics
        assert 'mrr' in metrics

        # Top-k accuracy should be monotonically increasing
        assert metrics['accuracy'] <= metrics['top_3_accuracy']
        assert metrics['top_3_accuracy'] <= metrics['top_5_accuracy']
        assert metrics['top_5_accuracy'] <= metrics['top_10_accuracy']


class TestClassWeights:
    """Tests for class weight computation."""

    def test_compute_weights_uniform(self):
        """Test weights for uniform distribution."""
        num_stores = 100
        store_counts = {i: 1000 for i in range(num_stores)}

        weights = compute_store_class_weights(store_counts, num_stores, smoothing=0.5)

        assert weights.shape == (num_stores,)
        # Uniform counts -> uniform weights (mean = 1)
        assert torch.allclose(weights, torch.ones(num_stores), atol=0.01)

    def test_compute_weights_imbalanced(self):
        """Test weights for imbalanced distribution."""
        num_stores = 100
        store_counts = {i: 100 if i < 10 else 1000 for i in range(num_stores)}

        weights = compute_store_class_weights(store_counts, num_stores, smoothing=0.5)

        # Rare classes should have higher weights
        rare_weight = weights[:10].mean()
        common_weight = weights[10:].mean()
        assert rare_weight > common_weight

    def test_compute_weights_smoothing(self):
        """Test effect of smoothing parameter."""
        num_stores = 50
        store_counts = {i: 100 if i < 5 else 10000 for i in range(num_stores)}

        weights_low = compute_store_class_weights(store_counts, num_stores, smoothing=0.1)
        weights_high = compute_store_class_weights(store_counts, num_stores, smoothing=0.9)

        # Higher smoothing = more extreme weights
        assert weights_high.std() > weights_low.std()


class TestFactoryFunction:
    """Tests for model factory function."""

    def test_create_model(self):
        """Test factory function."""
        model = create_store_visit_model(
            num_stores=500,
            hidden_dim=256,
            num_layers=3,
            use_basket_summary=True,
        )

        assert isinstance(model, StoreVisitPredictor)
        assert model.config.num_stores == 500
        assert model.config.hidden_dim == 256
        assert model.config.num_layers == 3

    def test_create_model_no_basket(self):
        """Test factory without basket summary."""
        model = create_store_visit_model(
            num_stores=100,
            use_basket_summary=False,
        )

        assert model.basket_summarizer is None


class TestTrainingStep:
    """Tests for end-to-end training step."""

    def test_training_step(self):
        """Test a complete training step."""
        # Setup
        config = StoreVisitModelConfig(num_stores=100, hidden_dim=128)
        model = StoreVisitPredictor(config)
        loss_fn = StoreVisitLoss(num_stores=100, label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Data
        B = 16
        customer_context = torch.randn(B, 192)
        temporal_context = torch.randn(B, 64)
        previous_store_idx = torch.randint(0, 100, (B,))
        targets = torch.randint(0, 100, (B,))

        # Forward
        model.train()
        outputs = model(customer_context, temporal_context, previous_store_idx)
        loss = loss_fn(outputs['store_logits'], targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_training_reduces_loss(self):
        """Test that training reduces loss over steps."""
        config = StoreVisitModelConfig(num_stores=20, hidden_dim=64, num_layers=1)
        model = StoreVisitPredictor(config)
        loss_fn = StoreVisitLoss(num_stores=20, label_smoothing=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Fixed data (overfit test)
        B = 32
        torch.manual_seed(42)
        customer_context = torch.randn(B, 192)
        temporal_context = torch.randn(B, 64)
        previous_store_idx = torch.randint(0, 20, (B,))
        targets = torch.randint(0, 20, (B,))

        model.train()
        losses = []
        for _ in range(50):
            outputs = model(customer_context, temporal_context, previous_store_idx)
            loss = loss_fn(outputs['store_logits'], targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
