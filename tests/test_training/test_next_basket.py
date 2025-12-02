"""
Unit tests for Next-Basket Prediction Pipeline.

Tests:
- NextBasketModelConfig: Configuration validation
- NextBasketWorldModel: Forward pass, output shapes, gradients
- NextBasketLoss: Loss computation, focal weighting
- NextBasketMetrics: Precision, Recall, F1, NDCG
- Integration: End-to-end training step
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.model_next_basket import (
    NextBasketModelConfig,
    NextBasketWorldModel,
    InputEncoder,
    BasketPredictor,
    create_next_basket_model,
)
from src.training.losses_next_basket import (
    FocalBCELoss,
    NextBasketLoss,
    NextBasketLossConfig,
    NextBasketMetrics,
)


class TestNextBasketModelConfig:
    """Tests for NextBasketModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NextBasketModelConfig()

        assert config.vocab_size == 5000
        assert config.product_dim == 256
        assert config.hidden_dim == 512
        assert config.encoder_layers == 4
        assert config.decoder_layers == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = NextBasketModelConfig(
            vocab_size=1000,
            hidden_dim=256,
            encoder_layers=2
        )

        assert config.vocab_size == 1000
        assert config.hidden_dim == 256
        assert config.encoder_layers == 2


class TestInputEncoder:
    """Tests for InputEncoder."""

    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape."""
        config = NextBasketModelConfig()
        encoder = InputEncoder(config)

        B, S = 4, 20
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        context = torch.randn(B, 400)

        output = encoder(product_emb, price_feat, attention_mask, context)

        assert output.shape == (B, config.hidden_dim)

    def test_encoder_variable_sequence_length(self):
        """Test encoder handles variable sequence lengths."""
        config = NextBasketModelConfig()
        encoder = InputEncoder(config)

        B = 4
        context = torch.randn(B, 400)

        for seq_len in [5, 20, 50]:
            product_emb = torch.randn(B, seq_len, 256)
            price_feat = torch.randn(B, seq_len, 64)
            attention_mask = torch.ones(B, seq_len)

            output = encoder(product_emb, price_feat, attention_mask, context)
            assert output.shape == (B, config.hidden_dim)

    def test_encoder_gradient_flow(self):
        """Test gradients flow through encoder."""
        config = NextBasketModelConfig()
        encoder = InputEncoder(config)

        B, S = 2, 10
        product_emb = torch.randn(B, S, 256, requires_grad=True)
        price_feat = torch.randn(B, S, 64, requires_grad=True)
        attention_mask = torch.ones(B, S)
        context = torch.randn(B, 400, requires_grad=True)

        output = encoder(product_emb, price_feat, attention_mask, context)
        loss = output.sum()
        loss.backward()

        assert product_emb.grad is not None
        assert price_feat.grad is not None
        assert context.grad is not None


class TestBasketPredictor:
    """Tests for BasketPredictor."""

    def test_predictor_output_shape(self):
        """Test predictor produces correct output shapes."""
        config = NextBasketModelConfig(vocab_size=5000)
        predictor = BasketPredictor(config)

        B = 4
        encoded = torch.randn(B, config.hidden_dim)

        outputs = predictor(encoded)

        assert outputs['product_logits'].shape == (B, 5000)
        assert outputs['basket_size'].shape == (B, config.num_basket_size)
        assert outputs['mission_type'].shape == (B, config.num_mission_types)

    def test_predictor_all_outputs(self):
        """Test predictor returns all required outputs."""
        config = NextBasketModelConfig()
        predictor = BasketPredictor(config)

        encoded = torch.randn(4, config.hidden_dim)
        outputs = predictor(encoded)

        required_keys = ['product_logits', 'basket_size', 'mission_type',
                        'mission_focus', 'price_sensitivity']
        for key in required_keys:
            assert key in outputs


class TestNextBasketWorldModel:
    """Tests for complete NextBasketWorldModel."""

    def test_model_forward(self):
        """Test model forward pass."""
        config = NextBasketModelConfig(vocab_size=5000)
        model = NextBasketWorldModel(config)

        B, S = 4, 20
        outputs = model(
            input_embeddings=torch.randn(B, S, 256),
            input_price_features=torch.randn(B, S, 64),
            input_attention_mask=torch.ones(B, S),
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            store_context=torch.randn(B, 96),
            trip_context=torch.randn(B, 48),
        )

        assert outputs['product_logits'].shape == (B, 5000)

    def test_model_parameter_count(self):
        """Test model has reasonable parameter count."""
        config = NextBasketModelConfig(vocab_size=5000)
        model = NextBasketWorldModel(config)

        n_params = sum(p.numel() for p in model.parameters())
        # Should be around 10-20M parameters
        assert 5_000_000 < n_params < 50_000_000

    def test_model_gradient_flow(self):
        """Test gradients flow through entire model."""
        config = NextBasketModelConfig(vocab_size=1000)
        model = NextBasketWorldModel(config)

        B, S = 2, 10
        input_emb = torch.randn(B, S, 256, requires_grad=True)
        price_feat = torch.randn(B, S, 64, requires_grad=True)
        customer_ctx = torch.randn(B, 192, requires_grad=True)

        outputs = model(
            input_embeddings=input_emb,
            input_price_features=price_feat,
            input_attention_mask=torch.ones(B, S),
            customer_context=customer_ctx,
            temporal_context=torch.randn(B, 64),
            store_context=torch.randn(B, 96),
            trip_context=torch.randn(B, 48),
        )

        loss = outputs['product_logits'].sum()
        loss.backward()

        assert input_emb.grad is not None
        assert price_feat.grad is not None
        assert customer_ctx.grad is not None

    def test_model_predict_basket(self):
        """Test predict_basket method."""
        config = NextBasketModelConfig(vocab_size=1000)
        model = NextBasketWorldModel(config)
        model.eval()

        B, S = 4, 15
        predictions = model.predict_basket(
            input_embeddings=torch.randn(B, S, 256),
            input_price_features=torch.randn(B, S, 64),
            input_attention_mask=torch.ones(B, S),
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            store_context=torch.randn(B, 96),
            trip_context=torch.randn(B, 48),
            top_k=10,
        )

        # Should return exactly 10 products per sample
        assert predictions.shape == (B, 1000)
        assert torch.allclose(predictions.sum(dim=1), torch.tensor([10.0] * B))

    def test_model_get_probabilities(self):
        """Test get_product_probabilities for RL."""
        config = NextBasketModelConfig(vocab_size=1000)
        model = NextBasketWorldModel(config)
        model.eval()

        B, S = 2, 10
        probs = model.get_product_probabilities(
            input_embeddings=torch.randn(B, S, 256),
            input_price_features=torch.randn(B, S, 64),
            input_attention_mask=torch.ones(B, S),
            customer_context=torch.randn(B, 192),
            temporal_context=torch.randn(B, 64),
            store_context=torch.randn(B, 96),
            trip_context=torch.randn(B, 48),
        )

        # Probabilities should be in [0, 1]
        assert probs.shape == (B, 1000)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_create_model_factory(self):
        """Test factory function."""
        model = create_next_basket_model(
            vocab_size=500,
            hidden_dim=256,
            encoder_layers=2,
        )

        assert isinstance(model, NextBasketWorldModel)
        assert model.config.vocab_size == 500
        assert model.config.hidden_dim == 256


class TestFocalBCELoss:
    """Tests for FocalBCELoss."""

    def test_focal_loss_output(self):
        """Test focal loss computation."""
        loss_fn = FocalBCELoss(gamma=2.0, alpha=0.25)

        B, V = 4, 1000
        logits = torch.randn(B, V)
        targets = torch.zeros(B, V)
        # Sparse targets
        targets[0, [1, 2, 3]] = 1
        targets[1, [10, 20]] = 1

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_focal_loss_gradient(self):
        """Test focal loss gradients."""
        loss_fn = FocalBCELoss()

        logits = torch.randn(4, 100, requires_grad=True)
        targets = torch.zeros(4, 100)
        targets[0, :5] = 1

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None

    def test_focal_vs_bce(self):
        """Test focal loss differs from standard BCE."""
        focal_loss = FocalBCELoss(gamma=2.0)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        logits = torch.randn(4, 100)
        targets = torch.zeros(4, 100)
        targets[0, :5] = 1

        fl = focal_loss(logits, targets)
        bl = bce_loss(logits, targets)

        # Focal loss should generally be lower (down-weights easy examples)
        assert fl.item() != bl.item()


class TestNextBasketLoss:
    """Tests for combined NextBasketLoss."""

    def test_loss_computation(self):
        """Test combined loss computation."""
        loss_fn = NextBasketLoss()

        B, V = 4, 5000
        outputs = {
            'product_logits': torch.randn(B, V),
            'basket_size': torch.randn(B, 4),
            'mission_type': torch.randn(B, 5),
            'mission_focus': torch.randn(B, 6),
            'price_sensitivity': torch.randn(B, 4),
        }

        targets = torch.zeros(B, V)
        targets[0, [1, 2, 3]] = 1
        targets[1, [10, 20, 30]] = 1

        aux_labels = {
            'basket_size': torch.tensor([1, 2, 1, 2]),
            'mission_type': torch.tensor([1, 2, 3, 1]),
            'mission_focus': torch.tensor([1, 2, 3, 4]),
            'price_sensitivity': torch.tensor([1, 2, 1, 2]),
        }

        total_loss, loss_dict = loss_fn(outputs, targets, aux_labels)

        assert total_loss.dim() == 0
        assert 'product' in loss_dict
        assert 'auxiliary' in loss_dict
        assert 'total' in loss_dict

    def test_loss_components(self):
        """Test all loss components are computed."""
        loss_fn = NextBasketLoss()

        B, V = 2, 100
        outputs = {
            'product_logits': torch.randn(B, V),
            'basket_size': torch.randn(B, 4),
            'mission_type': torch.randn(B, 5),
            'mission_focus': torch.randn(B, 6),
            'price_sensitivity': torch.randn(B, 4),
        }

        targets = torch.zeros(B, V)
        targets[:, :5] = 1

        aux_labels = {
            'basket_size': torch.tensor([1, 2]),
            'mission_type': torch.tensor([1, 2]),
            'mission_focus': torch.tensor([1, 2]),
            'price_sensitivity': torch.tensor([1, 2]),
        }

        _, loss_dict = loss_fn(outputs, targets, aux_labels)

        assert 'basket_size' in loss_dict
        assert 'mission_type' in loss_dict
        assert 'mission_focus' in loss_dict
        assert 'price_sensitivity' in loss_dict


class TestNextBasketMetrics:
    """Tests for NextBasketMetrics."""

    def test_precision_at_k(self):
        """Test precision@k computation."""
        B, V = 4, 100

        # Create predictions where top-k DEFINITELY overlap with targets
        predictions = torch.zeros(B, V)
        predictions[:, :10] = 10.0  # Very high scores for first 10

        targets = torch.zeros(B, V)
        targets[:, :10] = 1  # First 10 are relevant

        precision = NextBasketMetrics.precision_at_k(predictions, targets, k=10)

        # All top-10 predictions should be correct
        assert torch.allclose(precision, torch.ones(B))

    def test_recall_at_k(self):
        """Test recall@k computation."""
        B, V = 4, 100

        predictions = torch.zeros(B, V)
        predictions[:, :10] = torch.rand(B, 10) + 1  # High scores

        targets = torch.zeros(B, V)
        targets[:, :20] = 1  # 20 relevant items

        recall = NextBasketMetrics.recall_at_k(predictions, targets, k=10)

        # Should recall 10/20 = 0.5
        assert torch.allclose(recall, torch.tensor([0.5] * B))

    def test_f1_at_k(self):
        """Test F1@k computation."""
        B, V = 2, 50

        predictions = torch.randn(B, V)
        targets = torch.zeros(B, V)
        targets[:, :5] = 1

        f1 = NextBasketMetrics.f1_at_k(predictions, targets, k=10)

        assert f1.shape == (B,)
        assert (f1 >= 0).all() and (f1 <= 1).all()

    def test_hit_rate_at_k(self):
        """Test hit rate@k computation."""
        B, V = 4, 100

        # Predictions where first sample has no overlap
        predictions = torch.zeros(B, V)
        predictions[0, 50:60] = 1  # High scores far from targets
        predictions[1:, :10] = 1   # High scores near targets

        targets = torch.zeros(B, V)
        targets[:, :5] = 1  # First 5 relevant

        hit_rate = NextBasketMetrics.hit_rate_at_k(predictions, targets, k=10)

        # First sample should miss, others should hit
        assert hit_rate[0] == 0.0
        assert (hit_rate[1:] == 1.0).all()

    def test_ndcg_at_k(self):
        """Test NDCG@k computation."""
        B, V = 4, 100

        predictions = torch.randn(B, V)
        targets = torch.zeros(B, V)
        targets[:, :10] = 1

        ndcg = NextBasketMetrics.ndcg_at_k(predictions, targets, k=10)

        assert ndcg.shape == (B,)
        assert (ndcg >= 0).all() and (ndcg <= 1).all()

    def test_compute_all_metrics(self):
        """Test compute_all returns all metrics."""
        B, V = 4, 100

        predictions = torch.randn(B, V)
        targets = torch.zeros(B, V)
        targets[:, :10] = 1

        metrics = NextBasketMetrics.compute_all(predictions, targets, k_values=[5, 10])

        expected_keys = [
            'precision@5', 'recall@5', 'f1@5', 'hit_rate@5', 'ndcg@5',
            'precision@10', 'recall@10', 'f1@10', 'hit_rate@10', 'ndcg@10',
        ]
        for key in expected_keys:
            assert key in metrics


class TestNextBasketIntegration:
    """Integration tests for next-basket prediction."""

    def test_training_step(self):
        """Test a complete training step."""
        # Model
        config = NextBasketModelConfig(vocab_size=1000)
        model = NextBasketWorldModel(config)

        # Loss
        loss_fn = NextBasketLoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Fake batch
        B, S = 4, 15
        input_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        customer_ctx = torch.randn(B, 192)
        temporal_ctx = torch.randn(B, 64)
        store_ctx = torch.randn(B, 96)
        trip_ctx = torch.randn(B, 48)

        targets = torch.zeros(B, 1000)
        for i in range(B):
            targets[i, torch.randint(0, 1000, (10,))] = 1

        aux_labels = {
            'basket_size': torch.randint(1, 4, (B,)),
            'mission_type': torch.randint(1, 5, (B,)),
            'mission_focus': torch.randint(1, 6, (B,)),
            'price_sensitivity': torch.randint(1, 4, (B,)),
        }

        # Forward
        outputs = model(
            input_emb, price_feat, attention_mask,
            customer_ctx, temporal_ctx, store_ctx, trip_ctx
        )

        # Loss
        loss, loss_dict = loss_fn(outputs, targets, aux_labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert 'product' in loss_dict

    def test_evaluation_step(self):
        """Test a complete evaluation step."""
        config = NextBasketModelConfig(vocab_size=500)
        model = NextBasketWorldModel(config)
        model.eval()

        B, S = 8, 20
        with torch.no_grad():
            outputs = model(
                input_embeddings=torch.randn(B, S, 256),
                input_price_features=torch.randn(B, S, 64),
                input_attention_mask=torch.ones(B, S),
                customer_context=torch.randn(B, 192),
                temporal_context=torch.randn(B, 64),
                store_context=torch.randn(B, 96),
                trip_context=torch.randn(B, 48),
            )

            probs = torch.sigmoid(outputs['product_logits'])

            targets = torch.zeros(B, 500)
            targets[:, :15] = 1

            metrics = NextBasketMetrics.compute_all(probs, targets)

        assert 'f1@10' in metrics
        assert metrics['f1@10'] >= 0

    def test_model_determinism(self):
        """Test model produces deterministic output in eval mode."""
        config = NextBasketModelConfig(vocab_size=500)
        model = NextBasketWorldModel(config)
        model.eval()

        B, S = 2, 10
        input_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        mask = torch.ones(B, S)
        customer = torch.randn(B, 192)
        temporal = torch.randn(B, 64)
        store = torch.randn(B, 96)
        trip = torch.randn(B, 48)

        with torch.no_grad():
            out1 = model(input_emb, price_feat, mask, customer, temporal, store, trip)
            out2 = model(input_emb, price_feat, mask, customer, temporal, store, trip)

        assert torch.allclose(out1['product_logits'], out2['product_logits'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
