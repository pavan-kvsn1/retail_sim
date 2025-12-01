"""
Unit tests for World Model architecture.

Tests:
- WorldModelConfig: Configuration validation
- ContextFusion: Dense context projection
- ProductSequenceFusion: Sequence feature projection
- MambaBlock/Encoder: State-space model computation
- TransformerDecoder: Cross-attention mechanism
- OutputHeads: Multi-task prediction heads
- WorldModel: End-to-end forward pass
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.model import (
    WorldModelConfig,
    ContextFusion,
    ProductSequenceFusion,
    MambaBlock,
    MambaEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
    OutputHeads,
    WorldModel,
    create_world_model,
)


class TestWorldModelConfig:
    """Tests for WorldModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorldModelConfig()

        assert config.d_model == 512
        assert config.d_context == 400
        assert config.mamba_num_layers == 4
        assert config.decoder_num_layers == 2
        assert config.n_products == 5003

    def test_custom_config(self):
        """Test custom configuration."""
        config = WorldModelConfig(
            d_model=256,
            n_products=1000,
            mamba_num_layers=2
        )

        assert config.d_model == 256
        assert config.n_products == 1000
        assert config.mamba_num_layers == 2


class TestContextFusion:
    """Tests for ContextFusion layer."""

    def test_context_fusion_output_shape(self):
        """Test context fusion produces correct shape."""
        config = WorldModelConfig()
        fusion = ContextFusion(config)

        dense_context = torch.randn(8, 400)
        output = fusion(dense_context)

        assert output.shape == (8, 512)

    def test_context_fusion_different_batch_sizes(self):
        """Test context fusion with various batch sizes."""
        config = WorldModelConfig()
        fusion = ContextFusion(config)

        for batch_size in [1, 16, 64]:
            dense_context = torch.randn(batch_size, 400)
            output = fusion(dense_context)
            assert output.shape == (batch_size, 512)

    def test_context_fusion_gradient_flow(self):
        """Test gradients flow through context fusion."""
        config = WorldModelConfig()
        fusion = ContextFusion(config)

        dense_context = torch.randn(4, 400, requires_grad=True)
        output = fusion(dense_context)
        loss = output.sum()
        loss.backward()

        assert dense_context.grad is not None


class TestProductSequenceFusion:
    """Tests for ProductSequenceFusion layer."""

    def test_product_fusion_output_shape(self):
        """Test product fusion produces correct shape."""
        config = WorldModelConfig()
        fusion = ProductSequenceFusion(config)

        product_emb = torch.randn(8, 20, 256)
        price_feat = torch.randn(8, 20, 64)
        output = fusion(product_emb, price_feat)

        assert output.shape == (8, 20, 512)

    def test_product_fusion_variable_length(self):
        """Test product fusion with variable sequence lengths."""
        config = WorldModelConfig()
        fusion = ProductSequenceFusion(config)

        for seq_len in [5, 20, 50]:
            product_emb = torch.randn(4, seq_len, 256)
            price_feat = torch.randn(4, seq_len, 64)
            output = fusion(product_emb, price_feat)
            assert output.shape == (4, seq_len, 512)

    def test_product_fusion_positional_encoding(self):
        """Test positional encoding is applied."""
        config = WorldModelConfig()
        fusion = ProductSequenceFusion(config)

        product_emb = torch.zeros(1, 10, 256)
        price_feat = torch.zeros(1, 10, 64)

        output = fusion(product_emb, price_feat)

        # With zero input, output should be positional encoding
        assert not torch.allclose(output[0, 0], output[0, 1])


class TestMambaBlock:
    """Tests for MambaBlock."""

    def test_mamba_block_output_shape(self):
        """Test Mamba block produces correct shape."""
        config = WorldModelConfig()
        block = MambaBlock(config)

        x = torch.randn(4, 20, 512)
        output = block(x)

        assert output.shape == x.shape

    def test_mamba_block_residual(self):
        """Test Mamba block has residual connection."""
        config = WorldModelConfig()
        block = MambaBlock(config)

        x = torch.randn(4, 20, 512)
        output = block(x)

        # Output should be different from input but not too far
        assert not torch.allclose(output, x)

    def test_mamba_block_gradient_flow(self):
        """Test gradients flow through Mamba block."""
        config = WorldModelConfig()
        block = MambaBlock(config)

        x = torch.randn(4, 20, 512, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestMambaEncoder:
    """Tests for MambaEncoder."""

    def test_mamba_encoder_output_shape(self):
        """Test Mamba encoder produces correct shape."""
        config = WorldModelConfig()
        encoder = MambaEncoder(config)

        x = torch.randn(4, 30, 512)
        output = encoder(x)

        assert output.shape == x.shape

    def test_mamba_encoder_num_layers(self):
        """Test encoder has correct number of layers."""
        config = WorldModelConfig(mamba_num_layers=4)
        encoder = MambaEncoder(config)

        assert len(encoder.layers) == 4


class TestTransformerDecoderLayer:
    """Tests for TransformerDecoderLayer."""

    def test_decoder_layer_output_shape(self):
        """Test decoder layer produces correct shape."""
        config = WorldModelConfig()
        layer = TransformerDecoderLayer(config)

        x = torch.randn(4, 10, 512)
        encoder_out = torch.randn(4, 20, 512)
        output = layer(x, encoder_out)

        assert output.shape == x.shape

    def test_decoder_layer_cross_attention(self):
        """Test decoder layer uses cross-attention."""
        config = WorldModelConfig()
        layer = TransformerDecoderLayer(config)

        x = torch.randn(4, 10, 512)
        encoder_out = torch.randn(4, 20, 512)

        # With different encoder outputs, decoder output should differ
        output1 = layer(x, encoder_out)
        output2 = layer(x, encoder_out * 2)

        assert not torch.allclose(output1, output2)


class TestTransformerDecoder:
    """Tests for TransformerDecoder."""

    def test_decoder_output_shape(self):
        """Test decoder produces correct shape."""
        config = WorldModelConfig()
        decoder = TransformerDecoder(config)

        x = torch.randn(4, 10, 512)
        encoder_out = torch.randn(4, 20, 512)
        output = decoder(x, encoder_out)

        assert output.shape == x.shape

    def test_decoder_num_layers(self):
        """Test decoder has correct number of layers."""
        config = WorldModelConfig(decoder_num_layers=2)
        decoder = TransformerDecoder(config)

        assert len(decoder.layers) == 2

    def test_decoder_causal_mask(self):
        """Test decoder generates causal mask correctly."""
        config = WorldModelConfig()
        decoder = TransformerDecoder(config)

        mask = decoder._generate_causal_mask(5, torch.device('cpu'))

        # Upper triangle should be -inf
        assert mask[0, 1] == float('-inf')
        assert mask[0, 0] == 0


class TestOutputHeads:
    """Tests for OutputHeads."""

    def test_output_heads_with_positions(self):
        """Test output heads with specified masked positions."""
        config = WorldModelConfig()
        heads = OutputHeads(config)

        decoder_out = torch.randn(4, 20, 512)
        masked_positions = torch.randint(0, 20, (4, 3))

        logits, aux_logits = heads(decoder_out, masked_positions)

        assert logits.shape == (4, 3, 5003)
        assert 'basket_size' in aux_logits
        assert aux_logits['basket_size'].shape == (4, 4)

    def test_output_heads_without_positions(self):
        """Test output heads without masked positions."""
        config = WorldModelConfig()
        heads = OutputHeads(config)

        decoder_out = torch.randn(4, 20, 512)

        logits, aux_logits = heads(decoder_out, None)

        assert logits.shape == (4, 20, 5003)

    def test_output_heads_all_auxiliary(self):
        """Test all auxiliary heads are present."""
        config = WorldModelConfig()
        heads = OutputHeads(config)

        decoder_out = torch.randn(4, 10, 512)
        _, aux_logits = heads(decoder_out, None)

        assert 'basket_size' in aux_logits
        assert 'price_sensitivity' in aux_logits
        assert 'mission_type' in aux_logits
        assert 'mission_focus' in aux_logits


class TestWorldModel:
    """Tests for complete WorldModel."""

    def test_world_model_forward(self):
        """Test World Model forward pass."""
        model = WorldModel()

        B, S = 4, 20
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        masked_positions = torch.randint(0, S, (B, 3))

        logits, aux_logits, encoder_out = model(
            dense_context, product_emb, price_feat,
            attention_mask, masked_positions
        )

        assert logits.shape == (B, 3, 5003)
        assert encoder_out.shape[0] == B
        assert encoder_out.shape[2] == 512

    def test_world_model_without_masked_positions(self):
        """Test World Model without masked positions."""
        model = WorldModel()

        B, S = 4, 20
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)

        logits, aux_logits, encoder_out = model(
            dense_context, product_emb, price_feat,
            attention_mask, None
        )

        # Full sequence logits
        assert logits.shape == (B, S + 1, 5003)  # +1 for CLS token

    def test_world_model_parameter_count(self):
        """Test model has expected parameter count."""
        model = WorldModel()

        n_params = model.num_parameters
        # Should be around 15-25M parameters
        assert 10_000_000 < n_params < 50_000_000

    def test_world_model_gradient_flow(self):
        """Test gradients flow through entire model."""
        model = WorldModel()

        B, S = 2, 10
        dense_context = torch.randn(B, 400, requires_grad=True)
        product_emb = torch.randn(B, S, 256, requires_grad=True)
        price_feat = torch.randn(B, S, 64, requires_grad=True)
        attention_mask = torch.ones(B, S)
        masked_positions = torch.randint(0, S, (B, 2))

        logits, aux_logits, encoder_out = model(
            dense_context, product_emb, price_feat,
            attention_mask, masked_positions
        )

        loss = logits.sum() + aux_logits['basket_size'].sum()
        loss.backward()

        assert dense_context.grad is not None
        assert product_emb.grad is not None
        assert price_feat.grad is not None

    def test_world_model_encoder_output(self):
        """Test get_encoder_output method."""
        model = WorldModel()

        B, S = 4, 15
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)

        encoder_out = model.get_encoder_output(
            dense_context, product_emb, price_feat, attention_mask
        )

        assert encoder_out.shape == (B, S + 1, 512)  # +1 for CLS

    def test_create_world_model_factory(self):
        """Test factory function creates model correctly."""
        model = create_world_model(
            n_products=1000,
            d_model=256,
            mamba_layers=2,
            decoder_layers=1
        )

        assert isinstance(model, WorldModel)
        assert model.config.n_products == 1000
        assert model.config.d_model == 256


class TestWorldModelIntegration:
    """Integration tests for World Model with realistic data."""

    def test_batch_processing(self):
        """Test processing batch of varied samples."""
        model = WorldModel()
        model.eval()

        # Simulate batch with different sequence lengths
        B = 8
        max_S = 30
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, max_S, 256)
        price_feat = torch.randn(B, max_S, 64)

        # Variable attention masks
        attention_mask = torch.zeros(B, max_S)
        for i in range(B):
            seq_len = np.random.randint(5, max_S)
            attention_mask[i, :seq_len] = 1

        masked_positions = torch.randint(0, max_S, (B, 3))

        with torch.no_grad():
            logits, aux_logits, _ = model(
                dense_context, product_emb, price_feat,
                attention_mask, masked_positions
            )

        assert logits.shape == (B, 3, 5003)

    def test_deterministic_output(self):
        """Test model produces deterministic output in eval mode."""
        model = WorldModel()
        model.eval()

        B, S = 2, 10
        dense_context = torch.randn(B, 400)
        product_emb = torch.randn(B, S, 256)
        price_feat = torch.randn(B, S, 64)
        attention_mask = torch.ones(B, S)
        masked_positions = torch.randint(0, S, (B, 2))

        with torch.no_grad():
            out1, _, _ = model(
                dense_context, product_emb, price_feat,
                attention_mask, masked_positions
            )
            out2, _, _ = model(
                dense_context, product_emb, price_feat,
                attention_mask, masked_positions
            )

        assert torch.allclose(out1, out2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
