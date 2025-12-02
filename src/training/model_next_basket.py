"""
World Model for Next-Basket Prediction.

Architecture:
1. Encoder: Process customer context + input basket (time t)
2. Decoder: Generate next basket (time t+1) as multi-label prediction

Key differences from masked prediction model:
- Output is multi-label (sigmoid) not single-token (softmax)
- No masking - full input basket visible
- Predicts entire next basket at once (set prediction)

For RL/simulation: Given state (history, context), predict action distribution
(probability of purchasing each product in next trip).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class NextBasketModelConfig:
    """Configuration for NextBasketWorldModel."""
    # Dimensions
    vocab_size: int = 5000
    product_dim: int = 256
    context_dim: int = 400  # T1(192) + T3(64) + T5(96) + T6(48)
    hidden_dim: int = 512

    # Encoder
    encoder_layers: int = 4
    encoder_heads: int = 8
    encoder_ff_dim: int = 1024

    # Decoder
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_ff_dim: int = 1024

    # Regularization
    dropout: float = 0.1

    # Output heads
    num_mission_types: int = 5
    num_mission_focus: int = 6
    num_price_sens: int = 4
    num_basket_size: int = 4


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class InputEncoder(nn.Module):
    """
    Encodes input basket + context into a latent representation.

    Uses Transformer encoder to process the input basket sequence,
    then combines with context for a rich representation.
    """

    def __init__(self, config: NextBasketModelConfig):
        super().__init__()
        self.config = config

        # Project product embeddings to hidden dim
        self.product_proj = nn.Linear(config.product_dim + 64, config.hidden_dim)  # +64 for price

        # Context projection
        self.context_proj = nn.Linear(config.context_dim, config.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim, dropout=config.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        # Disable nested tensor optimization - not supported on MPS (Apple Silicon)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )

        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)

        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def forward(
        self,
        product_embeddings: torch.Tensor,   # [B, S, 256]
        price_features: torch.Tensor,       # [B, S, 64]
        attention_mask: torch.Tensor,       # [B, S]
        context: torch.Tensor,              # [B, 400]
    ) -> torch.Tensor:
        """
        Encode input basket and context.

        Returns:
            [B, hidden_dim] - encoded representation
        """
        B, S, _ = product_embeddings.shape

        # Combine product and price features
        product_input = torch.cat([product_embeddings, price_features], dim=-1)  # [B, S, 320]
        product_hidden = self.product_proj(product_input)  # [B, S, hidden]

        # Add positional encoding
        product_hidden = self.pos_encoder(product_hidden)

        # Create attention mask for transformer (True = ignore)
        src_key_padding_mask = (attention_mask == 0)

        # Encode with transformer
        if self._gradient_checkpointing and self.training:
            encoded = torch.utils.checkpoint.checkpoint(
                self.transformer,
                product_hidden,
                None,  # mask
                src_key_padding_mask,
                use_reentrant=False
            )
        else:
            encoded = self.transformer(product_hidden, src_key_padding_mask=src_key_padding_mask)

        # Pool over sequence (mean of valid positions)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
        pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Combine with context
        context_hidden = self.context_proj(context)  # [B, hidden]
        combined = self.norm(pooled + context_hidden)

        return combined


class BasketPredictor(nn.Module):
    """
    Predicts next basket as multi-label classification.

    Output: probability for each product being in the next basket.
    This is a SET prediction (not sequence) - order doesn't matter.
    """

    def __init__(self, config: NextBasketModelConfig):
        super().__init__()
        self.config = config

        # MLP to expand representation
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.decoder_ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_ff_dim, config.decoder_ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Output projection to vocabulary
        self.output_proj = nn.Linear(config.decoder_ff_dim, config.vocab_size)

        # Auxiliary heads
        self.basket_size_head = nn.Linear(config.hidden_dim, config.num_basket_size)
        self.mission_type_head = nn.Linear(config.hidden_dim, config.num_mission_types)
        self.mission_focus_head = nn.Linear(config.hidden_dim, config.num_mission_focus)
        self.price_sens_head = nn.Linear(config.hidden_dim, config.num_price_sens)

    def forward(
        self,
        encoded: torch.Tensor,  # [B, hidden_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next basket.

        Returns:
            Dict with:
            - 'product_logits': [B, vocab_size] - logits for each product
            - 'basket_size': [B, 4] - logits for basket size class
            - 'mission_type': [B, 5] - logits for mission type
            - 'mission_focus': [B, 6] - logits for mission focus
            - 'price_sensitivity': [B, 4] - logits for price sensitivity
        """
        # Main prediction
        hidden = self.mlp(encoded)
        product_logits = self.output_proj(hidden)

        # Auxiliary predictions
        return {
            'product_logits': product_logits,
            'basket_size': self.basket_size_head(encoded),
            'mission_type': self.mission_type_head(encoded),
            'mission_focus': self.mission_focus_head(encoded),
            'price_sensitivity': self.price_sens_head(encoded),
        }


class NextBasketWorldModel(nn.Module):
    """
    World Model for next-basket prediction.

    Given:
    - Customer history (encoded in customer_context)
    - Current basket at time t
    - Context for time t+1 (store, temporal)

    Predicts:
    - Products in next basket (time t+1) as multi-label
    - Auxiliary: basket size, mission type, etc.
    """

    def __init__(self, config: NextBasketModelConfig):
        super().__init__()
        self.config = config

        self.encoder = InputEncoder(config)
        self.predictor = BasketPredictor(config)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.encoder.gradient_checkpointing_enable()

    def forward(
        self,
        input_embeddings: torch.Tensor,     # [B, S, 256]
        input_price_features: torch.Tensor, # [B, S, 64]
        input_attention_mask: torch.Tensor, # [B, S]
        customer_context: torch.Tensor,     # [B, 192]
        temporal_context: torch.Tensor,     # [B, 64]
        store_context: torch.Tensor,        # [B, 96]
        trip_context: torch.Tensor,         # [B, 48]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for next-basket prediction.

        Returns:
            Dict with product_logits and auxiliary predictions
        """
        # Concatenate context vectors
        context = torch.cat([
            customer_context,
            temporal_context,
            store_context,
            trip_context,
        ], dim=-1)  # [B, 400]

        # Encode input
        encoded = self.encoder(
            input_embeddings,
            input_price_features,
            input_attention_mask,
            context,
        )

        # Predict next basket
        outputs = self.predictor(encoded)

        return outputs

    def predict_basket(
        self,
        input_embeddings: torch.Tensor,
        input_price_features: torch.Tensor,
        input_attention_mask: torch.Tensor,
        customer_context: torch.Tensor,
        temporal_context: torch.Tensor,
        store_context: torch.Tensor,
        trip_context: torch.Tensor,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict next basket products.

        Args:
            threshold: Probability threshold for including product
            top_k: If set, return top-k products instead of threshold

        Returns:
            [B, vocab_size] - binary predictions or probabilities
        """
        with torch.no_grad():
            outputs = self.forward(
                input_embeddings,
                input_price_features,
                input_attention_mask,
                customer_context,
                temporal_context,
                store_context,
                trip_context,
            )

            probs = torch.sigmoid(outputs['product_logits'])

            if top_k is not None:
                # Return top-k as binary
                _, top_indices = probs.topk(top_k, dim=-1)
                predictions = torch.zeros_like(probs)
                predictions.scatter_(1, top_indices, 1.0)
                return predictions
            else:
                # Threshold
                return (probs > threshold).float()

    def get_product_probabilities(
        self,
        input_embeddings: torch.Tensor,
        input_price_features: torch.Tensor,
        input_attention_mask: torch.Tensor,
        customer_context: torch.Tensor,
        temporal_context: torch.Tensor,
        store_context: torch.Tensor,
        trip_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get probability distribution over products for RL.

        Returns:
            [B, vocab_size] - probability for each product
        """
        with torch.no_grad():
            outputs = self.forward(
                input_embeddings,
                input_price_features,
                input_attention_mask,
                customer_context,
                temporal_context,
                store_context,
                trip_context,
            )
            return torch.sigmoid(outputs['product_logits'])


def create_next_basket_model(
    vocab_size: int,
    product_dim: int = 256,
    hidden_dim: int = 512,
    encoder_layers: int = 4,
    decoder_layers: int = 2,
    dropout: float = 0.1,
) -> NextBasketWorldModel:
    """Factory function to create next-basket model."""
    config = NextBasketModelConfig(
        vocab_size=vocab_size,
        product_dim=product_dim,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        dropout=dropout,
    )
    return NextBasketWorldModel(config)


if __name__ == '__main__':
    # Quick test
    config = NextBasketModelConfig(vocab_size=5000)
    model = NextBasketWorldModel(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
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

    print(f"Product logits: {outputs['product_logits'].shape}")
    print(f"Basket size: {outputs['basket_size'].shape}")

    # Test prediction
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
    print(f"Predictions (top-10): {predictions.sum(dim=1)}")  # Should be 10 per sample
