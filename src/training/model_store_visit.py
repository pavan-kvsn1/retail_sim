"""
Store Visit Prediction Model.

Predicts which store a customer will visit next, given:
- Customer history and preferences
- Previous store visited
- Temporal context (when is next visit)

This is STAGE 1 of the two-stage world model:
1. Store Visit Prediction (this model)
2. Next Basket Prediction (conditioned on predicted store)

For RL/simulation: Predicts P(store | customer, time, history)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class StoreVisitModelConfig:
    """Configuration for StoreVisitPredictor."""
    # Input dimensions
    customer_dim: int = 192      # T1: Customer context
    temporal_dim: int = 64       # T3: Temporal context
    store_dim: int = 96          # T5: Store embedding
    basket_summary_dim: int = 64 # Summary of previous basket (optional)

    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    # Output
    num_stores: int = 800  # Number of possible stores

    # Training
    use_basket_summary: bool = True  # Include previous basket info


class BasketSummarizer(nn.Module):
    """
    Summarizes a basket into a fixed-size vector.

    Takes variable-length basket and produces a summary embedding
    that captures what was purchased (useful for store prediction).
    """

    def __init__(self, product_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.product_dim = product_dim
        self.output_dim = output_dim

        # Project and pool
        self.proj = nn.Linear(product_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        product_embeddings: torch.Tensor,  # [B, S, product_dim]
        attention_mask: torch.Tensor,       # [B, S]
    ) -> torch.Tensor:
        """
        Summarize basket into fixed vector.

        Returns:
            [B, output_dim] - basket summary
        """
        # Project products
        projected = self.proj(product_embeddings)  # [B, S, output_dim]

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
        pooled = (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.norm(pooled)


class StoreVisitPredictor(nn.Module):
    """
    Predicts which store a customer will visit next.

    Architecture:
    1. Concatenate context vectors (customer, temporal, previous store, basket summary)
    2. MLP to process combined context
    3. Output layer predicts probability over all stores

    For customers who always shop at one store, this will learn strong preferences.
    For customers who visit multiple stores, it learns patterns (e.g., weekday vs weekend stores).
    """

    def __init__(self, config: StoreVisitModelConfig):
        super().__init__()
        self.config = config

        # Calculate input dimension
        input_dim = config.customer_dim + config.temporal_dim + config.store_dim
        if config.use_basket_summary:
            input_dim += config.basket_summary_dim
            self.basket_summarizer = BasketSummarizer(
                product_dim=256,  # Standard product embedding dim
                output_dim=config.basket_summary_dim
            )
        else:
            self.basket_summarizer = None

        # MLP layers
        layers = []
        current_dim = input_dim

        for i in range(config.num_layers):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            current_dim = config.hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output head - predicts store
        self.store_head = nn.Linear(config.hidden_dim, config.num_stores)

        # Store embeddings (learnable, for encoding previous store)
        self.store_embeddings = nn.Embedding(config.num_stores + 1, config.store_dim)  # +1 for unknown

    def forward(
        self,
        customer_context: torch.Tensor,      # [B, 192]
        temporal_context: torch.Tensor,       # [B, 64]
        previous_store_idx: torch.Tensor,     # [B] - index of previous store
        previous_basket_embeddings: Optional[torch.Tensor] = None,  # [B, S, 256]
        previous_basket_mask: Optional[torch.Tensor] = None,        # [B, S]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next store visit.

        Returns:
            Dict with:
            - 'store_logits': [B, num_stores] - raw logits
            - 'store_probs': [B, num_stores] - probabilities (softmax)
        """
        B = customer_context.shape[0]

        # Get previous store embedding
        prev_store_emb = self.store_embeddings(previous_store_idx)  # [B, store_dim]

        # Build input
        inputs = [customer_context, temporal_context, prev_store_emb]

        # Add basket summary if configured
        if self.basket_summarizer is not None and previous_basket_embeddings is not None:
            basket_summary = self.basket_summarizer(
                previous_basket_embeddings,
                previous_basket_mask if previous_basket_mask is not None else torch.ones(B, previous_basket_embeddings.shape[1], device=customer_context.device)
            )
            inputs.append(basket_summary)
        elif self.basket_summarizer is not None:
            # No basket provided, use zeros
            inputs.append(torch.zeros(B, self.config.basket_summary_dim, device=customer_context.device))

        # Concatenate and process
        combined = torch.cat(inputs, dim=-1)
        hidden = self.mlp(combined)

        # Predict store
        store_logits = self.store_head(hidden)
        store_probs = F.softmax(store_logits, dim=-1)

        return {
            'store_logits': store_logits,
            'store_probs': store_probs,
        }

    def predict_store(
        self,
        customer_context: torch.Tensor,
        temporal_context: torch.Tensor,
        previous_store_idx: torch.Tensor,
        previous_basket_embeddings: Optional[torch.Tensor] = None,
        previous_basket_mask: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k most likely stores.

        Returns:
            - top_k_indices: [B, k] - store indices
            - top_k_probs: [B, k] - probabilities
        """
        with torch.no_grad():
            outputs = self.forward(
                customer_context,
                temporal_context,
                previous_store_idx,
                previous_basket_embeddings,
                previous_basket_mask,
            )

            probs = outputs['store_probs']
            top_k_probs, top_k_indices = probs.topk(top_k, dim=-1)

            return top_k_indices, top_k_probs

    def sample_store(
        self,
        customer_context: torch.Tensor,
        temporal_context: torch.Tensor,
        previous_store_idx: torch.Tensor,
        previous_basket_embeddings: Optional[torch.Tensor] = None,
        previous_basket_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample a store from the predicted distribution.

        Args:
            temperature: Higher = more random, lower = more deterministic

        Returns:
            [B] - sampled store indices
        """
        with torch.no_grad():
            outputs = self.forward(
                customer_context,
                temporal_context,
                previous_store_idx,
                previous_basket_embeddings,
                previous_basket_mask,
            )

            logits = outputs['store_logits'] / temperature
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

            return sampled


def create_store_visit_model(
    num_stores: int = 800,
    hidden_dim: int = 256,
    num_layers: int = 2,
    use_basket_summary: bool = True,
) -> StoreVisitPredictor:
    """Factory function to create store visit prediction model."""
    config = StoreVisitModelConfig(
        num_stores=num_stores,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_basket_summary=use_basket_summary,
    )
    return StoreVisitPredictor(config)


if __name__ == '__main__':
    # Quick test
    config = StoreVisitModelConfig(num_stores=761)
    model = StoreVisitPredictor(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    B = 4

    outputs = model(
        customer_context=torch.randn(B, 192),
        temporal_context=torch.randn(B, 64),
        previous_store_idx=torch.randint(0, 761, (B,)),
        previous_basket_embeddings=torch.randn(B, 10, 256),
        previous_basket_mask=torch.ones(B, 10),
    )

    print(f"Store logits: {outputs['store_logits'].shape}")
    print(f"Store probs sum: {outputs['store_probs'].sum(dim=-1)}")  # Should be ~1.0

    # Test prediction
    top_stores, top_probs = model.predict_store(
        customer_context=torch.randn(B, 192),
        temporal_context=torch.randn(B, 64),
        previous_store_idx=torch.randint(0, 761, (B,)),
        top_k=5,
    )
    print(f"Top 5 stores: {top_stores.shape}")
    print(f"Top 5 probs: {top_probs[0]}")

    # Test sampling
    sampled = model.sample_store(
        customer_context=torch.randn(B, 192),
        temporal_context=torch.randn(B, 64),
        previous_store_idx=torch.randint(0, 761, (B,)),
        temperature=1.0,
    )
    print(f"Sampled stores: {sampled}")
