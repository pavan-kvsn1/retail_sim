"""
World Model Architecture for RetailSim.

Implements hybrid Mamba-Transformer architecture:
- Context Fusion Layer: Combines customer, temporal, store, trip contexts
- Product Sequence Fusion: Combines product embeddings with price features
- Mamba Encoder (4 layers): Efficient long-sequence processing for customer history
- Transformer Decoder (2 layers): Basket generation with cross-attention
- Multi-task Output Heads: Masked product prediction + auxiliary tasks

Reference: RetailSim_Data_Pipeline_and_World_Model_Design.md Section 5
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """Configuration for World Model architecture."""
    # Dimensions
    d_model: int = 512
    d_context: int = 400  # T1(192) + T3(64) + T5(96) + T6(48)
    d_product: int = 256
    d_price: int = 64
    d_sequence: int = 320  # product(256) + price(64)

    # Mamba encoder
    mamba_state_size: int = 64
    mamba_conv_kernel: int = 4
    mamba_num_layers: int = 4
    mamba_expand: int = 2

    # Transformer decoder
    decoder_num_layers: int = 2
    decoder_num_heads: int = 8
    decoder_d_ff: int = 2048
    decoder_dropout: float = 0.1

    # Vocabulary
    n_products: int = 5003  # Products + PAD + EOS + MASK
    pad_token_id: int = 0
    eos_token_id: int = 5001
    mask_token_id: int = 5002

    # Auxiliary tasks
    n_basket_sizes: int = 4
    n_price_sens: int = 4
    n_mission_types: int = 5
    n_mission_focus: int = 6

    # Sequence limits
    max_history_len: int = 117  # Max weeks of history
    max_basket_len: int = 50   # Max products per basket


class ContextFusion(nn.Module):
    """
    Fuses dense context tensors into single context vector.

    Input: T1[192] + T3[64] + T5[96] + T6[48] = [400d]
    Output: context_vector [512d]
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.projection = nn.Linear(config.d_context, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.activation = nn.GELU()

    def forward(self, dense_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense_context: [B, 400] concatenated context

        Returns:
            context_vector: [B, 512]
        """
        x = self.projection(dense_context)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x


class ProductSequenceFusion(nn.Module):
    """
    Fuses product embeddings and price features into unified sequence.

    Input: T2[256] + T4[64] = [320d] per position
    Output: product_sequence [B, S, 512]
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.projection = nn.Linear(config.d_sequence, config.d_model)
        self.positional_encoding = self._init_positional_encoding(
            config.max_basket_len, config.d_model
        )
        self.dropout = nn.Dropout(config.decoder_dropout)

    def _init_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def forward(
        self,
        product_embeddings: torch.Tensor,
        price_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            product_embeddings: [B, S, 256]
            price_features: [B, S, 64]

        Returns:
            product_sequence: [B, S, 512]
        """
        B, S, _ = product_embeddings.shape

        # Concatenate and project
        combined = torch.cat([product_embeddings, price_features], dim=-1)  # [B, S, 320]
        projected = self.projection(combined)  # [B, S, 512]

        # Add positional encoding
        projected = projected + self.positional_encoding[:S].unsqueeze(0)

        return self.dropout(projected)


class MambaBlock(nn.Module):
    """
    Single Mamba block for efficient sequence processing.

    Implements selective state-space model with:
    - Input projection and gating
    - Depthwise convolution for local context
    - Selective SSM for long-range dependencies
    - Residual connection

    Note: This is a simplified implementation. For production,
    use the official mamba-ssm library for CUDA-optimized kernels.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.mamba_state_size
        self.d_conv = config.mamba_conv_kernel
        self.expand = config.mamba_expand

        self.d_inner = self.d_model * self.expand

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Initialize A, D (log space for A)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] input sequence

        Returns:
            output: [B, S, D] processed sequence
        """
        B, S, _ = x.shape
        residual = x

        # Input projection: split into main path and gate
        xz = self.in_proj(x)  # [B, S, 2*d_inner]
        x_main, z = xz.chunk(2, dim=-1)  # Each [B, S, d_inner]

        # Depthwise convolution (transpose for conv1d: B,D,S)
        x_conv = x_main.transpose(1, 2)  # [B, d_inner, S]
        x_conv = self.conv1d(x_conv)[:, :, :S]  # Causal: trim extra
        x_conv = x_conv.transpose(1, 2)  # [B, S, d_inner]
        x_conv = F.silu(x_conv)

        # Selective SSM
        # Compute delta (time step), B, C
        x_dbl = self.x_proj(x_conv)  # [B, S, d_state*2]
        B_proj, C_proj = x_dbl.chunk(2, dim=-1)  # [B, S, d_state] each

        dt = F.softplus(self.dt_proj(x_conv))  # [B, S, d_inner]

        # Discretize A
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Simplified selective scan (sequential for clarity)
        # For efficiency, use parallel scan or mamba-ssm CUDA kernel
        y = self._selective_scan(x_conv, dt, A, B_proj, C_proj)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        # Residual + LayerNorm
        return self.layer_norm(output + residual)

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified selective scan implementation.

        For production, replace with mamba-ssm's CUDA kernel.
        """
        B_size, S, D = x.shape
        device = x.device

        # Initialize hidden state
        h = torch.zeros(B_size, D, self.d_state, device=device)

        outputs = []
        for t in range(S):
            # Discretized A, B
            dt_t = dt[:, t, :].unsqueeze(-1)  # [B, D, 1]
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # [B, D, d_state]
            B_bar = dt_t * B[:, t, :].unsqueeze(1)  # [B, D, d_state]

            # State update
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)

            # Output
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # [B, D]
            y_t = y_t + self.D * x[:, t, :]  # Skip connection

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [B, S, D]


class MambaEncoder(nn.Module):
    """
    Mamba encoder for customer history processing.

    4 layers of Mamba blocks for O(n) sequence processing.
    Efficiently handles long customer histories (up to 117 weeks).
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.mamba_num_layers)
        ])
        self._gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] customer history sequence

        Returns:
            encoder_output: [B, S, D] encoded sequence
        """
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer with cross-attention.

    - Masked self-attention (causal)
    - Cross-attention to customer state
    - Position-wise feedforward
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.decoder_num_heads
        self.d_ff = config.decoder_d_ff
        self.dropout = config.decoder_dropout

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(self.d_model)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(self.d_model)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(self.dropout)
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] basket sequence
            encoder_output: [B, T, D] encoder output (customer history)
            self_attn_mask: [S, S] causal mask for self-attention
            cross_attn_mask: [B, T] encoder padding mask

        Returns:
            output: [B, S, D] decoded sequence
        """
        # Self-attention (causal)
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = residual + x

        # Cross-attention to encoder output
        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(
            x, encoder_output, encoder_output,
            key_padding_mask=cross_attn_mask
        )
        x = residual + x

        # Feedforward
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for basket generation.

    2 layers with cross-attention to customer history.
    Generates basket sequence conditioned on customer state.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config.decoder_num_layers)
        ])
        self.max_len = config.max_basket_len
        self._gradient_checkpointing = False

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] basket sequence
            encoder_output: [B, T, D] customer history encoding
            encoder_mask: [B, T] mask for encoder output (0 = valid, 1 = padding)

        Returns:
            decoder_output: [B, S, D]
        """
        B, S, _ = x.shape

        # Generate causal mask
        causal_mask = self._generate_causal_mask(S, x.device)

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, encoder_output, causal_mask, encoder_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, encoder_output, causal_mask, encoder_mask)

        return x

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False


class OutputHeads(nn.Module):
    """
    Multi-task output heads for World Model.

    - Masked product prediction (primary task)
    - Basket size prediction
    - Price sensitivity prediction
    - Mission type prediction
    - Mission focus prediction
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_products = config.n_products

        # Primary: Masked product prediction
        self.product_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.n_products)
        )

        # Auxiliary heads (from CLS token)
        self.basket_size_head = nn.Linear(config.d_model, config.n_basket_sizes)
        self.price_sens_head = nn.Linear(config.d_model, config.n_price_sens)
        self.mission_type_head = nn.Linear(config.d_model, config.n_mission_types)
        self.mission_focus_head = nn.Linear(config.d_model, config.n_mission_focus)

    def forward(
        self,
        decoder_output: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            decoder_output: [B, S, D] decoder output
            masked_positions: [B, M] indices of masked positions

        Returns:
            masked_logits: [B, M, V] or [B, S, V] if no positions specified
            auxiliary_logits: Dict of auxiliary task logits
        """
        B, S, D = decoder_output.shape

        # Masked product logits
        if masked_positions is not None:
            # Gather representations at masked positions
            # masked_positions: [B, M]
            M = masked_positions.shape[1]
            # Expand for gather: [B, M, D]
            idx = masked_positions.unsqueeze(-1).expand(-1, -1, D)
            masked_repr = torch.gather(decoder_output, 1, idx)  # [B, M, D]
            masked_logits = self.product_head(masked_repr)  # [B, M, V]
        else:
            # Full sequence logits
            masked_logits = self.product_head(decoder_output)  # [B, S, V]

        # CLS token (position 0) for auxiliary tasks
        cls_repr = decoder_output[:, 0, :]  # [B, D]

        auxiliary_logits = {
            'basket_size': self.basket_size_head(cls_repr),
            'price_sensitivity': self.price_sens_head(cls_repr),
            'mission_type': self.mission_type_head(cls_repr),
            'mission_focus': self.mission_focus_head(cls_repr),
        }

        return masked_logits, auxiliary_logits


class WorldModel(nn.Module):
    """
    World Model for retail basket prediction.

    Architecture:
    1. Context Fusion: Dense contexts -> [B, 512]
    2. Product Fusion: Product + price -> [B, S, 512]
    3. Mamba Encoder: Customer history encoding
    4. Transformer Decoder: Basket generation with cross-attention
    5. Output Heads: Multi-task prediction

    Total parameters: ~14.7M
    """

    def __init__(self, config: Optional[WorldModelConfig] = None):
        super().__init__()
        self.config = config or WorldModelConfig()

        # Context fusion
        self.context_fusion = ContextFusion(self.config)

        # Product sequence fusion
        self.product_fusion = ProductSequenceFusion(self.config)

        # Encoder: Mamba for customer history
        self.encoder = MambaEncoder(self.config)

        # Decoder: Transformer for basket generation
        self.decoder = TransformerDecoder(self.config)

        # Output heads
        self.output_heads = OutputHeads(self.config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        dense_context: torch.Tensor,
        product_embeddings: torch.Tensor,
        price_features: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None,
        history_context: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through World Model.

        Args:
            dense_context: [B, 400] concatenated context (T1+T3+T5+T6)
            product_embeddings: [B, S, 256] product embeddings (T2)
            price_features: [B, S, 64] price features (T4)
            attention_mask: [B, S] valid positions (1) vs padding (0)
            masked_positions: [B, M] indices of masked positions for MLM
            history_context: [B, H, 512] optional pre-computed history encoding
            history_mask: [B, H] mask for history (0=valid, 1=padding)

        Returns:
            masked_logits: [B, M, V] logits for masked positions
            auxiliary_logits: Dict of auxiliary task logits
            encoder_output: [B, S+1, D] encoder output for contrastive loss
        """
        B = dense_context.shape[0]

        # 1. Context fusion
        context_vector = self.context_fusion(dense_context)  # [B, 512]

        # 2. Product sequence fusion
        product_sequence = self.product_fusion(product_embeddings, price_features)  # [B, S, 512]

        # 3. Prepend context as "CLS" token
        # Treat context as first token for encoder input
        context_token = context_vector.unsqueeze(1)  # [B, 1, 512]
        encoder_input = torch.cat([context_token, product_sequence], dim=1)  # [B, S+1, 512]

        # Update attention mask
        ones = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        encoder_mask = torch.cat([ones, attention_mask], dim=1)  # [B, S+1]

        # 4. Mamba encoder
        if history_context is not None:
            # Use pre-computed history + current context
            encoder_output = torch.cat([history_context, encoder_input], dim=1)
            if history_mask is not None:
                encoder_mask = torch.cat([history_mask, encoder_mask], dim=1)
        else:
            encoder_output = self.encoder(encoder_input)  # [B, S+1, 512]

        # 5. Transformer decoder (decode basket from encoded state)
        # For training, decoder input = encoder output (self-decoding)
        decoder_output = self.decoder(
            encoder_input,
            encoder_output,
            encoder_mask=(1 - encoder_mask).bool()  # Convert to padding mask format
        )

        # 6. Output heads
        # Adjust masked positions for CLS token offset
        if masked_positions is not None:
            adjusted_positions = masked_positions + 1  # Offset for CLS
        else:
            adjusted_positions = None

        masked_logits, auxiliary_logits = self.output_heads(
            decoder_output, adjusted_positions
        )

        return masked_logits, auxiliary_logits, encoder_output

    def get_encoder_output(
        self,
        dense_context: torch.Tensor,
        product_embeddings: torch.Tensor,
        price_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get encoder output for downstream tasks (e.g., RL)."""
        B = dense_context.shape[0]

        context_vector = self.context_fusion(dense_context)
        product_sequence = self.product_fusion(product_embeddings, price_features)

        context_token = context_vector.unsqueeze(1)
        encoder_input = torch.cat([context_token, product_sequence], dim=1)

        return self.encoder(encoder_input)

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce memory usage at the cost of speed."""
        self._gradient_checkpointing = True
        # Enable for encoder (Mamba layers)
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        # Enable for decoder (Transformer layers)
        if hasattr(self.decoder, 'gradient_checkpointing_enable'):
            self.decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        if hasattr(self.encoder, 'gradient_checkpointing_disable'):
            self.encoder.gradient_checkpointing_disable()
        if hasattr(self.decoder, 'gradient_checkpointing_disable'):
            self.decoder.gradient_checkpointing_disable()


def create_world_model(
    n_products: int = 5003,
    d_model: int = 512,
    mamba_layers: int = 4,
    decoder_layers: int = 2,
    **kwargs
) -> WorldModel:
    """Factory function to create World Model with custom config."""
    config = WorldModelConfig(
        n_products=n_products,
        d_model=d_model,
        mamba_num_layers=mamba_layers,
        decoder_num_layers=decoder_layers,
        **kwargs
    )
    return WorldModel(config)


if __name__ == '__main__':
    # Test model
    print("Testing WorldModel...")

    config = WorldModelConfig()
    model = WorldModel(config)

    print(f"Model parameters: {model.num_parameters:,}")

    # Create dummy inputs
    B, S = 4, 20
    dense_context = torch.randn(B, 400)
    product_embeddings = torch.randn(B, S, 256)
    price_features = torch.randn(B, S, 64)
    attention_mask = torch.ones(B, S)
    masked_positions = torch.randint(0, S, (B, 3))

    # Forward pass
    with torch.no_grad():
        masked_logits, aux_logits, encoder_out = model(
            dense_context,
            product_embeddings,
            price_features,
            attention_mask,
            masked_positions
        )

    print(f"\nOutput shapes:")
    print(f"  Masked logits: {masked_logits.shape}")
    print(f"  Encoder output: {encoder_out.shape}")
    print(f"  Auxiliary logits:")
    for name, logits in aux_logits.items():
        print(f"    {name}: {logits.shape}")
