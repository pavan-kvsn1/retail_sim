"""
Loss Functions for World Model Training.

Implements:
- Focal Loss: Handles class imbalance for masked product prediction
- Contrastive Loss: Learns product co-occurrence relationships
- Auxiliary Cross-Entropy: For basket size, price sensitivity, mission type

Reference: RetailSim_Data_Pipeline_and_World_Model_Design.md Section 5.5.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling sparse multi-class classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Down-weights easy examples, focuses on hard examples.
    Critical for product vocabulary of 5000+ items.

    Args:
        gamma: Focusing parameter (default 2.0)
        alpha: Class weight balancing (optional)
        reduction: 'none', 'mean', or 'sum'
        ignore_index: Index to ignore in loss computation (default -100)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: [B, C] or [B, S, C] class logits
            targets: [B] or [B, S] class indices
            mask: Optional [B] or [B, S] mask (1 = compute loss, 0 = ignore)

        Returns:
            Loss value (scalar or per-element depending on reduction)
        """
        # Flatten if sequence input
        if logits.dim() == 3:
            B, S, C = logits.shape
            logits = logits.view(B * S, C)
            targets = targets.view(B * S)
            if mask is not None:
                mask = mask.view(B * S)

        # Get valid mask (not ignore_index)
        valid_mask = (targets != self.ignore_index)
        if mask is not None:
            valid_mask = valid_mask & (mask == 1)

        # Compute CE loss (no reduction)
        ce_loss = F.cross_entropy(
            logits,
            targets.clamp(min=0),  # Clamp to avoid index errors
            reduction='none'
        )

        # Get probabilities for true class
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets.clamp(min=0))
            focal_weight = focal_weight * alpha_t

        # Focal loss
        focal_loss = focal_weight * ce_loss

        # Apply valid mask
        focal_loss = focal_loss * valid_mask.float()

        # Reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            if valid_mask.sum() > 0:
                return focal_loss.sum() / valid_mask.sum()
            return torch.tensor(0.0, device=logits.device)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class ContrastiveLoss(nn.Module):
    """
    InfoNCE Contrastive Loss for product relationships.

    Learns that co-purchased products should have similar embeddings.

    L = -log(exp(sim(anchor, positive) / tau) / sum(exp(sim(anchor, neg_i) / tau)))

    Args:
        temperature: Temperature for softmax (default 0.07)
        n_negatives: Number of negative samples per anchor
    """

    def __init__(
        self,
        temperature: float = 0.07,
        n_negatives: int = 512
    ):
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives

    def forward(
        self,
        product_embeddings: torch.Tensor,
        product_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for co-purchased products.

        Args:
            product_embeddings: [B, S, D] encoded product representations
            product_ids: [B, S] product token IDs
            attention_mask: [B, S] valid positions (1) vs padding (0)

        Returns:
            Contrastive loss (scalar)
        """
        B, S, D = product_embeddings.shape
        device = product_embeddings.device

        # Normalize embeddings
        embeddings = F.normalize(product_embeddings, dim=-1)

        # Sample anchors and positives from valid positions
        total_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for b in range(B):
            # Get valid positions (not padding, not special tokens)
            valid = (attention_mask[b] == 1) & (product_ids[b] > 0)
            valid_positions = torch.where(valid)[0]

            if len(valid_positions) < 2:
                continue

            # Sample anchor and positive from same basket
            n_anchors = min(len(valid_positions) // 2, 8)
            if n_anchors == 0:
                continue

            anchor_indices = valid_positions[:n_anchors]
            positive_indices = valid_positions[n_anchors:2*n_anchors]

            if len(positive_indices) < n_anchors:
                positive_indices = valid_positions[-n_anchors:]

            anchor_embeds = embeddings[b, anchor_indices]  # [n_anchors, D]
            positive_embeds = embeddings[b, positive_indices]  # [n_anchors, D]

            # Negatives: products from other baskets
            other_batches = [i for i in range(B) if i != b]
            if len(other_batches) == 0:
                continue

            neg_embeddings = []
            for other_b in other_batches[:4]:  # Sample from up to 4 other baskets
                other_valid = (attention_mask[other_b] == 1) & (product_ids[other_b] > 0)
                other_positions = torch.where(other_valid)[0]
                if len(other_positions) > 0:
                    neg_embeddings.append(embeddings[other_b, other_positions])

            if len(neg_embeddings) == 0:
                continue

            negatives = torch.cat(neg_embeddings, dim=0)  # [N_neg, D]

            # Compute similarities
            # Anchor-Positive: [n_anchors]
            pos_sim = (anchor_embeds * positive_embeds).sum(dim=-1) / self.temperature

            # Anchor-Negative: [n_anchors, N_neg]
            neg_sim = torch.mm(anchor_embeds, negatives.t()) / self.temperature

            # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # = -pos + log(exp(pos) + sum(exp(neg)))
            # = -pos + logsumexp([pos, neg_1, neg_2, ...])

            all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [n_anchors, 1+N_neg]
            logsumexp = torch.logsumexp(all_sim, dim=-1)  # [n_anchors]

            loss = -pos_sim + logsumexp
            total_loss = total_loss + loss.sum()
            n_pairs += n_anchors

        if n_pairs > 0:
            return total_loss / n_pairs
        return torch.tensor(0.0, device=device)


class AuxiliaryLoss(nn.Module):
    """
    Combined auxiliary task losses.

    Predicts:
    - Basket size (S/M/L)
    - Price sensitivity (LA/MM/UM)
    - Mission type (Top-up/Full Shop/etc.)
    - Mission focus (Fresh/Grocery/Mixed/etc.)
    """

    def __init__(
        self,
        n_basket_sizes: int = 4,
        n_price_sens: int = 4,
        n_mission_types: int = 5,
        n_mission_focus: int = 6
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary task losses.

        Args:
            logits_dict: Dict of task -> logits [B, C]
            labels_dict: Dict of task -> labels [B]

        Returns:
            Dict of task -> loss
        """
        losses = {}

        for task in ['basket_size', 'price_sensitivity', 'mission_type', 'mission_focus']:
            if task in logits_dict and task in labels_dict:
                logits = logits_dict[task]
                labels = labels_dict[task]
                losses[task] = self.ce_loss(logits, labels)
            else:
                losses[task] = torch.tensor(0.0)

        return losses


class WorldModelLoss(nn.Module):
    """
    Combined multi-task loss for World Model training.

    L_total = w1 * L_focal + w2 * L_contrastive +
              w3 * L_basket_size + w4 * L_price_sens + w5 * L_mission

    Default weights from design doc:
        w1 = 0.60 (focal - primary task)
        w2 = 0.20 (contrastive - representation learning)
        w3 = 0.08 (basket size)
        w4 = 0.08 (price sensitivity)
        w5 = 0.04 (mission type/focus combined)
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        contrastive_temperature: float = 0.07,
        w_focal: float = 0.60,
        w_contrastive: float = 0.20,
        w_basket_size: float = 0.08,
        w_price_sens: float = 0.08,
        w_mission: float = 0.04,
        n_products: int = 5003
    ):
        super().__init__()

        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
        self.auxiliary_loss = AuxiliaryLoss()

        self.w_focal = w_focal
        self.w_contrastive = w_contrastive
        self.w_basket_size = w_basket_size
        self.w_price_sens = w_price_sens
        self.w_mission = w_mission

        self.n_products = n_products

    def forward(
        self,
        masked_logits: torch.Tensor,
        masked_targets: torch.Tensor,
        masked_mask: Optional[torch.Tensor],
        product_embeddings: torch.Tensor,
        product_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        auxiliary_logits: Optional[Dict[str, torch.Tensor]] = None,
        auxiliary_labels: Optional[Dict[str, torch.Tensor]] = None,
        phase: str = 'main'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss with component breakdown.

        Args:
            masked_logits: [B, M, V] logits for masked positions
            masked_targets: [B, M] true product IDs for masked positions
            masked_mask: [B, M] which positions are actually masked (1) vs padding (0)
            product_embeddings: [B, S, D] encoded product representations
            product_ids: [B, S] all product token IDs
            attention_mask: [B, S] valid positions mask
            auxiliary_logits: Dict of auxiliary task logits
            auxiliary_labels: Dict of auxiliary task labels
            phase: 'warmup', 'main', or 'finetune' (affects loss weights)

        Returns:
            Tuple of (total_loss, loss_dict with components)
        """
        loss_dict = {}

        # Focal loss for masked product prediction
        if masked_logits is not None and masked_targets is not None:
            focal = self.focal_loss(masked_logits, masked_targets, masked_mask)
            loss_dict['focal'] = focal
        else:
            loss_dict['focal'] = torch.tensor(0.0, device=product_embeddings.device)

        # Contrastive loss (disabled in warmup)
        if phase != 'warmup' and product_embeddings is not None:
            contrastive = self.contrastive_loss(
                product_embeddings, product_ids, attention_mask
            )
            loss_dict['contrastive'] = contrastive
        else:
            loss_dict['contrastive'] = torch.tensor(0.0, device=product_embeddings.device)

        # Auxiliary losses (disabled in warmup)
        if phase != 'warmup' and auxiliary_logits is not None and auxiliary_labels is not None:
            aux_losses = self.auxiliary_loss(auxiliary_logits, auxiliary_labels)
            loss_dict.update(aux_losses)
        else:
            loss_dict['basket_size'] = torch.tensor(0.0, device=product_embeddings.device)
            loss_dict['price_sensitivity'] = torch.tensor(0.0, device=product_embeddings.device)
            loss_dict['mission_type'] = torch.tensor(0.0, device=product_embeddings.device)
            loss_dict['mission_focus'] = torch.tensor(0.0, device=product_embeddings.device)

        # Compute weighted total
        if phase == 'warmup':
            # Warmup: only focal loss
            total = loss_dict['focal']
        else:
            total = (
                self.w_focal * loss_dict['focal'] +
                self.w_contrastive * loss_dict['contrastive'] +
                self.w_basket_size * loss_dict.get('basket_size', 0.0) +
                self.w_price_sens * loss_dict.get('price_sensitivity', 0.0) +
                self.w_mission * (loss_dict.get('mission_type', 0.0) +
                                  loss_dict.get('mission_focus', 0.0)) / 2
            )

        loss_dict['total'] = total
        return total, loss_dict

    def set_phase(self, phase: str):
        """Update loss weights for training phase."""
        if phase == 'warmup':
            # Only focal loss
            self.w_focal = 1.0
            self.w_contrastive = 0.0
            self.w_basket_size = 0.0
            self.w_price_sens = 0.0
            self.w_mission = 0.0
        elif phase == 'main':
            # Full multi-task
            self.w_focal = 0.60
            self.w_contrastive = 0.20
            self.w_basket_size = 0.08
            self.w_price_sens = 0.08
            self.w_mission = 0.04
        elif phase == 'finetune':
            # Same as main but lower LR elsewhere
            self.w_focal = 0.60
            self.w_contrastive = 0.20
            self.w_basket_size = 0.08
            self.w_price_sens = 0.08
            self.w_mission = 0.04


if __name__ == '__main__':
    # Quick test
    print("Testing FocalLoss...")
    focal = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 100)  # B=4, C=100
    targets = torch.randint(0, 100, (4,))
    loss = focal(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")

    print("\nTesting ContrastiveLoss...")
    contrastive = ContrastiveLoss()
    embeddings = torch.randn(4, 10, 256)  # B=4, S=10, D=256
    ids = torch.randint(1, 100, (4, 10))
    mask = torch.ones(4, 10)
    loss = contrastive(embeddings, ids, mask)
    print(f"Contrastive loss: {loss.item():.4f}")

    print("\nTesting WorldModelLoss...")
    wm_loss = WorldModelLoss()
    masked_logits = torch.randn(4, 3, 5003)  # B=4, M=3, V=5003
    masked_targets = torch.randint(1, 5000, (4, 3))
    masked_mask = torch.ones(4, 3)

    total, loss_dict = wm_loss(
        masked_logits, masked_targets, masked_mask,
        embeddings, ids, mask,
        phase='main'
    )
    print(f"Total loss: {total.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
