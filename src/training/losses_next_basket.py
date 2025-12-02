"""
Loss Functions for Next-Basket Prediction.

Key differences from masked prediction:
1. Multi-label loss (BCE) instead of single-label (CE)
2. Class imbalance handling (most products NOT purchased)
3. F1-based metrics more relevant than accuracy

Loss Components:
1. Product Prediction Loss (main): BCE with positive weighting
2. Basket Size Loss: Help model calibrate number of items
3. Auxiliary Losses: Mission type, focus, price sensitivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NextBasketLossConfig:
    """Configuration for next-basket loss computation."""
    # Main loss weighting
    product_weight: float = 0.7
    auxiliary_weight: float = 0.3

    # Positive class weight for BCE (products are sparse)
    # If avg basket has 10 items out of 5000 products:
    # positive_weight = (5000 - 10) / 10 = ~500
    # But we cap it for stability
    pos_weight: float = 50.0

    # Focal loss parameters (optional)
    use_focal: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # Label smoothing
    label_smoothing: float = 0.0

    # Auxiliary task weights
    basket_size_weight: float = 1.0
    mission_type_weight: float = 0.5
    mission_focus_weight: float = 0.5
    price_sens_weight: float = 0.5


class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross-Entropy Loss for multi-label classification.

    Addresses class imbalance by down-weighting easy negatives.
    Essential for basket prediction where most products are NOT bought.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[float] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,      # [B, V]
        targets: torch.Tensor,     # [B, V] multi-hot
        mask: Optional[torch.Tensor] = None,  # [B, V] valid positions
    ) -> torch.Tensor:
        """
        Compute focal BCE loss.

        Args:
            logits: Raw predictions before sigmoid
            targets: Multi-hot target vectors
            mask: Optional mask for valid products
        """
        probs = torch.sigmoid(logits)

        # BCE component
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Focal weighting
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting (balance pos/neg)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Additional positive class weight
        if self.pos_weight is not None:
            pos_weight = torch.where(
                targets == 1,
                torch.tensor(self.pos_weight, device=logits.device),
                torch.tensor(1.0, device=logits.device),
            )
            focal_weight = focal_weight * pos_weight

        loss = alpha_weight * focal_weight * bce

        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum().clamp(min=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class NextBasketLoss(nn.Module):
    """
    Combined loss for next-basket prediction.

    Components:
    1. Product prediction: Focal BCE (handles class imbalance)
    2. Basket size: Cross-entropy (calibration)
    3. Auxiliary: Mission type, focus, price sensitivity
    """

    def __init__(self, config: Optional[NextBasketLossConfig] = None):
        super().__init__()
        self.config = config or NextBasketLossConfig()

        # Main product loss
        if self.config.use_focal:
            self.product_loss = FocalBCELoss(
                gamma=self.config.focal_gamma,
                alpha=self.config.focal_alpha,
                pos_weight=self.config.pos_weight,
            )
        else:
            self.product_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.config.pos_weight)
            )

        # Auxiliary losses
        self.basket_size_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.mission_type_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.mission_focus_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.price_sens_loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,              # [B, V] multi-hot
        auxiliary_labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.

        Args:
            outputs: Model outputs with 'product_logits' and auxiliary predictions
            targets: Multi-hot target vectors [B, vocab_size]
            auxiliary_labels: Dict with 'basket_size', 'mission_type', etc.

        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}

        # 1. Product prediction loss (main)
        product_logits = outputs['product_logits']
        product_loss = self.product_loss(product_logits, targets)
        loss_dict['product'] = product_loss

        # 2. Auxiliary losses
        aux_total = 0.0

        if 'basket_size' in outputs and 'basket_size' in auxiliary_labels:
            bs_loss = self.basket_size_loss(
                outputs['basket_size'],
                auxiliary_labels['basket_size']
            )
            loss_dict['basket_size'] = bs_loss
            aux_total += self.config.basket_size_weight * bs_loss

        if 'mission_type' in outputs and 'mission_type' in auxiliary_labels:
            mt_loss = self.mission_type_loss(
                outputs['mission_type'],
                auxiliary_labels['mission_type']
            )
            loss_dict['mission_type'] = mt_loss
            aux_total += self.config.mission_type_weight * mt_loss

        if 'mission_focus' in outputs and 'mission_focus' in auxiliary_labels:
            mf_loss = self.mission_focus_loss(
                outputs['mission_focus'],
                auxiliary_labels['mission_focus']
            )
            loss_dict['mission_focus'] = mf_loss
            aux_total += self.config.mission_focus_weight * mf_loss

        if 'price_sensitivity' in outputs and 'price_sensitivity' in auxiliary_labels:
            ps_loss = self.price_sens_loss(
                outputs['price_sensitivity'],
                auxiliary_labels['price_sensitivity']
            )
            loss_dict['price_sensitivity'] = ps_loss
            aux_total += self.config.price_sens_weight * ps_loss

        loss_dict['auxiliary'] = aux_total

        # Total loss
        total = (
            self.config.product_weight * product_loss +
            self.config.auxiliary_weight * aux_total
        )
        loss_dict['total'] = total

        return total, loss_dict


class NextBasketMetrics:
    """
    Metrics for next-basket prediction evaluation.

    Key metrics:
    - Precision@k: Of predicted items, how many were actually bought
    - Recall@k: Of bought items, how many were predicted
    - F1@k: Harmonic mean
    - NDCG: Ranking quality
    - Hit Rate: Did we predict at least one correct item
    """

    @staticmethod
    def precision_at_k(
        predictions: torch.Tensor,  # [B, V] probabilities
        targets: torch.Tensor,      # [B, V] multi-hot
        k: int = 10,
    ) -> torch.Tensor:
        """Precision@k - fraction of top-k predictions that are correct."""
        _, top_k_indices = predictions.topk(k, dim=-1)

        # Gather target values at predicted positions
        hits = targets.gather(1, top_k_indices)

        return hits.sum(dim=1).float() / k

    @staticmethod
    def recall_at_k(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 10,
    ) -> torch.Tensor:
        """Recall@k - fraction of actual items that appear in top-k."""
        _, top_k_indices = predictions.topk(k, dim=-1)

        hits = targets.gather(1, top_k_indices)
        actual_counts = targets.sum(dim=1).clamp(min=1)

        return hits.sum(dim=1).float() / actual_counts

    @staticmethod
    def f1_at_k(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 10,
    ) -> torch.Tensor:
        """F1@k - harmonic mean of precision and recall."""
        precision = NextBasketMetrics.precision_at_k(predictions, targets, k)
        recall = NextBasketMetrics.recall_at_k(predictions, targets, k)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

    @staticmethod
    def hit_rate_at_k(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 10,
    ) -> torch.Tensor:
        """Hit rate@k - fraction of samples with at least one hit in top-k."""
        _, top_k_indices = predictions.topk(k, dim=-1)
        hits = targets.gather(1, top_k_indices)

        return (hits.sum(dim=1) > 0).float()

    @staticmethod
    def ndcg_at_k(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 10,
    ) -> torch.Tensor:
        """NDCG@k - normalized discounted cumulative gain."""
        B = predictions.shape[0]

        _, top_k_indices = predictions.topk(k, dim=-1)
        hits = targets.gather(1, top_k_indices).float()

        # DCG
        positions = torch.arange(1, k + 1, device=predictions.device).float()
        discounts = torch.log2(positions + 1)
        dcg = (hits / discounts).sum(dim=1)

        # Ideal DCG (all relevant items at top)
        num_relevant = targets.sum(dim=1).clamp(max=k)
        ideal_hits = torch.zeros(B, k, device=predictions.device)
        for i in range(B):
            n_rel = int(num_relevant[i].item())
            ideal_hits[i, :n_rel] = 1.0
        idcg = (ideal_hits / discounts).sum(dim=1).clamp(min=1e-8)

        return dcg / idcg

    @staticmethod
    def compute_all(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k_values: list = [5, 10, 20],
    ) -> Dict[str, float]:
        """Compute all metrics at various k values."""
        metrics = {}

        for k in k_values:
            metrics[f'precision@{k}'] = NextBasketMetrics.precision_at_k(predictions, targets, k).mean().item()
            metrics[f'recall@{k}'] = NextBasketMetrics.recall_at_k(predictions, targets, k).mean().item()
            metrics[f'f1@{k}'] = NextBasketMetrics.f1_at_k(predictions, targets, k).mean().item()
            metrics[f'hit_rate@{k}'] = NextBasketMetrics.hit_rate_at_k(predictions, targets, k).mean().item()
            metrics[f'ndcg@{k}'] = NextBasketMetrics.ndcg_at_k(predictions, targets, k).mean().item()

        return metrics


if __name__ == '__main__':
    # Test losses
    B, V = 4, 5000

    logits = torch.randn(B, V)
    # Sparse targets (only ~10 products per basket)
    targets = torch.zeros(B, V)
    for i in range(B):
        pos_indices = torch.randint(0, V, (10,))
        targets[i, pos_indices] = 1.0

    print("Testing NextBasketLoss...")
    loss_fn = NextBasketLoss()

    outputs = {
        'product_logits': logits,
        'basket_size': torch.randn(B, 4),
        'mission_type': torch.randn(B, 5),
        'mission_focus': torch.randn(B, 6),
        'price_sensitivity': torch.randn(B, 4),
    }

    aux_labels = {
        'basket_size': torch.randint(1, 4, (B,)),
        'mission_type': torch.randint(1, 5, (B,)),
        'mission_focus': torch.randint(1, 6, (B,)),
        'price_sensitivity': torch.randint(1, 4, (B,)),
    }

    total_loss, loss_dict = loss_fn(outputs, targets, aux_labels)
    print(f"Total loss: {total_loss.item():.4f}")
    for name, val in loss_dict.items():
        print(f"  {name}: {val.item() if hasattr(val, 'item') else val:.4f}")

    # Test metrics
    print("\nTesting metrics...")
    probs = torch.sigmoid(logits)
    metrics = NextBasketMetrics.compute_all(probs, targets)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")
