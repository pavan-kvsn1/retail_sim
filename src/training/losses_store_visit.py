"""
Loss Functions for Store Visit Prediction.

Main losses:
- Cross-entropy: Standard classification loss with class weights
- Focal Loss: Down-weights easy examples, focuses on hard predictions

Both support:
- Class weights for store frequency imbalance
- Label smoothing for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for handling both class imbalance and example difficulty.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
    - p_t = probability of correct class
    - gamma = focusing parameter (higher = more focus on hard examples)
    - alpha_t = class weight for the target class

    Benefits over cross-entropy:
    - Down-weights easy/confident predictions (loyal customers)
    - Focuses learning on hard examples (store switchers)
    - Works alongside class weights for double imbalance handling

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            gamma: Focusing parameter. 0 = cross-entropy, 2 = typical focal loss.
                   Higher values put more focus on hard examples.
            alpha: Optional [num_classes] tensor of class weights.
                   Can be used alongside gamma for handling class imbalance.
            label_smoothing: Label smoothing factor (0.0 = none).
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,   # [B, C]
        targets: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or [B] losses
        """
        num_classes = logits.size(-1)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)  # [B, C]

        # Get probability of target class: p_t
        # Gather the probability at the target index
        targets_one_hot = F.one_hot(targets, num_classes).float()  # [B, C]

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
        else:
            targets_smooth = targets_one_hot

        # p_t for each sample
        p_t = (probs * targets_one_hot).sum(dim=-1)  # [B]

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma  # [B]

        # Cross-entropy component: -log(p_t)
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        ce_loss = -(targets_smooth * log_probs).sum(dim=-1)  # [B]

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # [B]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class StoreVisitLoss(nn.Module):
    """
    Cross-entropy loss for store visit prediction.

    Supports:
    - Class weights for imbalanced stores
    - Label smoothing for regularization
    """

    def __init__(
        self,
        num_stores: int,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            num_stores: Number of store classes
            class_weights: Optional [num_stores] tensor of class weights
            label_smoothing: Label smoothing factor (0.0 = none, 0.1 = typical)
        """
        super().__init__()
        self.num_stores = num_stores
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,      # [B, num_stores]
        targets: torch.Tensor,      # [B] - store indices
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Returns:
            Scalar loss value
        """
        loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        return loss


class StoreVisitMetrics:
    """Metrics for store visit prediction."""

    @staticmethod
    def compute_accuracy(
        logits: torch.Tensor,  # [B, num_stores]
        targets: torch.Tensor,  # [B]
    ) -> float:
        """Compute top-1 accuracy."""
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float().mean()
        return correct.item()

    @staticmethod
    def compute_top_k_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5,
    ) -> float:
        """Compute top-k accuracy."""
        _, top_k = logits.topk(k, dim=-1)
        correct = (top_k == targets.unsqueeze(-1)).any(dim=-1).float().mean()
        return correct.item()

    @staticmethod
    def compute_mrr(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute Mean Reciprocal Rank.

        MRR = 1/|Q| * sum(1/rank_i)
        """
        # Get ranks (1-indexed)
        sorted_indices = logits.argsort(dim=-1, descending=True)
        ranks = (sorted_indices == targets.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1
        mrr = (1.0 / ranks.float()).mean()
        return mrr.item()

    @staticmethod
    def compute_all(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            'accuracy': StoreVisitMetrics.compute_accuracy(logits, targets),
            'top_3_accuracy': StoreVisitMetrics.compute_top_k_accuracy(logits, targets, k=3),
            'top_5_accuracy': StoreVisitMetrics.compute_top_k_accuracy(logits, targets, k=5),
            'top_10_accuracy': StoreVisitMetrics.compute_top_k_accuracy(logits, targets, k=10),
            'mrr': StoreVisitMetrics.compute_mrr(logits, targets),
        }


def compute_store_class_weights(
    store_counts: Dict[int, int],
    num_stores: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute class weights for imbalanced stores.

    Uses inverse frequency with smoothing:
        weight_i = (total / count_i) ^ smoothing

    Args:
        store_counts: Dict mapping store_idx -> count
        num_stores: Total number of stores
        smoothing: Smoothing factor (0 = equal weights, 1 = full inverse freq)

    Returns:
        [num_stores] tensor of class weights
    """
    total = sum(store_counts.values())
    weights = torch.ones(num_stores)

    for store_idx, count in store_counts.items():
        if store_idx < num_stores and count > 0:
            weights[store_idx] = (total / count) ** smoothing

    # Normalize so mean weight = 1
    weights = weights / weights.mean()

    return weights


if __name__ == '__main__':
    # Quick test
    B = 32
    num_stores = 761

    logits = torch.randn(B, num_stores)
    targets = torch.randint(0, num_stores, (B,))

    # Test cross-entropy loss
    print("=== Cross-Entropy Loss ===")
    ce_loss_fn = StoreVisitLoss(num_stores, label_smoothing=0.1)
    ce_loss = ce_loss_fn(logits, targets)
    print(f"CE Loss: {ce_loss.item():.4f}")

    # Test focal loss (no class weights)
    print("\n=== Focal Loss (gamma=2.0) ===")
    focal_loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
    focal_loss = focal_loss_fn(logits, targets)
    print(f"Focal Loss: {focal_loss.item():.4f}")

    # Test focal loss with class weights
    print("\n=== Focal Loss with Class Weights ===")
    store_counts = {i: np.random.randint(100, 10000) for i in range(num_stores)}
    weights = compute_store_class_weights(store_counts, num_stores, smoothing=0.3)
    focal_weighted_fn = FocalLoss(gamma=2.0, alpha=weights, label_smoothing=0.1)
    focal_weighted_loss = focal_weighted_fn(logits, targets)
    print(f"Focal Loss (weighted): {focal_weighted_loss.item():.4f}")

    # Compare: easy vs hard examples
    print("\n=== Easy vs Hard Example Comparison ===")
    # Create "easy" predictions (high confidence on correct class)
    easy_logits = torch.zeros(4, num_stores)
    easy_targets = torch.tensor([0, 1, 2, 3])
    easy_logits[0, 0] = 10.0  # Very confident
    easy_logits[1, 1] = 10.0
    easy_logits[2, 2] = 10.0
    easy_logits[3, 3] = 10.0

    # Create "hard" predictions (low confidence)
    hard_logits = torch.randn(4, num_stores) * 0.5  # Uncertain
    hard_targets = torch.tensor([0, 1, 2, 3])

    focal_fn = FocalLoss(gamma=2.0)
    ce_fn = StoreVisitLoss(num_stores, label_smoothing=0.0)

    print(f"Easy examples - CE: {ce_fn(easy_logits, easy_targets):.4f}, Focal: {focal_fn(easy_logits, easy_targets):.4f}")
    print(f"Hard examples - CE: {ce_fn(hard_logits, hard_targets):.4f}, Focal: {focal_fn(hard_logits, hard_targets):.4f}")
    print("(Focal loss should be much lower for easy examples)")

    # Test metrics
    print("\n=== Metrics ===")
    metrics = StoreVisitMetrics.compute_all(logits, targets)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nClass weights: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
