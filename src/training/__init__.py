"""
Training module for RetailSim World Model.

This module provides TWO training paradigms:

1. MASKED PREDICTION (original):
   - BERT-style masking within current basket
   - Use: WorldModelDataset, WorldModel, WorldModelLoss
   - Good for learning product co-occurrence

2. NEXT-BASKET PREDICTION (recommended for RL/simulation):
   - Predict entire next basket given current basket
   - Use: NextBasketDataset, NextBasketWorldModel, NextBasketLoss
   - Required for proper world model / RL environment

Architecture follows RetailSim_Data_Pipeline_and_World_Model_Design.md v7.6
"""

from .prepare_samples import enhance_temporal_metadata
from .prepare_tensor_cache import prepare_tensor_cache

# Original masked prediction
from .dataset import WorldModelDataset, WorldModelDataLoader, EvaluationDataLoader
from .model import WorldModel, WorldModelConfig, create_world_model
from .losses import FocalLoss, ContrastiveLoss, WorldModelLoss
from .train import TrainingConfig, Trainer
from .evaluate import Evaluator, EvaluationMetrics, run_evaluation

# Next-basket prediction (for RL/simulation)
from .dataset_next_basket import NextBasketDataset, NextBasketDataLoader
from .model_next_basket import NextBasketWorldModel, NextBasketModelConfig
from .losses_next_basket import NextBasketLoss, NextBasketMetrics

__all__ = [
    # Data preparation
    'enhance_temporal_metadata',
    'prepare_tensor_cache',
    # Original masked prediction
    'WorldModelDataset',
    'WorldModelDataLoader',
    'EvaluationDataLoader',
    'WorldModel',
    'WorldModelConfig',
    'create_world_model',
    'FocalLoss',
    'ContrastiveLoss',
    'WorldModelLoss',
    'TrainingConfig',
    'Trainer',
    'Evaluator',
    'EvaluationMetrics',
    'run_evaluation',
    # Next-basket prediction (for RL)
    'NextBasketDataset',
    'NextBasketDataLoader',
    'NextBasketWorldModel',
    'NextBasketModelConfig',
    'NextBasketLoss',
    'NextBasketMetrics',
]
