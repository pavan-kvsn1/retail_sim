"""
Training module for RetailSim World Model.

This module provides the unified training pipeline that combines:
- Temporal split management (train/validation/test)
- Tensor caching for static embeddings
- WorldModelDataset and DataLoader with bucket batching
- World Model architecture (Mamba encoder + Transformer decoder)
- Multi-task loss functions (Focal, Contrastive, Auxiliary)
- Training loop with three-phase schedule
- Evaluation utilities for day/hour-level metrics

Architecture follows RetailSim_Data_Pipeline_and_World_Model_Design.md v7.6
"""

from .prepare_samples import enhance_temporal_metadata
from .prepare_tensor_cache import prepare_tensor_cache
from .dataset import WorldModelDataset, WorldModelDataLoader, EvaluationDataLoader
from .model import WorldModel, WorldModelConfig, create_world_model
from .losses import FocalLoss, ContrastiveLoss, WorldModelLoss
from .train import TrainingConfig, Trainer
from .evaluate import Evaluator, EvaluationMetrics, run_evaluation

__all__ = [
    # Data preparation
    'enhance_temporal_metadata',
    'prepare_tensor_cache',
    # Dataset
    'WorldModelDataset',
    'WorldModelDataLoader',
    'EvaluationDataLoader',
    # Model
    'WorldModel',
    'WorldModelConfig',
    'create_world_model',
    # Losses
    'FocalLoss',
    'ContrastiveLoss',
    'WorldModelLoss',
    # Training
    'TrainingConfig',
    'Trainer',
    # Evaluation
    'Evaluator',
    'EvaluationMetrics',
    'run_evaluation',
]
