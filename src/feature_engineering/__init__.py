"""
Feature Engineering Module
==========================
Section 3 of RetailSim: Feature Engineering Design

Layers:
1. Pseudo-Brand Inference - Infer brand-like clusters from price/category
2. Fourier Price Encoding - 64d continuous price features
3. Graph Embeddings (GraphSAGE) - 256d product representations
4. Customer History Encoding - 160d customer behavioral signatures
5. Store Context Features - 96d store representations
"""

from .layer1_pseudo_brand import PseudoBrandInference
from .layer2_fourier_price import FourierPriceEncoder
from .layer3_graph_embeddings import GraphSAGEEncoder
from .layer4_customer_history import CustomerHistoryEncoder
from .layer5_store_context import StoreContextEncoder

__all__ = [
    'PseudoBrandInference',
    'FourierPriceEncoder',
    'GraphSAGEEncoder',
    'CustomerHistoryEncoder',
    'StoreContextEncoder'
]
