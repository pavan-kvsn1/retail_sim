"""
Data Pipeline Module
====================
Section 2 of RetailSim: Data Pipeline Architecture

Stages:
1. Price Derivation Pipeline - Derive prices from SPEND/QUANTITY
2. Product Graph Construction - Build heterogeneous product graph
3. Customer-Store Affinity - Compute customer-store relationships
4. Mission Pattern Extraction - Extract historical shopping patterns
"""

from .stage1_price_derivation import PriceDerivationPipeline
from .stage2_product_graph import ProductGraphBuilder
from .stage3_customer_store_affinity import CustomerStoreAffinityPipeline
from .stage4_mission_patterns import MissionPatternPipeline

__all__ = [
    'PriceDerivationPipeline',
    'ProductGraphBuilder',
    'CustomerStoreAffinityPipeline',
    'MissionPatternPipeline'
]
