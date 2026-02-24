"""
EAE-SpatialFlow v4 — Trajectory-Divergence Framework

A physics-informed spatio-temporal graph neural network framework for
decoding relapsing vs chronic trajectory divergence in EAE spatial
transcriptomics (Xenium).

Module hierarchy:
    EAESpatialFlow          Master pipeline (pipeline.py)
    ├── AnatomicalAligner   Geodesic coordinate normalisation (aligner.py)
    ├── RadialBasisTransform
    ├── SpatialGraphBuilder Hybrid spatial + LR graph (graph_builder.py)
    ├── SpatioTemporalGNN   GATv2 VGAE + temporal bridge (model.py)
    ├── NicheVelocity       Velocity + acceleration (velocity.py)
    ├── DivergenceScorer    Physics-informed vector field (divergence.py)
    ├── TrajectoryContrastiveLoss  Hard triplet loss (losses.py)
    ├── MechanisticAttributor      Captum IG attribution (attributor.py)
    └── VisualizationEngine         All plots (visualization.py)
"""

from .aligner import AnatomicalAligner, RadialBasisTransform
from .graph_builder import LRDatabase, SpatialGraphBuilder
from .model import (
    NicheMomentumEncoder,
    SpatioTemporalGNN,
    GRUTemporalBridge,
    LatentNeuralODEBridge,
)
from .velocity import NicheVelocity, InflammatoryAxisDefiner
from .divergence import DivergenceScorer, PAGAGraph
from .losses import TrajectoryContrastiveLoss, HardTripletMiner
from .attributor import MechanisticAttributor, DivergenceTarget
from .visualization import VisualizationEngine
from .pipeline import EAESpatialFlow

__version__ = "4.0.0"
__all__ = [
    "EAESpatialFlow",
    "AnatomicalAligner",
    "RadialBasisTransform",
    "LRDatabase",
    "SpatialGraphBuilder",
    "NicheMomentumEncoder",
    "SpatioTemporalGNN",
    "GRUTemporalBridge",
    "LatentNeuralODEBridge",
    "NicheVelocity",
    "InflammatoryAxisDefiner",
    "DivergenceScorer",
    "PAGAGraph",
    "TrajectoryContrastiveLoss",
    "HardTripletMiner",
    "MechanisticAttributor",
    "DivergenceTarget",
    "VisualizationEngine",
]
