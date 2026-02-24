# EAE-SpatialFlow v4 — Trajectory-Divergence Framework

A physics-informed spatio-temporal graph neural network framework for decoding
relapsing-remitting vs chronic trajectory divergence in EAE neuroinflammation
using Xenium spatial transcriptomics.

## Overview

EAE-SpatialFlow answers the central question:

> **Why do PLP mice relapse while MOG mice stabilise?**

The framework detects relapse-primed niches, trajectory bifurcations, inflammatory
momentum fields, and mechanistic gene drivers from million-cell Xenium datasets.

## Architecture

```
EAESpatialFlow
│
├── AnatomicalAligner        Geodesic landmark coordinate system
│   └── RadialBasisTransform (x,y) → (d_CC, d_GM, θ)
│
├── SpatialGraphBuilder      Hybrid spatial + LR interaction graph
│   └── LRDatabase           CellChat / Omnipath / custom
│
├── SpatioTemporalGNN        GATv2 VGAE + temporal bridge
│   ├── NicheMomentumEncoder 3-layer GATv2 → Z ∈ ℝ⁶⁴
│   ├── GRUTemporalBridge    GRU + attention over Z sequence
│   └── LatentNeuralODEBridge  Continuous-time alternative
│
├── NicheVelocity            V = Z(t) − Z(t−1), A = V(t) − V(t−1)
│
├── DivergenceScorer         ∇·V — bifurcation zone detection
│
├── TrajectoryContrastiveLoss  Hard triplet: PLP-Rem vs PLP-Peak vs MOG-Chronic
│
├── MechanisticAttributor    Captum Integrated Gradients
│
└── VisualizationEngine      UMAP, velocity fields, divergence heatmaps
```

## Key Biology

| Concept | Implementation |
|---|---|
| Geodesic distances | `MCP_Geometric` tissue-masked Dijkstra |
| Dorsal–ventral polarity | θ = 0 → dorsal midline |
| Bilateral symmetry | Lesion mirroring across midline |
| LR interaction edges | CellChat prior × ligand × receptor expression |
| Relapse priming | Acceleration Aₙ > 0 during remission |
| Bifurcation zones | ∇·V > threshold in peri-lesional WM |
| Hard triplet mining | Remission I (anchor) → Peak II (pos) vs MOG Chronic (neg) |

## Experimental Design

**MOG (Chronic):** baseline → non-symptomatic → early_onset → chronic_peak → chronic_long

**PLP (Relapsing-Remitting):** baseline → onset_I → onset_II → monophasic → peak_I → remission_I → peak_II → remission_II → peak_III

## Installation

```bash
pip install -e ".[full]"
# For Leiden clustering:
pip install leidenalg igraph
```

## Quick Start

```python
from spatialflow import EAESpatialFlow
import yaml

with open("configs/default_config.yaml") as f:
    config = yaml.safe_load(f)

pipeline = EAESpatialFlow(config=config, device="cuda", output_dir="outputs/")

# Preprocess and train
history = pipeline.train(sections, n_epochs=50)

# Inference
results = pipeline.predict(sections)

# Attribution on relapse-primed niches
attr = pipeline.attribute(plp_remission_section)
attr["top_genes"].to_csv("relapse_driving_genes.tsv", sep="\t")
```

## Training Script

```bash
python scripts/train.py \
    --data_dir /path/to/xenium_data \
    --config configs/default_config.yaml \
    --output_dir outputs/ \
    --device cuda \
    --n_epochs 50 \
    --validate \
    --attribute
```

## Output Artifacts

| File | Contents |
|---|---|
| `latent_embeddings.h5ad` | Z embeddings (AnnData) with metadata |
| `niche_velocity.parquet` | Per-cell velocity magnitude, inflammatory momentum |
| `divergence_scores.csv` | Per-cell divergence, bifurcation flags |
| `relapse_driving_genes.tsv` | Top genes by IG attribution |
| `figures/trajectory_umap_model.pdf` | MOG vs PLP UMAP |
| `figures/divergence_heatmap_all.pdf` | Spatial divergence map |
| `figures/relapse_probability_plp.pdf` | PLP relapse risk map |
| `figures/divergence_trajectory_plp.pdf` | Mean divergence over timepoints |

## Failure Mode Guards

- **Latent collapse**: KL annealing (β warmup over 500 steps)
- **Over-smoothing**: Residual connections in all GATv2 layers
- **Euclidean misalignment**: Geodesic `MCP_Geometric` distances
- **Timepoint imbalance**: Hard triplet mining + alignment loss
- **Graph over-connectivity**: Radius threshold + LR expression gates

## Tests

```bash
python -m pytest tests/test_components.py -v
```

## Future Extensions

- Human MS lesion data
- Spatial proteomics (CODEX, MIBI)
- MERFISH / CosMx data loaders
- Cross-species lesion alignment via RLCS normalisation
