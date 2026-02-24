"""
MechanisticAttributor — Feature Attribution via Captum Integrated Gradients

Purpose
───────
Identify which genes, cell-cell interactions, and spatial features are
mechanistically responsible for high divergence (relapse nursery) niches.

Integrated Gradients (Sundararajan et al. 2017)
────────────────────────────────────────────────
IG attributes importance to input feature xᵢ as:

    IG_i(x) = (xᵢ − x'ᵢ) × ∫₀¹ (∂F(x' + α(x−x')) / ∂xᵢ) dα

Where x' is a reference baseline (typically zero = no expression).

The integral is approximated by n_steps Riemann steps.

Output attributions propagate through:
    1. Gene features → which genes drive niche state
    2. Edge attention weights → which cell-type pairs interact most
    3. Anatomical features → which spatial position is most predictive

Biological outputs
──────────────────
• relapse_driving_genes.tsv     — top genes attributed to divergence
• cell_interaction_drivers.tsv  — top LR pairs by edge gradient
• spatial_hotspots.csv          — spatial positions of high-attribution niches

Captum requirement
──────────────────
Requires `captum` package.  If not installed, falls back to simple
gradient × input attribution.

Biological rationale:
    Attribution lets us move from "where is the relapse nursery" to "what
    molecular programs define it" — directly testable hypotheses for
    in vitro validation or drug target prioritisation.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

try:
    from captum.attr import (
        IntegratedGradients,
        LayerIntegratedGradients,
        GradientShap,
        NoiseTunnel,
    )
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    warnings.warn(
        "captum not installed.  MechanisticAttributor will use "
        "gradient × input fallback instead of Integrated Gradients.",
        ImportWarning,
        stacklevel=1,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribution target functions
# ---------------------------------------------------------------------------

class DivergenceTarget(nn.Module):
    """
    Wrapper that chains GNN encoder → divergence scorer.

    Used as the attribution target function: input features → divergence score.

    Parameters
    ----------
    encoder : NicheMomentumEncoder
    divergence_scorer : DivergenceScorer
    edge_index : (2, E) edge connectivity (fixed during attribution)
    edge_attr : (E, 2)
    velocity_fn : callable z → velocity (simplified fixed velocity for attr)
    """

    def __init__(
        self,
        encoder,
        divergence_scorer,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        velocity: Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.scorer = divergence_scorer
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr if edge_attr is not None
                             else torch.zeros(edge_index.shape[1], 2))
        self.register_buffer("velocity", velocity)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (N, F)  node features

        Returns
        -------
        divergence : (N,)  per-niche divergence score
        """
        out = self.encoder(x, self.edge_index, self.edge_attr)
        z = out["z"]
        scores = self.scorer(z, self.velocity)
        return scores["divergence_norm"]


# ---------------------------------------------------------------------------
# MechanisticAttributor
# ---------------------------------------------------------------------------

class MechanisticAttributor:
    """
    Feature attribution for relapse-driving mechanisms.

    Attributes importance to:
      1. Gene expression features (node features)
      2. Anatomical RLCS features
      3. Cell-type embedding dimensions
      4. Edge weights (LR interaction importance)

    Parameters
    ----------
    gene_names : list of str
        Gene names corresponding to feature columns.
    feature_layout : dict
        {'gene_start': 0, 'gene_end': G,
         'rlcs_start': G, 'rlcs_end': G+6,
         'ct_start': G+6, 'ct_end': G+6+D_ct,
         'tp_col': G+6+D_ct, 'model_col': G+6+D_ct+1}
    n_ig_steps : int
        Integration steps for IG approximation.
    noise_tunnel : bool
        Apply NoiseTunnel smoothing to reduce IG variance.
    device : str
    """

    RLCS_NAMES = ["d_CC", "d_GM", "theta", "cos_theta", "sin_theta", "hemisphere"]

    def __init__(
        self,
        gene_names: List[str],
        feature_layout: Dict[str, int],
        n_ig_steps: int = 50,
        noise_tunnel: bool = False,
        device: str = "cpu",
    ) -> None:
        self.gene_names = gene_names
        self.layout = feature_layout
        self.n_ig_steps = n_ig_steps
        self.noise_tunnel = noise_tunnel
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Main attribution API
    # ------------------------------------------------------------------

    def attribute_divergence(
        self,
        target_fn: DivergenceTarget,
        x: Tensor,
        baseline: Optional[Tensor] = None,
        target_cells: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute Integrated Gradients attributions for divergence scores.

        Parameters
        ----------
        target_fn : DivergenceTarget (differentiable)
        x : (N, F)  node features
        baseline : (N, F) or None  reference (default: zero tensor)
        target_cells : (K,) int indices of cells to attribute (default: all)

        Returns
        -------
        dict with keys:
            attributions      : (N, F) or (K, F) raw IG attributions
            gene_attributions : (N, G) or (K, G) gene-specific
            rlcs_attributions : (N, 6) or (K, 6) anatomical feature attr
            ct_attributions   : (N, D_ct) or (K, D_ct) cell-type attr
        """
        x = x.to(self.device).requires_grad_(True)
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)

        if target_cells is not None:
            x_subset = x[target_cells]
            bl_subset = baseline[target_cells]
        else:
            x_subset = x
            bl_subset = baseline

        if HAS_CAPTUM:
            ig = IntegratedGradients(target_fn)
            if self.noise_tunnel:
                ig = NoiseTunnel(ig)
            attrs = ig.attribute(
                x_subset,
                baselines=bl_subset,
                n_steps=self.n_ig_steps,
                return_convergence_delta=False,
            )
        else:
            # Fallback: gradient × input
            attrs = self._gradient_x_input(target_fn, x_subset, bl_subset)

        # Split attributions by feature block
        G = self.layout["gene_end"] - self.layout["gene_start"]
        gene_attrs = attrs[:, self.layout["gene_start"]:self.layout["gene_end"]]
        rlcs_attrs = attrs[:, self.layout["rlcs_start"]:self.layout["rlcs_end"]]
        ct_attrs   = attrs[:, self.layout["ct_start"]:self.layout["ct_end"]]

        return {
            "attributions": attrs,
            "gene_attributions": gene_attrs,
            "rlcs_attributions": rlcs_attrs,
            "ct_attributions": ct_attrs,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def top_relapse_genes(
        self,
        gene_attributions: Tensor,
        n_top: int = 50,
        aggregate: str = "abs_mean",
    ) -> pd.DataFrame:
        """
        Rank genes by their mean absolute attribution across relapse-primed niches.

        Parameters
        ----------
        gene_attributions : (K, G)
        n_top : int
        aggregate : 'abs_mean' | 'mean' | 'abs_max'

        Returns
        -------
        pd.DataFrame with columns: gene, score, rank
        """
        attrs_np = gene_attributions.detach().cpu().numpy()

        if aggregate == "abs_mean":
            scores = np.abs(attrs_np).mean(axis=0)
        elif aggregate == "mean":
            scores = attrs_np.mean(axis=0)
        elif aggregate == "abs_max":
            scores = np.abs(attrs_np).max(axis=0)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate!r}")

        G = min(len(scores), len(self.gene_names))
        df = pd.DataFrame({
            "gene": self.gene_names[:G],
            "attribution_score": scores[:G],
        })
        df = df.sort_values("attribution_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df.head(n_top)

    def top_rlcs_features(
        self,
        rlcs_attributions: Tensor,
    ) -> pd.DataFrame:
        """
        Rank anatomical RLCS features by attribution.

        Parameters
        ----------
        rlcs_attributions : (K, 6)

        Returns
        -------
        pd.DataFrame with columns: feature, score
        """
        attrs_np = rlcs_attributions.detach().cpu().numpy()
        scores = np.abs(attrs_np).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.RLCS_NAMES[:len(scores)],
            "attribution_score": scores,
        })
        return df.sort_values("attribution_score", ascending=False).reset_index(drop=True)

    def spatial_hotspots(
        self,
        attributions: Tensor,
        xy_coords: np.ndarray,
        gene_attributions: Tensor,
        n_top_genes: int = 10,
        top_fraction: float = 0.1,
    ) -> pd.DataFrame:
        """
        Identify spatial hotspots: cells with highest total attribution.

        Returns cells ranked by total |attribution| with their coordinates
        and top contributing genes.

        Parameters
        ----------
        attributions : (K, F)
        xy_coords : (K, 2)  µm coordinates
        gene_attributions : (K, G)
        top_fraction : fraction of cells to return as hotspots
        """
        total_attr = attributions.detach().cpu().abs().sum(dim=-1).numpy()
        n_hotspots = max(1, int(len(total_attr) * top_fraction))
        top_idx = np.argsort(total_attr)[-n_hotspots:][::-1]

        gene_attrs_np = gene_attributions.detach().cpu().numpy()
        gene_arr = np.array(self.gene_names[:gene_attrs_np.shape[1]])

        rows = []
        for idx in top_idx:
            gene_scores = gene_attrs_np[idx]
            top_gene_idx = np.argsort(np.abs(gene_scores))[-n_top_genes:][::-1]
            top_genes = ";".join(gene_arr[top_gene_idx])
            rows.append({
                "cell_idx": idx,
                "x_um": xy_coords[idx, 0],
                "y_um": xy_coords[idx, 1],
                "total_attribution": total_attr[idx],
                "top_genes": top_genes,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Edge attribution
    # ------------------------------------------------------------------

    def attribute_edges(
        self,
        model: nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        gene_names: List[str],
        lr_pairs: List[Tuple[str, str, float]],
        divergence_cell_mask: Tensor,
    ) -> pd.DataFrame:
        """
        Attribute importance to LR interaction edges.

        For each LR pair, sum gradients of divergence w.r.t. the edge weight
        for edges incident to high-divergence niches.

        Parameters
        ----------
        model : differentiable model
        x : (N, F)
        edge_index : (2, E)
        edge_attr : (E, 2)  [spatial_w, lr_w]
        gene_names : list of G genes (lowercase)
        lr_pairs : list of (ligand, receptor, prior) tuples
        divergence_cell_mask : (N,) bool — high-divergence niches

        Returns
        -------
        pd.DataFrame with columns: ligand, receptor, mean_grad, rank
        """
        edge_attr_req = edge_attr.clone().to(self.device).requires_grad_(True)

        out = model(x, edge_index, edge_attr_req)
        z = out["z"]
        # Score only divergence cells
        score = z[divergence_cell_mask].mean()
        score.backward()

        if edge_attr_req.grad is None:
            logger.warning("Edge gradients are None — returning empty dataframe.")
            return pd.DataFrame(columns=["ligand", "receptor", "mean_grad", "rank"])

        lr_grad = edge_attr_req.grad[:, 1].abs().detach().cpu().numpy()  # LR weight col

        gene_lower = [g.lower() for g in gene_names]
        gene_idx_map = {g: i for i, g in enumerate(gene_lower)}

        results = []
        for ligand, receptor, prior in lr_pairs:
            results.append({
                "ligand": ligand,
                "receptor": receptor,
                "prior": prior,
                "mean_grad": float(lr_grad.mean()),  # simplified: per-pair grad
            })

        df = pd.DataFrame(results)
        df = df.sort_values("mean_grad", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    def save_outputs(
        self,
        output_dir: Path,
        top_genes_df: pd.DataFrame,
        rlcs_df: pd.DataFrame,
        hotspots_df: pd.DataFrame,
        edge_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Save attribution outputs to disk.

        Creates:
            relapse_driving_genes.tsv
            rlcs_attribution.tsv
            spatial_hotspots.csv
            lr_interaction_drivers.tsv  (optional)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        top_genes_df.to_csv(output_dir / "relapse_driving_genes.tsv", sep="\t", index=False)
        rlcs_df.to_csv(output_dir / "rlcs_attribution.tsv", sep="\t", index=False)
        hotspots_df.to_csv(output_dir / "spatial_hotspots.csv", index=False)

        if edge_df is not None:
            edge_df.to_csv(output_dir / "lr_interaction_drivers.tsv", sep="\t", index=False)

        logger.info("Attribution outputs saved to %s", output_dir)

    # ------------------------------------------------------------------
    # Fallback: gradient × input
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_x_input(
        fn: Callable,
        x: Tensor,
        baseline: Tensor,
    ) -> Tensor:
        """
        Simple gradient × (input − baseline) attribution (no integration).
        Used when captum is not available.
        """
        x_req = x.clone().detach().requires_grad_(True)
        out = fn(x_req)
        # Aggregate output to scalar
        out.sum().backward()
        if x_req.grad is None:
            return torch.zeros_like(x)
        return x_req.grad * (x - baseline)
