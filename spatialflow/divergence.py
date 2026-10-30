"""
DivergenceScorer — Physics-Informed Vector Field Divergence

Physical motivation
───────────────────
Treating latent niche transitions as a vector field V(z, t) allows importing
fluid-dynamics concepts into transcriptomics:

    ∇ · V(n) ≈ 0    → locally stable niche, convergent trajectories
    ∇ · V(n) > 0    → source / bifurcation zone, divergent trajectories
    ∇ · V(n) < 0    → sink / attractor zone (resolution sink, fibrotic trap)

In EAE biology:
    Chronic stable (MOG): ∇ · V ≈ 0 everywhere — trajectories converge to a
    single chronic inflammatory attractor.

    Relapsing-remitting (PLP): ∇ · V > 0 in peri-lesional rim niches during
    remission — these niches *diverge* into either resolution (sink, rare)
    or re-activation (source, dominant) attractors.  High divergence
    zones are *relapse nurseries*.

Divergence estimation in high-dimensional latent space
────────────────────────────────────────────────────────
We use the Hutchinson stochastic trace estimator:

    ∇ · V ≈ E_v[ vᵀ · (∂V/∂z) · v ],   v ~ N(0, I)

This avoids computing the full Jacobian (D×D) and is differentiable.

Trajectory bifurcation detection
──────────────────────────────────
Niches with divergence > threshold AND located in the spatial transition
zone (intermediate d_CC, d_GM) are flagged as bifurcation seeds.

Downstream: PAGA / diffusion pseudotime analysis refines the topology of
the bifurcation tree.

Biological rationale:
    Divergence > 0 in peri-lesional zones during PLP remission periods marks
    the molecular indecision point — cells that could either resume
    pro-inflammatory programs or commit to repair.  Identifying these niches
    enables targeted intervention hypotheses.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hutchinson divergence estimator
# ---------------------------------------------------------------------------

def hutchinson_divergence(
    velocity_fn,
    z: Tensor,
    n_probes: int = 8,
) -> Tensor:
    """
    Stochastic trace estimator for the divergence of a vector field.

    ∇ · f ≈ (1/k) Σᵢ vᵢᵀ Jf(z) vᵢ,   vᵢ ~ Rademacher {±1}

    Using Rademacher vectors (rather than Gaussian) gives lower variance.

    Parameters
    ----------
    velocity_fn : callable
        Function z → v  (N, D) → (N, D).  Must be differentiable.
    z : (N, D)  latent positions at which to evaluate divergence
    n_probes : int   number of random probe vectors (higher = lower variance)

    Returns
    -------
    div : (N,)  estimated divergence per niche
    """
    N, D = z.shape
    div_accum = torch.zeros(N, device=z.device, dtype=z.dtype)

    for _ in range(n_probes):
        # Rademacher probe vector
        v = torch.randint(0, 2, (N, D), device=z.device, dtype=z.dtype) * 2 - 1

        z_req = z.requires_grad_(True)
        # Compute Jf(z) · v via JVP
        vf = velocity_fn(z_req)                # (N, D)

        # Compute vᵀ · (∂vf/∂z) using vjp: ∂(v·vf)/∂z_k = Σⱼ vⱼ (∂vf_j/∂z_k)
        grad_out = torch.autograd.grad(
            outputs=(vf * v).sum(),
            inputs=z_req,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]                                    # (N, D)

        # vᵀ Jf v = v · (Jf · v)ᵀ  — we computed Jf·v indirectly via scalar vjp
        # Actually we need the forward VJP approach: ∂(∑_j v_j f_j)/∂z_k = (v^T J)_k
        # So (v^T J) · v gives the trace estimate element-wise per-sample
        div_accum += (grad_out * v).sum(dim=-1)

    return div_accum / n_probes


# ---------------------------------------------------------------------------
# Finite-difference divergence in latent space (KNN-based)
# ---------------------------------------------------------------------------

def knn_divergence(
    z: Tensor,
    velocity: Tensor,
    k: int = 15,
) -> Tensor:
    """
    Estimate divergence using finite differences in k-nearest-neighbour
    latent space.

    For each niche n, the divergence is approximated as the mean outward
    velocity component toward its k nearest neighbours:

        ∇ · V(n) ≈ (1/k) Σᵢ∈kNN(n) (V(n) − V(i)) · (z(n) − z(i)) / ‖z(n) − z(i)‖²

    This is a particle-based divergence estimator (similar to SPH).

    Parameters
    ----------
    z : (N, D)
    velocity : (N, D)
    k : int

    Returns
    -------
    div : (N,)
    """
    N, D = z.shape
    if N <= 1:
        return torch.zeros(N, device=z.device, dtype=z.dtype)

    k_eff = min(max(int(k), 1), N - 1)
    # Pairwise distances (chunked for memory efficiency).
    # MPS has tighter memory behavior, so use a smaller chunk there.
    chunk_cap = 256 if z.device.type == "mps" else 512
    chunk = min(chunk_cap, N)
    divs = []
    z_norm2 = (z ** 2).sum(dim=1)  # (N,)
    z_t = z.transpose(0, 1)        # (D, N)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        z_chunk = z[start:end]           # (B, D)
        v_chunk = velocity[start:end]    # (B, D)

        # Distances from chunk to all N without materialising (B, N, D).
        z_chunk_norm2 = (z_chunk ** 2).sum(dim=1, keepdim=True)    # (B, 1)
        dists2 = z_chunk_norm2 + z_norm2.unsqueeze(0) - 2.0 * (z_chunk @ z_t)
        dists2 = dists2.clamp_min(0.0)

        # Exclude self distances for the current chunk rows.
        local = torch.arange(end - start, device=z.device)
        dists2[local, start + local] = float("inf")

        _, nn_idx = dists2.topk(k_eff, dim=1, largest=False)   # (B, k)

        div_chunk = torch.zeros(end - start, device=z.device, dtype=z.dtype)
        for b in range(end - start):
            nbrs = nn_idx[b]                         # (k,)
            delta_z = z_chunk[b].unsqueeze(0) - z[nbrs]     # (k, D)
            delta_v = v_chunk[b].unsqueeze(0) - velocity[nbrs]  # (k, D)
            dist2 = dists2[b, nbrs].clamp(min=1e-10)        # (k,)
            # Directional divergence: (ΔV · Δz) / |Δz|²
            div_chunk[b] = ((delta_v * delta_z).sum(dim=-1) / dist2).mean()

        divs.append(div_chunk)

    return torch.cat(divs, dim=0)


# ---------------------------------------------------------------------------
# DivergenceScorer
# ---------------------------------------------------------------------------

class DivergenceScorer(nn.Module):
    """
    Physics-informed divergence scoring over the latent velocity field.

    Combines KNN-based divergence estimation with optional learned
    divergence correction (a small neural network that refines raw
    finite-difference estimates using graph context).

    Parameters
    ----------
    latent_dim : int
    knn_k : int                KNN for finite-difference divergence
    n_hutchinson_probes : int  Rademacher probes for learned divergence
    bifurcation_threshold : float
        Divergence score above which a niche is flagged as a bifurcation seed.
    learn_correction : bool
        If True, add a learnable correction MLP to raw divergence.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        knn_k: int = 15,
        n_hutchinson_probes: int = 8,
        bifurcation_threshold: float = 0.5,
        learn_correction: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.knn_k = knn_k
        self.n_hutchinson_probes = n_hutchinson_probes
        self.bifurcation_threshold = bifurcation_threshold

        if learn_correction:
            # MLP: (z, raw_div) → correction scalar
            self.correction_mlp = nn.Sequential(
                nn.Linear(latent_dim + 1, 64),
                nn.ELU(),
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, 1),
            )
        else:
            self.correction_mlp = None

    def forward(
        self,
        z: Tensor,
        velocity: Tensor,
        anatomical_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute divergence scores for all niches.

        Parameters
        ----------
        z : (N, D)   latent niche embeddings (from one timepoint)
        velocity : (N, D)   velocity vectors
        anatomical_features : (N, 6)   optional RLCS features for spatial context

        Returns
        -------
        dict with keys:
            divergence_raw       : (N,)   KNN finite-difference divergence
            divergence_corrected : (N,)   after learned correction (if enabled)
            bifurcation_mask     : (N,)   bool, above threshold
            relapse_nursery_mask : (N,)   bool, bifurcation + peri-lesional location
            divergence_map       : (N,)   final divergence for downstream use
        """
        # 1. KNN divergence
        div_raw = knn_divergence(z, velocity, k=self.knn_k)   # (N,)

        # 2. Learned correction
        if self.correction_mlp is not None:
            inp = torch.cat([z, div_raw.unsqueeze(-1)], dim=-1)
            correction = self.correction_mlp(inp).squeeze(-1)  # (N,)
            div_corrected = div_raw + correction
        else:
            div_corrected = div_raw

        # 3. Normalise for interpretability
        div_norm = (div_corrected - div_corrected.mean()) / (div_corrected.std() + 1e-8)

        # 4. Bifurcation flags
        bifurcation_mask = div_norm > self.bifurcation_threshold

        # 5. Relapse nursery: bifurcation AND peri-lesional location
        # Peri-lesional = moderate d_CC (not in CC, not far WM periphery)
        relapse_nursery_mask = bifurcation_mask.clone()
        if anatomical_features is not None:
            # Accept numpy or tensor input; keep comparisons on the model device.
            anatomical_features = torch.as_tensor(
                anatomical_features, dtype=z.dtype, device=z.device
            )
            d_cc_norm = anatomical_features[:, 0]  # d_CC (µm)
            # Peri-lesional = d_CC in 50–300 µm range (configurable)
            peri_lesional = (d_cc_norm > 50.0) & (d_cc_norm < 300.0)
            relapse_nursery_mask = bifurcation_mask & peri_lesional

        return {
            "divergence_raw": div_raw,
            "divergence_corrected": div_corrected,
            "divergence_norm": div_norm,
            "bifurcation_mask": bifurcation_mask,
            "relapse_nursery_mask": relapse_nursery_mask,
            "divergence_map": div_norm,    # primary output for visualisation
        }

    # ------------------------------------------------------------------
    # Divergence loss (physics constraint)
    # ------------------------------------------------------------------

    def divergence_constraint_loss(
        self,
        divergence: Tensor,
        model_flags: Tensor,
    ) -> Tensor:
        """
        Physics-informed divergence constraint loss.

        For MOG (chronic stable) niches: ∇ · V should be ≈ 0.
        For PLP (relapsing) niches: allow ∇ · V > 0.

        L_div = mean( (∇·V)² [MOG niches] )
              + max(0, −∇·V) [PLP niches]  ← penalise negative divergence in RR

        Parameters
        ----------
        divergence : (N,)   divergence scores
        model_flags : (N,)  0=MOG, 1=PLP

        Returns
        -------
        loss : scalar Tensor
        """
        mog_mask = (model_flags == 0)
        plp_mask = (model_flags == 1)

        loss = torch.tensor(0.0, device=divergence.device)

        if mog_mask.sum() > 0:
            # MOG: penalise any divergence from zero
            loss = loss + (divergence[mog_mask] ** 2).mean()

        if plp_mask.sum() > 0:
            # PLP: penalise negative divergence (sinks should be rare in RR)
            loss = loss + F.relu(-divergence[plp_mask]).mean() * 0.5

        return loss

    # ------------------------------------------------------------------
    # Trajectory bifurcation analysis
    # ------------------------------------------------------------------

    def detect_bifurcation_timepoint(
        self,
        divergence_seq: Tensor,
        timepoint_names: List[str],
    ) -> Dict:
        """
        Identify the timepoint at which divergence peaks — the putative
        bifurcation event in the disease trajectory.

        Parameters
        ----------
        divergence_seq : (T, N)  divergence scores per timepoint
        timepoint_names : list of T strings

        Returns
        -------
        dict:
            peak_timepoint     : str
            peak_timepoint_idx : int
            mean_divergence    : (T,) Tensor
            bifurcation_frac   : (T,) fraction of niches above threshold
        """
        mean_div = divergence_seq.mean(dim=1)          # (T,)
        bif_frac = (divergence_seq > self.bifurcation_threshold).float().mean(dim=1)

        peak_idx = int(mean_div.argmax().item())
        peak_name = timepoint_names[peak_idx] if peak_idx < len(timepoint_names) else str(peak_idx)

        logger.info(
            "Bifurcation detected at timepoint %d ('%s'), mean_div=%.4f",
            peak_idx, peak_name, mean_div[peak_idx].item(),
        )
        return {
            "peak_timepoint": peak_name,
            "peak_timepoint_idx": peak_idx,
            "mean_divergence": mean_div,
            "bifurcation_frac": bif_frac,
        }


# ---------------------------------------------------------------------------
# PAGA-style trajectory graph (post-hoc)
# ---------------------------------------------------------------------------

class PAGAGraph:
    """
    Partition-based Graph Abstraction (PAGA) for trajectory topology.

    Groups niches into Leiden clusters and builds a graph where edge weight
    between clusters A and B reflects the fraction of KNN edges crossing
    the A–B boundary.  High inter-cluster connectivity = high probability of
    trajectory transition.

    This implementation is a lightweight numpy-based version.
    For full PAGA, use scanpy.tl.paga() on the AnnData output.

    Parameters
    ----------
    cluster_labels : (N,) int array   Leiden cluster assignment
    z : (N, D)  latent embeddings
    """

    def __init__(
        self,
        cluster_labels: np.ndarray,
        z: np.ndarray,
        knn_k: int = 15,
    ) -> None:
        self.labels = cluster_labels
        self.z = z
        self.n_clusters = int(cluster_labels.max()) + 1
        self.knn_k = knn_k
        self._connectivity = None

    def compute_connectivity(self) -> np.ndarray:
        """
        Build (n_clusters × n_clusters) PAGA connectivity matrix.

        Returns
        -------
        connectivity : (K, K)  float — normalised by geometric mean of cluster sizes
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(self.z)
        _, nn_idx = tree.query(self.z, k=self.knn_k + 1)
        nn_idx = nn_idx[:, 1:]   # exclude self

        K = self.n_clusters
        n_inter = np.zeros((K, K), dtype=np.float64)
        n_cells = np.bincount(self.labels, minlength=K).astype(np.float64)

        for i in range(len(self.z)):
            ci = self.labels[i]
            for j in nn_idx[i]:
                cj = self.labels[j]
                if ci != cj:
                    n_inter[ci, cj] += 1
                    n_inter[cj, ci] += 1

        # Normalise by geometric mean of cluster sizes
        sizes = n_cells[:, None] * n_cells[None, :]
        connectivity = n_inter / (np.sqrt(sizes) + 1e-10)
        np.fill_diagonal(connectivity, 0)
        self._connectivity = connectivity
        return connectivity

    @property
    def connectivity(self) -> np.ndarray:
        if self._connectivity is None:
            self.compute_connectivity()
        return self._connectivity
