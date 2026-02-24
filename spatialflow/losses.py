"""
TrajectoryContrastiveLoss — Trajectory-Aware Triplet Loss with Hard Mining

Core innovation
───────────────
Standard contrastive losses use random triplets.  Here the triplets are
chosen to specifically encode *relapse-vs-stabilisation* trajectory
divergence:

    Anchor   : PLP Remission I niche (remission state)
    Positive : PLP Peak II niche     (relapse state — same animal, later)
    Negative : MOG Chronic Long niche (stable chronic, different trajectory)

Objective:
    Pull (anchor → positive) closer in Z-space than (anchor → negative),
    forcing the model to learn the molecular programs that distinguish
    relapse-primed remission from genuine chronic stabilisation.

Hard Triplet Mining
───────────────────
Naïve random triplets give weak gradients when the model already separates
trajectories well.  Hard mining selects:

    Hard negatives : MOG chronic niches most similar to the PLP remission
                     anchor (closest transcriptionally to the anchor, yet
                     known to follow a different trajectory).

    Hard positives : PLP Peak II niches closest to the Remission I anchor
                     (most subtle, hard-to-learn relapse signals).

Biological rationale:
    During PLP Remission I, some niches appear transcriptionally similar to
    MOG chronic niches.  The hard triplet loss forces the model to find the
    latent dimensions that *distinguish* these apparently similar states —
    the molecular footprints of covert relapse priming.

Additional losses
─────────────────
NicheAlignmentLoss:  Ensures same-model same-timepoint niches cluster
                     together across animals (cross-animal alignment).

TemporalConsistencyLoss:  Penalises large discontinuities in velocity
                           between consecutive timepoints.

VelocityDirectionLoss:   Penalise PLP remission niches whose velocity
                          points away from the relapse attractor.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triplet definitions
# ---------------------------------------------------------------------------

# Timepoint name constants (must match AnatomicalAligner.TIMEPOINTS_*)
_PLP_REMISSION_I = "remission_I"
_PLP_PEAK_II     = "peak_II"
_MOG_CHRONIC_LONG = "chronic_long"


# ---------------------------------------------------------------------------
# Hard triplet sampler
# ---------------------------------------------------------------------------

class HardTripletMiner:
    """
    Mine hard triplets from a batch of niche embeddings.

    Strategy
    ────────
    For each anchor (PLP Remission I niche):
      • Hard positive: PLP Peak II niche that is *closest* to the anchor
        but temporally proximal to a relapse (i.e., not trivially different).
      • Hard negative: MOG Chronic Long niche that is *closest* to the anchor
        in embedding space (most confusable stable niche).

    Parameters
    ----------
    margin : float  triplet loss margin
    semi_hard : bool
        If True, select semi-hard negatives: negatives with
        d(anchor, negative) > d(anchor, positive) but within margin.
        Provides more informative gradients than pure hard negatives.
    max_triplets_per_anchor : int
        Cap on triplets per anchor to avoid memory explosion.
    """

    def __init__(
        self,
        margin: float = 1.0,
        semi_hard: bool = True,
        max_triplets_per_anchor: int = 32,
    ) -> None:
        self.margin = margin
        self.semi_hard = semi_hard
        self.max_triplets = max_triplets_per_anchor

    def mine(
        self,
        z: Tensor,
        timepoints: Tensor,
        model_flags: Tensor,
        timepoint_names_plp: List[str],
        timepoint_names_mog: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Mine anchor / positive / negative index triplets.

        Parameters
        ----------
        z : (N, D)  latent embeddings for the batch
        timepoints : (N,)  int  timepoint index (0-based within model)
        model_flags : (N,)  int8  0=MOG, 1=PLP
        timepoint_names_plp : ordered list of PLP timepoint names
        timepoint_names_mog : ordered list of MOG timepoint names

        Returns
        -------
        anchors, positives, negatives : 1-D LongTensor of cell indices
        """
        N = z.shape[0]
        device = z.device

        # Timepoint index lookups
        try:
            rem_I_idx  = timepoint_names_plp.index(_PLP_REMISSION_I)
            peak_II_idx = timepoint_names_plp.index(_PLP_PEAK_II)
            chr_long_idx = timepoint_names_mog.index(_MOG_CHRONIC_LONG)
        except ValueError as e:
            logger.warning("Triplet mining skipped — %s", e)
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        plp_mask = (model_flags == 1)
        mog_mask = (model_flags == 0)

        anchor_mask   = plp_mask & (timepoints == rem_I_idx)
        positive_mask = plp_mask & (timepoints == peak_II_idx)
        negative_mask = mog_mask & (timepoints == chr_long_idx)

        anchor_idx   = torch.where(anchor_mask)[0]
        positive_idx = torch.where(positive_mask)[0]
        negative_idx = torch.where(negative_mask)[0]

        if len(anchor_idx) == 0 or len(positive_idx) == 0 or len(negative_idx) == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        # Pairwise distances
        z_a = z[anchor_idx]   # (A, D)
        z_p = z[positive_idx] # (P, D)
        z_n = z[negative_idx] # (Neg, D)

        # d(a, p) and d(a, n)
        ap_dists = torch.cdist(z_a, z_p, p=2)   # (A, P)
        an_dists = torch.cdist(z_a, z_n, p=2)   # (A, Neg)

        triplet_anchors, triplet_pos, triplet_neg = [], [], []

        for a_local in range(len(anchor_idx)):
            ap = ap_dists[a_local]   # (P,)
            an = an_dists[a_local]   # (Neg,)

            # Hard positive: closest positive
            hard_p_local = ap.argmin()
            d_ap = ap[hard_p_local]

            if self.semi_hard:
                # Semi-hard negatives: d_an > d_ap  and  d_an < d_ap + margin
                semi_mask = (an > d_ap) & (an < d_ap + self.margin)
                cand_neg = torch.where(semi_mask)[0]
                if len(cand_neg) == 0:
                    # Fall back to hardest negative
                    cand_neg = an.argsort()[:self.max_triplets]
            else:
                # Hard negatives: smallest d_an
                cand_neg = an.argsort()[:self.max_triplets]

            n_triplets = min(len(cand_neg), self.max_triplets)
            for ni in range(n_triplets):
                triplet_anchors.append(anchor_idx[a_local])
                triplet_pos.append(positive_idx[hard_p_local])
                triplet_neg.append(negative_idx[cand_neg[ni]])

        if not triplet_anchors:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        return (
            torch.stack(triplet_anchors),
            torch.stack(triplet_pos),
            torch.stack(triplet_neg),
        )


# ---------------------------------------------------------------------------
# TrajectoryContrastiveLoss
# ---------------------------------------------------------------------------

class TrajectoryContrastiveLoss(nn.Module):
    """
    Trajectory-aware contrastive loss combining:
      1. Hard-mined triplet loss
      2. Niche alignment loss (cross-animal clustering)
      3. Temporal consistency loss
      4. Optional velocity direction loss

    Total loss:
        L = λ_trip · L_triplet
          + λ_align · L_align
          + λ_temp · L_temporal
          + λ_vel · L_velocity

    Parameters
    ----------
    margin : float
        Triplet loss margin.
    lambda_triplet : float
    lambda_align : float
    lambda_temporal : float
    lambda_velocity : float
    semi_hard : bool
    temperature : float
        InfoNCE temperature (used if using NT-Xent variant).
    """

    def __init__(
        self,
        margin: float = 1.0,
        lambda_triplet: float = 1.0,
        lambda_align: float = 0.3,
        lambda_temporal: float = 0.2,
        lambda_velocity: float = 0.1,
        semi_hard: bool = True,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.lam_trip = lambda_triplet
        self.lam_align = lambda_align
        self.lam_temp = lambda_temporal
        self.lam_vel = lambda_velocity
        self.temperature = temperature

        self.miner = HardTripletMiner(
            margin=margin, semi_hard=semi_hard
        )

    # ------------------------------------------------------------------
    # Component losses
    # ------------------------------------------------------------------

    def triplet_loss(
        self,
        z: Tensor,
        anchor_idx: Tensor,
        pos_idx: Tensor,
        neg_idx: Tensor,
    ) -> Tensor:
        """
        Margin-based triplet loss:
            L = mean( max(0, d(a, p) − d(a, n) + margin) )
        """
        if len(anchor_idx) == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        z_a = z[anchor_idx]
        z_p = z[pos_idx]
        z_n = z[neg_idx]

        d_ap = F.pairwise_distance(z_a, z_p, p=2)
        d_an = F.pairwise_distance(z_a, z_n, p=2)

        loss = F.relu(d_ap - d_an + self.margin).mean()
        return loss

    def alignment_loss(
        self,
        z: Tensor,
        model_flags: Tensor,
        timepoints: Tensor,
        n_pairs: int = 256,
    ) -> Tensor:
        """
        Cross-animal niche alignment:  same-model same-timepoint cells
        should be closer than cross-model or cross-timepoint cells.

        Uses NT-Xent (normalised temperature-scaled cross-entropy) on
        random positive pairs sampled from same (model, timepoint) group.
        """
        device = z.device
        # Build group labels
        group = model_flags.long() * 100 + timepoints.long()
        unique_groups = group.unique()

        pos_i, pos_j = [], []
        for g in unique_groups:
            idx = torch.where(group == g)[0]
            if len(idx) < 2:
                continue
            # Sample min(n_pairs, |idx|//2) pairs
            n = min(n_pairs // len(unique_groups) + 1, len(idx) // 2)
            perm = torch.randperm(len(idx), device=device)[:2 * n]
            pos_i.append(idx[perm[:n]])
            pos_j.append(idx[perm[n:2 * n]])

        if not pos_i:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_i = torch.cat(pos_i)
        pos_j = torch.cat(pos_j)

        # NT-Xent on sampled pairs
        z_i = F.normalize(z[pos_i], dim=-1)
        z_j = F.normalize(z[pos_j], dim=-1)

        N_pairs = z_i.shape[0]
        sim = torch.mm(z_i, z_j.T) / self.temperature   # (N, N)
        labels = torch.arange(N_pairs, device=device)
        loss = (F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.T, labels)) / 2
        return loss

    def temporal_consistency_loss(
        self,
        z_seq: Tensor,
    ) -> Tensor:
        """
        Penalise large discontinuities in the latent trajectory:
            L = mean( ‖Z(t) − Z(t−1)‖² ) over consecutive timepoints

        This regularises the velocity field to be smooth (no teleporting).

        Parameters
        ----------
        z_seq : (T, N, D)
        """
        if z_seq.shape[0] < 2:
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)

        diffs = z_seq[1:] - z_seq[:-1]         # (T-1, N, D)
        return (diffs ** 2).mean()

    def velocity_direction_loss(
        self,
        velocity: Tensor,
        model_flags: Tensor,
        inf_axis: Tensor,
    ) -> Tensor:
        """
        For PLP niches: velocity should have positive projection onto the
        inflammatory axis during peak timepoints (relapse momentum preserved).
        For MOG niches: velocity should be small (stable trajectory).

        Parameters
        ----------
        velocity : (N, D)
        model_flags : (N,)  0=MOG, 1=PLP
        inf_axis : (D,)   unit vector

        Returns
        -------
        scalar loss
        """
        proj = (velocity * inf_axis.unsqueeze(0)).sum(dim=-1)  # (N,)

        plp_mask = (model_flags == 1)
        mog_mask = (model_flags == 0)

        loss = torch.tensor(0.0, device=velocity.device)

        # MOG: small total velocity magnitude
        if mog_mask.sum() > 0:
            loss = loss + velocity[mog_mask].norm(dim=-1).mean() * 0.5

        # PLP: at least some positive momentum (not suppressed to zero)
        if plp_mask.sum() > 0:
            # Penalise strongly negative projection (velocity away from inflammation)
            loss = loss + F.relu(-proj[plp_mask]).mean()

        return loss

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: Tensor,
        model_flags: Tensor,
        timepoints: Tensor,
        timepoint_names_plp: List[str],
        timepoint_names_mog: List[str],
        z_seq: Optional[Tensor] = None,
        velocity: Optional[Tensor] = None,
        inf_axis: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute combined trajectory-aware contrastive loss.

        Parameters
        ----------
        z : (N, D)  projected niche embeddings for current batch
        model_flags : (N,)   int  0=MOG, 1=PLP
        timepoints : (N,)    int  timepoint indices
        timepoint_names_plp : list of PLP timepoint names (for mining)
        timepoint_names_mog : list of MOG timepoint names
        z_seq : (T, N, D)    optional — for temporal consistency
        velocity : (N, D)    optional — for velocity direction loss
        inf_axis : (D,)      optional — inflammatory axis unit vector

        Returns
        -------
        total_loss : scalar Tensor
        loss_dict  : component losses
        """
        # 1. Mine hard triplets
        a_idx, p_idx, n_idx = self.miner.mine(
            z, timepoints, model_flags,
            timepoint_names_plp, timepoint_names_mog,
        )
        loss_triplet = self.triplet_loss(z, a_idx, p_idx, n_idx)

        # 2. Alignment loss
        loss_align = self.alignment_loss(z, model_flags, timepoints)

        # 3. Temporal consistency
        if z_seq is not None:
            loss_temp = self.temporal_consistency_loss(z_seq)
        else:
            loss_temp = torch.tensor(0.0, device=z.device)

        # 4. Velocity direction
        if velocity is not None and inf_axis is not None:
            loss_vel = self.velocity_direction_loss(velocity, model_flags, inf_axis)
        else:
            loss_vel = torch.tensor(0.0, device=z.device)

        total = (
            self.lam_trip * loss_triplet
            + self.lam_align * loss_align
            + self.lam_temp * loss_temp
            + self.lam_vel * loss_vel
        )

        loss_dict = {
            "triplet_loss": loss_triplet.detach(),
            "alignment_loss": loss_align.detach(),
            "temporal_consistency_loss": loss_temp.detach(),
            "velocity_direction_loss": loss_vel.detach(),
            "n_triplets": torch.tensor(len(a_idx), dtype=torch.float32),
            "total_contrastive_loss": total.detach(),
        }

        return total, loss_dict
