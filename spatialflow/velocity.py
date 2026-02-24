"""
NicheVelocity — Niche Velocity and Acceleration

Definitions
───────────
Velocity    Vₙ(t) = Zₙ(t) − Zₙ(t−1)

    Measures first-order niche transcriptional motion in the latent manifold.
    A large |V| indicates a niche is rapidly shifting state — consistent with
    active inflammatory re-programming or resolution.

Acceleration  Aₙ(t) = Vₙ(t) − Vₙ(t−1)  =  Zₙ(t) − 2·Zₙ(t−1) + Zₙ(t−2)

    Detects imminent trajectory changes.  In PLP mice, high niche acceleration
    during Remission I predicts an impending Peak II relapse: the niche has
    not truly resolved but is *accelerating back* toward the inflammatory
    attractor state.

Inflammatory Axis Projection
────────────────────────────
An *inflammatory momentum scalar* is obtained by projecting velocity onto
a biological axis defined from the difference of prototypical activated vs
quiescent niche embeddings.  High positive projection = pro-inflammatory
momentum; negative = resolution momentum.

Temporal smoothing
──────────────────
Raw finite-difference velocities are noisy at single-cell resolution.
Optional Gaussian / exponential moving average smoothing over the latent
trajectory stabilises velocity estimates without introducing large temporal
lag.

Biological rationale:
    Velocity captures whether a niche is *moving toward* or *away from*
    the relapse-primed state.  Acceleration detects niches that appear
    resolved but retain a residual 'spring' back toward inflammation —
    the molecular correlate of disease priming.  Both quantities are
    inputs to the DivergenceScorer.
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
# NicheVelocity
# ---------------------------------------------------------------------------

class NicheVelocity(nn.Module):
    """
    Compute velocity and acceleration tensors over a sequence of latent
    embeddings Z₀ … Zₜ.

    Also projects velocities onto a learned or provided inflammatory axis
    to produce scalar inflammatory momentum scores per niche.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space Z.
    smooth_sigma : float
        Gaussian smoothing σ applied to Z trajectories before differencing.
        0 = no smoothing.
    learn_inf_axis : bool
        If True, the inflammatory axis is a learned parameter (initialised
        randomly and trained jointly).  If False, it must be set via
        set_inflammatory_axis().
    """

    def __init__(
        self,
        latent_dim: int = 64,
        smooth_sigma: float = 0.0,
        learn_inf_axis: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.smooth_sigma = smooth_sigma

        if learn_inf_axis:
            # Learnable axis, ‖axis‖ = 1 enforced at forward time
            self._inf_axis = nn.Parameter(torch.randn(latent_dim))
        else:
            self.register_buffer(
                "_inf_axis",
                torch.zeros(latent_dim),
            )
        self._axis_set = learn_inf_axis

    # ------------------------------------------------------------------
    # Inflammatory axis setup
    # ------------------------------------------------------------------

    def set_inflammatory_axis(
        self,
        activated_z: Tensor,
        quiescent_z: Tensor,
    ) -> None:
        """
        Define the inflammatory axis from prototypical niche embeddings.

        The axis is the unit vector pointing from the mean quiescent niche
        embedding toward the mean activated niche embedding:

            axis = (mean(activated_z) − mean(quiescent_z)) / ‖…‖

        Biologically, activated_z should represent:
            Activated microglia + cytotoxic T cells + reactive astrocytes

        Quiescent_z should represent:
            Homeostatic microglia + oligodendrocytes + intact myelinated tracts

        Parameters
        ----------
        activated_z : (N_act, latent_dim)
        quiescent_z : (N_qui, latent_dim)
        """
        mu_act = activated_z.mean(dim=0)
        mu_qui = quiescent_z.mean(dim=0)
        axis = mu_act - mu_qui
        norm = axis.norm()
        if norm < 1e-8:
            warnings.warn(
                "Inflammatory axis has near-zero norm — "
                "activated and quiescent centroids are identical.",
                stacklevel=2,
            )
        axis = axis / (norm + 1e-8)
        with torch.no_grad():
            self._inf_axis.copy_(axis)
        self._axis_set = True
        logger.info("Inflammatory axis set (‖axis‖ = %.4f).", norm.item())

    @property
    def inflammatory_axis(self) -> Tensor:
        axis = self._inf_axis
        return F.normalize(axis.unsqueeze(0), dim=-1).squeeze(0)

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _smooth_sequence(self, z_seq: Tensor) -> Tensor:
        """
        Apply 1-D Gaussian smoothing along the time axis.

        Parameters
        ----------
        z_seq : (T, N, D)
        """
        if self.smooth_sigma <= 0:
            return z_seq

        T = z_seq.shape[0]
        # Build Gaussian kernel over time window
        half = min(int(3 * self.smooth_sigma), T - 1)
        k_range = torch.arange(-half, half + 1, dtype=z_seq.dtype,
                               device=z_seq.device)
        kernel = torch.exp(-0.5 * (k_range / self.smooth_sigma) ** 2)
        kernel = kernel / kernel.sum()           # normalise

        # Convolve along T using 1-D convolution (treat N*D as batch channels)
        N, D = z_seq.shape[1], z_seq.shape[2]
        z_flat = z_seq.permute(1, 2, 0).reshape(N * D, 1, T)  # (N*D, 1, T)
        ker = kernel.view(1, 1, -1)
        pad = half
        smoothed = F.conv1d(z_flat, ker, padding=pad)          # (N*D, 1, T)
        smoothed = smoothed[:, :, :T].reshape(N, D, T).permute(2, 0, 1)
        return smoothed

    # ------------------------------------------------------------------
    # Velocity and acceleration
    # ------------------------------------------------------------------

    def compute_velocity(self, z_seq: Tensor) -> Tensor:
        """
        Vₙ(t) = Zₙ(t) − Zₙ(t−1)

        Parameters
        ----------
        z_seq : (T, N, D)

        Returns
        -------
        velocity : (T, N, D)
            Index 0 is zero-padded (no preceding timepoint).
        """
        z_smooth = self._smooth_sequence(z_seq)
        T, N, D = z_smooth.shape
        vel = torch.zeros_like(z_smooth)
        vel[1:] = z_smooth[1:] - z_smooth[:-1]
        return vel

    def compute_acceleration(self, z_seq: Tensor) -> Tensor:
        """
        Aₙ(t) = Vₙ(t) − Vₙ(t−1) = Zₙ(t) − 2Zₙ(t−1) + Zₙ(t−2)

        Parameters
        ----------
        z_seq : (T, N, D)

        Returns
        -------
        acceleration : (T, N, D)
            Indices 0 and 1 are zero-padded.
        """
        vel = self.compute_velocity(z_seq)
        T, N, D = vel.shape
        acc = torch.zeros_like(vel)
        acc[2:] = vel[2:] - vel[1:-1]
        return acc

    def velocity_magnitude(self, velocity: Tensor) -> Tensor:
        """
        ‖Vₙ(t)‖₂

        Parameters
        ----------
        velocity : (T, N, D)

        Returns
        -------
        magnitude : (T, N)
        """
        return velocity.norm(dim=-1)

    def acceleration_magnitude(self, acceleration: Tensor) -> Tensor:
        """
        ‖Aₙ(t)‖₂

        Parameters
        ----------
        acceleration : (T, N, D)

        Returns
        -------
        magnitude : (T, N)
        """
        return acceleration.norm(dim=-1)

    # ------------------------------------------------------------------
    # Inflammatory axis projection
    # ------------------------------------------------------------------

    def inflammatory_momentum(self, velocity: Tensor) -> Tensor:
        """
        Project velocity onto the inflammatory axis.

        score(n, t) = V(n, t) · axis

        Positive  → niche moving toward inflammatory state (pro-relapse)
        Negative  → niche moving away (resolution / remission momentum)

        Parameters
        ----------
        velocity : (T, N, D) or (N, D)

        Returns
        -------
        momentum : (T, N) or (N,)
        """
        if not self._axis_set:
            warnings.warn(
                "Inflammatory axis not set — momentum scores will be arbitrary. "
                "Call set_inflammatory_axis() before use.",
                stacklevel=2,
            )
        axis = self.inflammatory_axis            # (D,)
        if velocity.dim() == 3:
            # (T, N, D) × (D,) → (T, N)
            return (velocity * axis).sum(dim=-1)
        return (velocity * axis).sum(dim=-1)

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(self, z_seq: Tensor) -> Dict[str, Tensor]:
        """
        Compute all velocity-derived quantities from a latent sequence.

        Parameters
        ----------
        z_seq : (T, N, D)

        Returns
        -------
        dict with keys:
            velocity            : (T, N, D)
            acceleration        : (T, N, D)
            velocity_magnitude  : (T, N)
            accel_magnitude     : (T, N)
            inf_momentum        : (T, N)   inflammatory axis projection
            relapse_risk_score  : (T, N)   heuristic: |V| + 2·|A|
        """
        vel = self.compute_velocity(z_seq)
        acc = self.compute_acceleration(z_seq)
        vel_mag = self.velocity_magnitude(vel)
        acc_mag = self.acceleration_magnitude(acc)
        inf_mom = self.inflammatory_momentum(vel)

        # Heuristic relapse risk: high velocity + high acceleration = priming
        relapse_risk = vel_mag + 2.0 * acc_mag

        return {
            "velocity": vel,
            "acceleration": acc,
            "velocity_magnitude": vel_mag,
            "accel_magnitude": acc_mag,
            "inf_momentum": inf_mom,
            "relapse_risk_score": relapse_risk,
        }


# ---------------------------------------------------------------------------
# InflammatoryAxisDefiner — helper to build axis from cell-type groups
# ---------------------------------------------------------------------------

class InflammatoryAxisDefiner:
    """
    Utility to construct the multi-cell inflammatory axis from latent
    embeddings split by cell type annotation.

    Pro-inflammatory cell types:
        Activated microglia, cytotoxic T cells, reactive astrocytes,
        stressed OPCs

    Quiescent cell types:
        Homeostatic microglia, mature oligodendrocytes, vascular endothelium,
        intact neurons

    Parameters
    ----------
    activated_types : list of str
        Cell type labels for the activated / pro-inflammatory pole.
    quiescent_types : list of str
        Cell type labels for the quiescent pole.
    """

    ACTIVATED_DEFAULTS = [
        "activated_microglia",
        "cytotoxic_t_cell",
        "reactive_astrocyte",
        "stressed_opc",
        "inflammatory_macrophage",
    ]
    QUIESCENT_DEFAULTS = [
        "homeostatic_microglia",
        "mature_oligodendrocyte",
        "endothelial",
        "neuron",
        "pericyte",
    ]

    def __init__(
        self,
        activated_types: Optional[List[str]] = None,
        quiescent_types: Optional[List[str]] = None,
    ) -> None:
        self.activated_types = set(
            activated_types or self.ACTIVATED_DEFAULTS
        )
        self.quiescent_types = set(
            quiescent_types or self.QUIESCENT_DEFAULTS
        )

    def extract_axis(
        self,
        z: Tensor,
        cell_types: List[str],
        velocity_module: NicheVelocity,
    ) -> None:
        """
        Extract prototypical embeddings and set the inflammatory axis on the
        provided NicheVelocity module.

        Parameters
        ----------
        z : (N, D) latent embeddings for all cells
        cell_types : list of N cell type strings (lowercase)
        velocity_module : NicheVelocity instance to update
        """
        ct_arr = np.array([c.lower() for c in cell_types])

        act_mask = np.isin(ct_arr, list(self.activated_types))
        qui_mask = np.isin(ct_arr, list(self.quiescent_types))

        if act_mask.sum() == 0 or qui_mask.sum() == 0:
            logger.warning(
                "Could not find activated (%d) or quiescent (%d) cells "
                "matching known labels.  Axis not updated.",
                act_mask.sum(), qui_mask.sum(),
            )
            return

        act_z = z[torch.tensor(act_mask, dtype=torch.bool)]
        qui_z = z[torch.tensor(qui_mask, dtype=torch.bool)]
        velocity_module.set_inflammatory_axis(act_z, qui_z)
        logger.info(
            "Inflammatory axis computed from %d activated / %d quiescent cells.",
            act_mask.sum(), qui_mask.sum(),
        )
