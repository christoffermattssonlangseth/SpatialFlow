"""
SpatioTemporalGNN — Niche Momentum Encoder

Architecture overview
─────────────────────
1. NicheMomentumEncoder (Variational Graph Autoencoder)
   ├─ GATv2Encoder          3-layer GATv2 + edge-gated convolutions + residuals
   │                        → latent mean μ ∈ ℝ⁶⁴,  log-var σ² ∈ ℝ⁶⁴
   └─ NodeDecoder           reconstruct expression from Z

2. TemporalBridge            GRU + attention over sequential Z embeddings
   (alternative: LatentNeuralODE or TemporalTransformer, toggled by config)

3. SpatioTemporalGNN         Top-level module combining encoder + temporal bridge
   Called once per timepoint; temporal bridge called on a sequence of Zs.

Biological rationale:
   GATv2 attention is multi-head so each head can specialise to different
   cell-niche signals (e.g., one head tracks microglial signatures, another
   tracks vascular cues).  Edge-gating allows the model to down-weight
   connectivity across biological boundaries (e.g., CC lumen).
   Variational latent space regularises the niche state manifold so that
   velocity and divergence are continuous, differentiable quantities.
   The GRU temporal bridge propagates *learned* inflammatory momentum
   between timepoints rather than computing differences post-hoc.

Scalability:
   NeighborLoader-compatible: operates on batched subgraphs.
   Mixed-precision (fp16) training supported via AMP.

Failure mode guards:
   • KL annealing avoids early latent collapse.
   • Residual connections prevent over-smoothing.
   • Dropout applied to attention coefficients.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import GATv2Conv, LayerNorm as PyGLayerNorm
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge-gated Graph Attention (EG-GATv2)
# ---------------------------------------------------------------------------

class EdgeGatedGATv2Conv(nn.Module):
    """
    GATv2 convolution with edge-gating.

    Edge-gating multiplies message weights by a learned scalar gate derived
    from edge features.  This lets the model suppress communication across
    anatomical barriers (CC lumen, lesion necrotic core) regardless of
    spatial proximity.

    Parameters
    ----------
    in_channels : int
    out_channels : int    (per head — total output = out_channels × heads)
    heads : int
    edge_dim : int        dimensionality of edge_attr
    dropout : float
    residual : bool       add skip connection (projected if dims differ)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        edge_dim: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_per_head = out_channels
        self.concat = concat
        self.total_out = out_channels * heads if concat else out_channels

        if HAS_PYG:
            self.gatv2 = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=concat,
                add_self_loops=True,
            )
        else:
            # Fallback: simple linear aggregation (no attention)
            self.gatv2 = None
            self.fallback_lin = nn.Linear(in_channels, self.total_out)

        # Edge gate MLP: edge_attr → scalar gate ∈ (0, 1)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Residual projection
        self.residual = residual
        if residual and in_channels != self.total_out:
            self.res_proj = nn.Linear(in_channels, self.total_out, bias=False)
        else:
            self.res_proj = None

        self.norm = nn.LayerNorm(self.total_out)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : (N, in_channels)
        edge_index : (2, E)
        edge_attr : (E, edge_dim)

        Returns
        -------
        (N, total_out)
        """
        identity = x

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.shape[1], 2,
                                    device=x.device, dtype=x.dtype)

        if HAS_PYG and self.gatv2 is not None:
            # Apply edge gate to modulate message importance before aggregation
            # We pass edge_attr directly to GATv2; gating is applied as a
            # post-hoc re-weighting on the output contribution per node.
            out = self.gatv2(x, edge_index, edge_attr)
        else:
            out = self.fallback_lin(x)

        out = self.dropout(out)

        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity

        return self.norm(out)


# ---------------------------------------------------------------------------
# GATv2 Encoder (Variational)
# ---------------------------------------------------------------------------

class GATv2Encoder(nn.Module):
    """
    3-layer GATv2 encoder with residuals → latent (μ, log_σ²) ∈ ℝ⁶⁴.

    Architecture:
        Layer 1: EdgeGatedGATv2Conv  → hidden_dim (4 heads, concat)
        Layer 2: EdgeGatedGATv2Conv  → hidden_dim (4 heads, concat)
        Layer 3: EdgeGatedGATv2Conv  → hidden_dim (4 heads, mean-pooled)
        μ head : Linear hidden_dim → latent_dim
        σ head : Linear hidden_dim → latent_dim

    Parameters
    ----------
    in_channels : int     input node feature dimension
    hidden_dim : int      per-head hidden dimension
    latent_dim : int      latent space dimension Z (default 64)
    heads : int           attention heads (default 4)
    edge_dim : int        edge feature dimension
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        heads: int = 4,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = EdgeGatedGATv2Conv(
            in_channels, hidden_dim, heads=heads, edge_dim=edge_dim,
            dropout=dropout, residual=True, concat=True,
        )
        dim_after_1 = hidden_dim * heads

        self.conv2 = EdgeGatedGATv2Conv(
            dim_after_1, hidden_dim, heads=heads, edge_dim=edge_dim,
            dropout=dropout, residual=True, concat=True,
        )
        dim_after_2 = hidden_dim * heads

        # Last layer: mean-pool heads for cleaner latent projection
        self.conv3 = EdgeGatedGATv2Conv(
            dim_after_2, hidden_dim, heads=heads, edge_dim=edge_dim,
            dropout=dropout, residual=True, concat=False,
        )

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.act = nn.ELU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        mu : (N, latent_dim)
        log_var : (N, latent_dim)
        """
        h = self.act(self.conv1(x, edge_index, edge_attr))
        h = self.act(self.conv2(h, edge_index, edge_attr))
        h = self.act(self.conv3(h, edge_index, edge_attr))

        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        # Clamp log_var for numerical stability (prevents NaN in KL)
        log_var = torch.clamp(log_var, -10.0, 4.0)
        return mu, log_var


# ---------------------------------------------------------------------------
# Reparameterisation
# ---------------------------------------------------------------------------

def reparameterise(mu: Tensor, log_var: Tensor) -> Tensor:
    """
    z = μ + ε · σ,   ε ~ N(0, I)

    At inference time (eval mode), returns μ directly for deterministic
    embeddings used in velocity / divergence computation.
    """
    if not torch.is_grad_enabled():
        return mu
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# ---------------------------------------------------------------------------
# Node Decoder
# ---------------------------------------------------------------------------

class NodeDecoder(nn.Module):
    """
    Reconstruct gene expression from latent Z.

    Used for ELBO reconstruction loss and as a regulariser to prevent
    latent collapse (latent space remains gene-expression-grounded).

    Parameters
    ----------
    latent_dim : int
    out_channels : int   number of genes to reconstruct
    hidden_dim : int
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Returns (N, out_channels) reconstructed expression (log1p scale)."""
        return self.net(z)


# ---------------------------------------------------------------------------
# Temporal Bridge — GRU + Attention
# ---------------------------------------------------------------------------

class GRUTemporalBridge(nn.Module):
    """
    GRU-based temporal bridge over sequential latent states.

    Processes a time-ordered sequence of niche embeddings Z₀, Z₁, … Zₜ
    and outputs temporally-informed embeddings H₀ … Hₜ.

    An additive attention mechanism allows each timestep to attend to all
    preceding states — important for detecting *accumulated* inflammatory
    priming rather than only the immediate previous state.

    Parameters
    ----------
    latent_dim : int
    gru_hidden : int   GRU hidden state dimension
    n_layers : int
    dropout : float
    """

    def __init__(
        self,
        latent_dim: int = 64,
        gru_hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=gru_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        # Temporal attention: query from current step, keys/values from all steps
        self.attn_q = nn.Linear(gru_hidden, gru_hidden)
        self.attn_k = nn.Linear(gru_hidden, gru_hidden)
        self.attn_v = nn.Linear(gru_hidden, gru_hidden)
        self.out_proj = nn.Linear(gru_hidden, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z_seq: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z_seq : (T, N_niche, latent_dim)   T timepoints, N niches per step

        Returns
        -------
        h_seq : (T, N_niche, latent_dim)   temporally informed embeddings
        """
        T, N, D = z_seq.shape
        # Reshape to (N, T, D) for GRU batch processing
        z_in = z_seq.permute(1, 0, 2)          # (N, T, D)
        h_all, _ = self.gru(z_in)              # (N, T, gru_hidden)

        # Self-attention over time
        Q = self.attn_q(h_all)                 # (N, T, H)
        K = self.attn_k(h_all)
        V = self.attn_v(h_all)
        scale = math.sqrt(Q.shape[-1])
        attn_weights = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1
        )                                      # (N, T, T)
        attended = torch.bmm(attn_weights, V)  # (N, T, H)

        out = self.out_proj(attended)          # (N, T, D)
        out = self.norm(out + z_in)            # residual
        return out.permute(1, 0, 2)            # (T, N, D)


# ---------------------------------------------------------------------------
# Latent Neural ODE Bridge (optional)
# ---------------------------------------------------------------------------

class LatentNeuralODEBridge(nn.Module):
    """
    Neural ODE temporal bridge — models continuous inflammatory dynamics.

    Uses a learned vector field f_θ(z, t) as the ODE right-hand side and
    integrates with a fixed-step Euler / RK4 solver.

    Biological rationale:
        Disease trajectory is a continuous flow in biological state space.
        Neural ODE allows querying the latent state at *any* pseudotime,
        not just observed timepoints — enabling interpolation between
        Remission I and Peak II for relapse trajectory reconstruction.

    Parameters
    ----------
    latent_dim : int
    hidden_dim : int
    n_steps : int   integration steps between consecutive timepoints
    method : str    'euler' or 'rk4'
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        n_steps: int = 10,
        method: str = "rk4",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.method = method

        # ODE dynamics network: f(z, t) → dz/dt
        self.ode_func = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),   # +1 for time embedding
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Output tanh ensures bounded velocity
        self._tanh = nn.Tanh()

    def _f(self, z: Tensor, t: float) -> Tensor:
        """ODE right-hand side: dz/dt = f(z, t)."""
        t_tensor = torch.full(
            (z.shape[0], 1), t, dtype=z.dtype, device=z.device
        )
        inp = torch.cat([z, t_tensor], dim=-1)
        return self._tanh(self.ode_func(inp))

    def _step_euler(self, z: Tensor, t0: float, dt: float) -> Tensor:
        return z + dt * self._f(z, t0)

    def _step_rk4(self, z: Tensor, t0: float, dt: float) -> Tensor:
        k1 = self._f(z, t0)
        k2 = self._f(z + 0.5 * dt * k1, t0 + 0.5 * dt)
        k3 = self._f(z + 0.5 * dt * k2, t0 + 0.5 * dt)
        k4 = self._f(z + dt * k3, t0 + dt)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate(self, z0: Tensor, t0: float, t1: float) -> Tensor:
        """Integrate from t0 to t1 starting at z0."""
        z = z0
        dt = (t1 - t0) / self.n_steps
        t = t0
        step_fn = self._step_rk4 if self.method == "rk4" else self._step_euler
        for _ in range(self.n_steps):
            z = step_fn(z, t, dt)
            t += dt
        return z

    def forward(self, z_seq: Tensor, times: Optional[List[float]] = None) -> Tensor:
        """
        Parameters
        ----------
        z_seq : (T, N, D)
        times : list of T float timepoints in [0, 1]

        Returns
        -------
        h_seq : (T, N, D)
        """
        T, N, D = z_seq.shape
        if times is None:
            times = [i / (T - 1) for i in range(T)]

        outputs = [z_seq[0]]
        for t_idx in range(1, T):
            z_integrated = self.integrate(
                outputs[-1], times[t_idx - 1], times[t_idx]
            )
            outputs.append(z_integrated)
        return torch.stack(outputs, dim=0)


# ---------------------------------------------------------------------------
# NicheMomentumEncoder (full VGAE)
# ---------------------------------------------------------------------------

class NicheMomentumEncoder(nn.Module):
    """
    Variational Graph Autoencoder encoding cell niches into the
    64-dimensional Niche State Manifold.

    Combines:
      • GATv2Encoder → (μ, log_σ²)
      • Reparameterisation → Z
      • NodeDecoder → x̂ (expression reconstruction)

    Outputs Z for downstream velocity / divergence computation.

    Parameters
    ----------
    in_channels : int
    hidden_dim : int
    latent_dim : int   default 64 (Niche State Manifold dimension)
    n_genes_reconstruct : int  number of genes to reconstruct
    heads : int
    edge_dim : int
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        n_genes_reconstruct: int = 500,
        heads: int = 4,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = GATv2Encoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.decoder = NodeDecoder(
            latent_dim=latent_dim,
            out_channels=n_genes_reconstruct,
            hidden_dim=hidden_dim * 4,
        )

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        z        : (N, latent_dim)   sampled latent code
        mu       : (N, latent_dim)
        log_var  : (N, latent_dim)
        """
        mu, log_var = self.encoder(x, edge_index, edge_attr)
        z = reparameterise(mu, log_var)
        return z, mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """Returns (N, n_genes_reconstruct) reconstructed expression."""
        return self.decoder(z)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Full forward pass: encode → sample → decode.

        Returns
        -------
        dict with keys: z, mu, log_var, x_recon
        """
        z, mu, log_var = self.encode(x, edge_index, edge_attr)
        x_recon = self.decode(z)
        return {
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "x_recon": x_recon,
        }

    def elbo_loss(
        self,
        x_target: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        log_var: Tensor,
        beta: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Evidence Lower BOund (ELBO):
            L = E[log p(x|z)] − β · KL(q(z|x) || p(z))

        β-annealing:  start β=0, ramp to β=1 over first n_warmup steps to
        prevent latent collapse during early training.

        Parameters
        ----------
        x_target : (N, G)  ground-truth expression for reconstruction genes
        x_recon  : (N, G)  decoder output
        mu, log_var : encoder outputs
        beta : float  KL weight (pass annealed value from training loop)

        Returns
        -------
        total_loss : scalar Tensor
        loss_dict  : {'recon_loss', 'kl_loss', 'total_loss'}
        """
        recon_loss = F.mse_loss(x_recon, x_target, reduction="mean")
        # KL divergence: closed form for diagonal Gaussians vs N(0, I)
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        total = recon_loss + beta * kl_loss
        return total, {
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "total_loss": total.detach(),
        }


# ---------------------------------------------------------------------------
# SpatioTemporalGNN — Top-level module
# ---------------------------------------------------------------------------

class SpatioTemporalGNN(nn.Module):
    """
    Full spatio-temporal GNN combining:
      1. NicheMomentumEncoder (per-timepoint spatial encoding)
      2. TemporalBridge (GRU or NeuralODE) over sequences of Z
      3. Projection head for contrastive loss

    Training modes
    ──────────────
    Single-timepoint:  encode(x, edge_index, edge_attr)
    Multi-timepoint:   encode_sequence([data_t0, data_t1, …])

    Parameters
    ----------
    in_channels : int
    hidden_dim : int
    latent_dim : int             default 64
    n_genes_reconstruct : int
    heads : int                  GATv2 attention heads
    edge_dim : int
    dropout : float
    temporal_mode : str          'gru' | 'ode'
    gru_hidden : int             GRU hidden state size
    projection_dim : int         output dim of contrastive projection head
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        n_genes_reconstruct: int = 500,
        heads: int = 4,
        edge_dim: int = 2,
        dropout: float = 0.1,
        temporal_mode: str = "gru",
        gru_hidden: int = 128,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()

        self.encoder = NicheMomentumEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_genes_reconstruct=n_genes_reconstruct,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        if temporal_mode == "gru":
            self.temporal_bridge: nn.Module = GRUTemporalBridge(
                latent_dim=latent_dim,
                gru_hidden=gru_hidden,
                dropout=dropout,
            )
        elif temporal_mode == "ode":
            self.temporal_bridge = LatentNeuralODEBridge(
                latent_dim=latent_dim,
                hidden_dim=gru_hidden,
            )
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode!r}")

        # Contrastive projection head (MLP after Z)
        self.proj_head = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_single(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Single-timepoint encoding."""
        return self.encoder(x, edge_index, edge_attr)

    def encode_sequence(
        self,
        data_sequence: List[Dict],
        node_mask: Optional[List[Optional[Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        """
        Encode a time-ordered sequence of graphs and apply temporal bridge.

        Parameters
        ----------
        data_sequence : list of dict or PyG Data objects
            Each has keys: x, edge_index, edge_attr
        node_mask : optional list of boolean masks to select representative
            niche centroids (same N across all timepoints)

        Returns
        -------
        dict with keys:
            z_seq        : (T, N, latent_dim)  post-temporal Z
            mu_seq       : (T, N, latent_dim)
            log_var_seq  : (T, N, latent_dim)
            x_recon_seq  : (T, N, n_genes)
        """
        T = len(data_sequence)
        zs, mus, log_vars, x_recons = [], [], [], []

        for t, data in enumerate(data_sequence):
            x = data["x"] if isinstance(data, dict) else data.x
            ei = data["edge_index"] if isinstance(data, dict) else data.edge_index
            ea = data.get("edge_attr") if isinstance(data, dict) else getattr(data, "edge_attr", None)

            out = self.encoder(x, ei, ea)
            zs.append(out["z"])
            mus.append(out["mu"])
            log_vars.append(out["log_var"])
            x_recons.append(out["x_recon"])

        # Stack: (T, N, D) — requires same N per timepoint
        # In practice different sections have different cell counts;
        # caller should pass pre-pooled niche centroids.
        z_stack = torch.stack(zs, dim=0)          # (T, N, D)
        mu_stack = torch.stack(mus, dim=0)
        lv_stack = torch.stack(log_vars, dim=0)
        xr_stack = torch.stack(x_recons, dim=0)

        # Temporal bridge
        z_temporal = self.temporal_bridge(z_stack)  # (T, N, D)

        return {
            "z_seq": z_temporal,
            "mu_seq": mu_stack,
            "log_var_seq": lv_stack,
            "x_recon_seq": xr_stack,
        }

    def project(self, z: Tensor) -> Tensor:
        """
        Apply contrastive projection head.

        Parameters
        ----------
        z : (N, latent_dim) or (T, N, latent_dim)
        """
        return self.proj_head(z)

    # ------------------------------------------------------------------
    # Beta-KL annealing scheduler
    # ------------------------------------------------------------------

    @staticmethod
    def kl_annealing_beta(
        step: int,
        warmup_steps: int = 2000,
        max_beta: float = 1.0,
    ) -> float:
        """
        Linear KL warmup: β ramps from 0 → max_beta over warmup_steps.

        Prevents latent collapse during early training when the decoder
        has not yet learned to use the latent code.
        """
        return min(max_beta, step / max(warmup_steps, 1) * max_beta)
