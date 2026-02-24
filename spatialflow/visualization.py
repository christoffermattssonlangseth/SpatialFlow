"""
VisualizationEngine — Spatial and Latent Space Visualizations

Outputs
───────
1. Niche trajectory UMAP
   Latent Z coloured by: timepoint, model (MOG/PLP), niche cluster,
   relapse risk score.  Arrows show velocity direction in UMAP space.

2. Velocity vector fields
   Spatial scatter of cells with velocity arrows overlaid on tissue
   coordinates (µm).  One plot per timepoint.

3. Divergence heatmaps
   Continuous divergence score plotted as 2-D heatmap over tissue
   coordinates.  Bifurcation zones highlighted.

4. Relapse probability maps
   Per-cell relapse risk score interpolated to spatial heatmap.

5. Trajectory UMAP with bifurcation overlay
   PAGA graph edges overlaid on UMAP.

All plots:
  • Use anatomical coordinates where applicable (µm, dorsal up)
  • Publication-quality figures (300 dpi, vector-friendly)
  • Return matplotlib Figure objects (can be saved externally)

Biological rationale:
    Spatial visualisation on tissue coordinates (not arbitrary UMAP positions)
    allows direct comparison with histology (IHC, haematoxylin-eosin staining)
    and validates that computational relapse nurseries correspond to
    perivascular infiltration zones or demyelination lesion rims.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for server-side rendering
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrow
    from matplotlib.collections import LineCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not installed — visualizations disabled.", ImportWarning)

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("umap-learn not installed — UMAP projections unavailable.", ImportWarning)

try:
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

_MODEL_COLOURS = {"MOG": "#2166AC", "PLP": "#D6604D"}  # blue / red

_PLP_TIMEPOINT_CMAP = [
    "#440154", "#3B528B", "#21908C",
    "#5DC863", "#FDE725",            # baseline → peak_III
    "#F77F00", "#E31A1C", "#BD0026", "#800026",
]
_MOG_TIMEPOINT_CMAP = [
    "#DEEBF7", "#9ECAE1", "#4292C6", "#2171B5", "#084594",
]

_DIVERGENCE_CMAP = "RdBu_r"   # blue=sink, red=source/bifurcation
_VELOCITY_CMAP   = "viridis"
_RELAPSE_CMAP    = "YlOrRd"


def _check_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib required for visualization.")


# ---------------------------------------------------------------------------
# VisualizationEngine
# ---------------------------------------------------------------------------

class VisualizationEngine:
    """
    Generate all required visualizations for EAE-SpatialFlow.

    Parameters
    ----------
    output_dir : Path | str
        Directory where figures are saved.
    figsize : tuple
        Default figure size in inches.
    dpi : int
        Figure DPI for saved files.
    dorsal_up : bool
        If True, flip y-axis so dorsal is at top (standard neuroanatomy).
    """

    def __init__(
        self,
        output_dir: Union[Path, str] = "outputs/figures",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150,
        dorsal_up: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        self.dorsal_up = dorsal_up

    # ------------------------------------------------------------------
    # 1. Niche trajectory UMAP
    # ------------------------------------------------------------------

    def plot_trajectory_umap(
        self,
        z: np.ndarray,
        labels: np.ndarray,
        colour_by: str = "model",
        label_map: Optional[Dict] = None,
        velocities_2d: Optional[np.ndarray] = None,
        title: str = "Niche Trajectory UMAP",
        save_name: str = "trajectory_umap.pdf",
        umap_coords: Optional[np.ndarray] = None,
        umap_kwargs: Optional[Dict] = None,
    ) -> Optional["plt.Figure"]:
        """
        UMAP projection of latent Z coloured by labels.

        Parameters
        ----------
        z : (N, D)  latent embeddings
        labels : (N,)  int or str labels
        colour_by : 'model' | 'timepoint' | 'cluster' | 'score'
        label_map : dict mapping label value → string name (for legend)
        velocities_2d : (N, 2)  pre-projected velocity in UMAP space
        title : str
        save_name : str
        umap_coords : (N, 2) pre-computed UMAP coords (skip recomputation)
        """
        _check_mpl()

        if umap_coords is None:
            if not HAS_UMAP:
                logger.warning("UMAP not available — returning None.")
                return None
            kw = umap_kwargs or {"n_neighbors": 30, "min_dist": 0.3, "random_state": 42}
            reducer = UMAP(**kw)
            umap_coords = reducer.fit_transform(z)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Scatter
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))

        for i, lb in enumerate(unique_labels):
            mask = labels == lb
            name = label_map.get(lb, str(lb)) if label_map else str(lb)
            ax.scatter(
                umap_coords[mask, 0], umap_coords[mask, 1],
                c=[cmap(i)], s=4, alpha=0.6, linewidths=0,
                label=name, rasterized=True,
            )

        # Velocity arrows in UMAP space
        if velocities_2d is not None:
            step = max(1, len(umap_coords) // 500)   # subsample for readability
            ax.quiver(
                umap_coords[::step, 0], umap_coords[::step, 1],
                velocities_2d[::step, 0], velocities_2d[::step, 1],
                scale=25, width=0.003, alpha=0.4, color="black",
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.legend(markerscale=3, framealpha=0.5, fontsize=8, loc="best")
        ax.set_aspect("equal")
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 2. Spatial velocity vector field
    # ------------------------------------------------------------------

    def plot_velocity_field(
        self,
        xy_coords: np.ndarray,
        velocity: np.ndarray,
        velocity_magnitude: Optional[np.ndarray] = None,
        anatomical_features: Optional[np.ndarray] = None,
        title: str = "Niche Velocity Field",
        save_name: str = "velocity_field.pdf",
        subsample: int = 300,
    ) -> Optional["plt.Figure"]:
        """
        Spatial scatter with velocity arrows.

        Parameters
        ----------
        xy_coords : (N, 2)  µm coordinates
        velocity : (N, 2)   UMAP-projected or PCA-projected velocity vectors
        velocity_magnitude : (N,)   for colour-coding arrow magnitude
        anatomical_features : (N, 6)  RLCS — d_CC used for background colouring
        subsample : int  max number of arrows to draw (readability)
        """
        _check_mpl()

        fig, ax = plt.subplots(figsize=self.figsize)

        # Background: all cells coloured by velocity magnitude
        if velocity_magnitude is not None:
            sc = ax.scatter(
                xy_coords[:, 0], xy_coords[:, 1],
                c=velocity_magnitude, cmap=_VELOCITY_CMAP,
                s=2, alpha=0.5, linewidths=0, rasterized=True,
            )
            plt.colorbar(sc, ax=ax, label="|Velocity| (latent units)")
        else:
            ax.scatter(xy_coords[:, 0], xy_coords[:, 1],
                       s=2, alpha=0.3, color="grey", linewidths=0, rasterized=True)

        # Velocity arrows (subsampled)
        idx = np.random.choice(len(xy_coords), min(subsample, len(xy_coords)),
                               replace=False)
        vel_scale = max(1e-8, np.abs(velocity[idx]).max())
        ax.quiver(
            xy_coords[idx, 0], xy_coords[idx, 1],
            velocity[idx, 0] / vel_scale,
            velocity[idx, 1] / vel_scale,
            scale=15, width=0.004, color="black", alpha=0.7,
        )

        if self.dorsal_up:
            ax.invert_yaxis()

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x (µm)", fontsize=11)
        ax.set_ylabel("y (µm)", fontsize=11)
        ax.set_aspect("equal")
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 3. Divergence heatmap
    # ------------------------------------------------------------------

    def plot_divergence_heatmap(
        self,
        xy_coords: np.ndarray,
        divergence: np.ndarray,
        bifurcation_mask: Optional[np.ndarray] = None,
        title: str = "Divergence Heatmap",
        save_name: str = "divergence_heatmap.pdf",
        grid_resolution: int = 200,
    ) -> Optional["plt.Figure"]:
        """
        Interpolated 2-D divergence heatmap over tissue space.

        Parameters
        ----------
        xy_coords : (N, 2)  µm
        divergence : (N,)   divergence scores
        bifurcation_mask : (N,) bool  overlay bifurcation nuclei
        grid_resolution : int  interpolation grid points per axis
        """
        _check_mpl()

        from scipy.interpolate import griddata

        x, y = xy_coords[:, 0], xy_coords[:, 1]
        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), divergence, (Xi, Yi), method="linear")

        fig, ax = plt.subplots(figsize=self.figsize)

        vmax = np.nanpercentile(np.abs(divergence), 99)
        im = ax.pcolormesh(
            Xi, Yi, Zi,
            cmap=_DIVERGENCE_CMAP,
            vmin=-vmax, vmax=vmax,
            shading="auto", rasterized=True,
        )
        plt.colorbar(im, ax=ax, label="Divergence ∇·V")

        if bifurcation_mask is not None and bifurcation_mask.any():
            ax.scatter(
                x[bifurcation_mask], y[bifurcation_mask],
                s=15, c="gold", marker="*", alpha=0.8,
                label="Bifurcation zone", zorder=5,
            )
            ax.legend(fontsize=9)

        if self.dorsal_up:
            ax.invert_yaxis()

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x (µm)", fontsize=11)
        ax.set_ylabel("y (µm)", fontsize=11)
        ax.set_aspect("equal")
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 4. Relapse probability map
    # ------------------------------------------------------------------

    def plot_relapse_probability(
        self,
        xy_coords: np.ndarray,
        relapse_risk: np.ndarray,
        relapse_nursery_mask: Optional[np.ndarray] = None,
        title: str = "Relapse Probability Map",
        save_name: str = "relapse_probability_map.pdf",
    ) -> Optional["plt.Figure"]:
        """
        Spatial heatmap of relapse risk scores.

        Parameters
        ----------
        xy_coords : (N, 2)
        relapse_risk : (N,)  relapse risk score (0–1 after normalisation)
        relapse_nursery_mask : (N,) bool  high-confidence relapse nurseries
        """
        _check_mpl()

        # Normalise to [0, 1] for interpretability
        r_min, r_max = relapse_risk.min(), relapse_risk.max()
        risk_norm = (relapse_risk - r_min) / (r_max - r_min + 1e-8)

        fig, ax = plt.subplots(figsize=self.figsize)
        sc = ax.scatter(
            xy_coords[:, 0], xy_coords[:, 1],
            c=risk_norm, cmap=_RELAPSE_CMAP,
            s=3, alpha=0.7, linewidths=0, vmin=0, vmax=1, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Relapse Risk Score")

        if relapse_nursery_mask is not None and relapse_nursery_mask.any():
            ax.scatter(
                xy_coords[relapse_nursery_mask, 0],
                xy_coords[relapse_nursery_mask, 1],
                s=20, facecolors="none", edgecolors="black",
                linewidths=0.5, alpha=0.9, label="Relapse Nursery", zorder=5,
            )
            ax.legend(fontsize=9)

        if self.dorsal_up:
            ax.invert_yaxis()

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x (µm)", fontsize=11)
        ax.set_ylabel("y (µm)", fontsize=11)
        ax.set_aspect("equal")
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 5. Multi-timepoint trajectory panel
    # ------------------------------------------------------------------

    def plot_trajectory_panel(
        self,
        umap_coords_list: List[np.ndarray],
        timepoint_names: List[str],
        colour_seq: Optional[List[str]] = None,
        model_type: str = "PLP",
        save_name: str = "trajectory_panel.pdf",
    ) -> Optional["plt.Figure"]:
        """
        Grid of UMAP plots, one per timepoint, showing temporal progression.

        Parameters
        ----------
        umap_coords_list : list of (N_t, 2) arrays, one per timepoint
        timepoint_names : list of T strings
        """
        _check_mpl()
        T = len(timepoint_names)
        ncols = min(T, 5)
        nrows = (T + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 3.5))
        axes = np.array(axes).flatten()

        cmap_list = (colour_seq or
                     (_PLP_TIMEPOINT_CMAP if model_type == "PLP" else _MOG_TIMEPOINT_CMAP))

        for t, (coords, name) in enumerate(zip(umap_coords_list, timepoint_names)):
            ax = axes[t]
            colour = cmap_list[t % len(cmap_list)]
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=colour, s=3, alpha=0.6, linewidths=0, rasterized=True)
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.axis("off")

        for ax in axes[T:]:
            ax.set_visible(False)

        fig.suptitle(f"{model_type} Trajectory Progression", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 6. Niche cluster annotation map
    # ------------------------------------------------------------------

    def plot_niche_clusters(
        self,
        xy_coords: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_names: Optional[Dict[int, str]] = None,
        title: str = "Niche Cluster Map",
        save_name: str = "niche_cluster_map.pdf",
    ) -> Optional["plt.Figure"]:
        """
        Spatial plot coloured by Leiden niche cluster.

        Parameters
        ----------
        cluster_names : dict {cluster_id: archetype_name}
            e.g. {0: 'Fibrotic Gatekeeper', 1: 'Relapse Spark', ...}
        """
        _check_mpl()

        K = int(cluster_labels.max()) + 1
        cmap = plt.cm.get_cmap("tab20", K)

        fig, ax = plt.subplots(figsize=self.figsize)
        for k in range(K):
            mask = cluster_labels == k
            name = cluster_names.get(k, f"Cluster {k}") if cluster_names else f"Cluster {k}"
            ax.scatter(
                xy_coords[mask, 0], xy_coords[mask, 1],
                c=[cmap(k)], s=3, alpha=0.6, linewidths=0,
                label=name, rasterized=True,
            )

        if self.dorsal_up:
            ax.invert_yaxis()

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x (µm)", fontsize=11)
        ax.set_ylabel("y (µm)", fontsize=11)
        ax.legend(markerscale=4, fontsize=7, loc="upper right",
                  framealpha=0.7, ncol=2)
        ax.set_aspect("equal")
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # 7. Mean divergence across timepoints (trajectory plot)
    # ------------------------------------------------------------------

    def plot_divergence_trajectory(
        self,
        mean_div_seq: np.ndarray,
        bif_frac_seq: np.ndarray,
        timepoint_names: List[str],
        model_type: str = "PLP",
        save_name: str = "divergence_trajectory.pdf",
    ) -> Optional["plt.Figure"]:
        """
        Line plot of mean divergence and bifurcation fraction over time.

        Highlights the bifurcation peak timepoint.
        """
        _check_mpl()

        T = len(timepoint_names)
        x_tick = np.arange(T)
        colour = _MODEL_COLOURS.get(model_type.upper(), "grey")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, T * 1.2), 6),
                                        sharex=True)

        ax1.plot(x_tick, mean_div_seq, marker="o", color=colour, linewidth=2)
        ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        peak_idx = int(np.argmax(mean_div_seq))
        ax1.axvline(peak_idx, color="red", linestyle=":", linewidth=1.5,
                    label=f"Peak: {timepoint_names[peak_idx]}")
        ax1.set_ylabel("Mean Divergence ∇·V", fontsize=11)
        ax1.set_title(f"{model_type} Divergence Over Trajectory", fontsize=12,
                      fontweight="bold")
        ax1.legend(fontsize=9)

        ax2.bar(x_tick, bif_frac_seq * 100, color=colour, alpha=0.7)
        ax2.set_ylabel("Bifurcating Niches (%)", fontsize=11)
        ax2.set_xticks(x_tick)
        ax2.set_xticklabels(timepoint_names, rotation=30, ha="right", fontsize=9)

        plt.tight_layout()
        self._save(fig, save_name)
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, fig: "plt.Figure", name: str) -> None:
        path = self.output_dir / name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        logger.info("Figure saved: %s", path)
        plt.close(fig)

    @staticmethod
    def project_velocity_to_2d(
        velocity: np.ndarray,
        umap_coords: np.ndarray,
        z: np.ndarray,
        n_neighbours: int = 20,
    ) -> np.ndarray:
        """
        Project high-dimensional velocity vectors to 2-D UMAP space.

        For each cell, the 2-D velocity is estimated as the weighted sum
        of displacements from the cell to its n_neighbours in UMAP space,
        weighted by similarity of the high-D velocity to the displacement
        direction in latent space.

        Parameters
        ----------
        velocity : (N, D)   high-dimensional velocity
        umap_coords : (N, 2)  UMAP embedding
        z : (N, D)   latent embeddings
        n_neighbours : int

        Returns
        -------
        velocity_2d : (N, 2)
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(z)
        _, nn_idx = tree.query(z, k=n_neighbours + 1)
        nn_idx = nn_idx[:, 1:]   # exclude self

        vel_2d = np.zeros((len(z), 2))

        for i in range(len(z)):
            nbrs = nn_idx[i]
            delta_z = z[nbrs] - z[i]         # (k, D)
            delta_umap = umap_coords[nbrs] - umap_coords[i]  # (k, 2)

            # Project velocity onto delta_z directions → scalar weights
            v = velocity[i]                   # (D,)
            norms = np.linalg.norm(delta_z, axis=1, keepdims=True) + 1e-8
            delta_z_unit = delta_z / norms
            weights = (v @ delta_z_unit.T).clip(min=0)   # (k,) keep positive projection

            if weights.sum() > 1e-8:
                vel_2d[i] = (weights[:, None] * delta_umap).sum(axis=0) / weights.sum()

        return vel_2d
