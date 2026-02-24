"""
AnatomicalAligner and RadialBasisTransform

Implements tissue-aware coordinate normalization for spinal cord Xenium data.

Biological rationale:
  Raw (x, y) Xenium coordinates are section-specific and carry no anatomical
  meaning across animals, slides, or timepoints.  All downstream comparisons
  require a shared *relative lesion coordinate system* anchored to invariant
  spinal cord landmarks:

    Central Canal (CC)  – ventricular zone; inflammatory spread origin
    Grey Matter border  – boundary between myelinated WM and cellular GM
    Dorsal midline      – θ = 0 reference for dorsal–ventral polarity

  Distances are computed as *geodesic* (tissue-masked) paths, NOT Euclidean,
  to prevent shortest paths from crossing empty lumen or tissue gaps.

Coordinate output:
    (x, y)  →  (d_CC, d_GM, θ, mirror_flag)

    d_CC   : geodesic distance to central canal
    d_GM   : geodesic distance to nearest GM border pixel
    θ      : polar angle relative to dorsal midline  [−π, π]
    mirror : +1 left hemisphere, −1 right hemisphere (bilateral symmetry)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_dilation, label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage.graph import MCP_Geometric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Ensure binary 2D uint8 mask."""
    if mask.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def _dorsal_midline_angle(coords_yx: np.ndarray,
                          centroid_yx: np.ndarray) -> np.ndarray:
    """
    Return polar angle θ ∈ [−π, π] for each cell relative to the dorsal
    midline.

    Convention:  θ = 0 → dorsal (superior in coronal view)
                 θ = ±π → ventral
                 Positive → right hemisphere

    Parameters
    ----------
    coords_yx : (N, 2)  cell coordinates in image (row, col) space
    centroid_yx : (2,)   centroid of central canal mask
    """
    delta = coords_yx - centroid_yx          # (N, 2)  row_delta, col_delta
    # In image coords: row increases downward (ventral in standard orientation)
    # We negate row delta so that dorsal → top → positive y-axis → θ near 0
    dy = -delta[:, 0]  # dorsal = up
    dx = delta[:, 1]   # right positive
    theta = np.arctan2(dx, dy)               # atan2(east, north) → CW from North
    return theta


# ---------------------------------------------------------------------------
# RadialBasisTransform
# ---------------------------------------------------------------------------

class RadialBasisTransform:
    """
    Transform raw (x, y) cell coordinates into anatomically grounded
    features: (d_CC, d_GM, θ, cos_θ, sin_θ, hemisphere).

    Distances are geodesic — shortest paths constrained to tissue pixels.

    Parameters
    ----------
    tissue_mask : np.ndarray (H, W), bool
        1 = tissue, 0 = background / lumen.  Distances will never cross 0.
    cc_mask : np.ndarray (H, W), bool
        Central canal pixels.  d_CC is measured to nearest cc_mask pixel.
    gm_mask : np.ndarray (H, W), bool
        Grey matter pixels.  d_GM is distance to GM border (WM-facing edge).
    resolution_um : float
        Physical pixel size in micrometres (default 1 µm for Xenium).
    dorsal_vector_yx : tuple[float, float] | None
        Unit vector pointing toward dorsal midline in image (row, col) space.
        If None, inferred from GM horn morphology (assumes standard
        coronal orientation: dorsal horns toward image top).
    """

    def __init__(
        self,
        tissue_mask: np.ndarray,
        cc_mask: np.ndarray,
        gm_mask: np.ndarray,
        resolution_um: float = 1.0,
        dorsal_vector_yx: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.resolution_um = resolution_um
        self.tissue_mask = _validate_mask(tissue_mask, "tissue_mask")
        self.cc_mask = _validate_mask(cc_mask, "cc_mask")
        self.gm_mask = _validate_mask(gm_mask, "gm_mask")

        # Precompute geodesic distance maps (in µm)
        logger.info("Computing geodesic distance to central canal …")
        self.d_cc_map = self._geodesic_distance(
            source_mask=self.cc_mask,
            traversable_mask=self.tissue_mask,
        ) * resolution_um

        logger.info("Computing geodesic distance to GM border …")
        gm_border = self._extract_gm_border(self.gm_mask, self.tissue_mask)
        self.d_gm_map = self._geodesic_distance(
            source_mask=gm_border,
            traversable_mask=self.tissue_mask,
        ) * resolution_um

        # Central canal centroid for angular reference
        cc_rows, cc_cols = np.where(self.cc_mask > 0)
        self.cc_centroid_yx = np.array(
            [cc_rows.mean(), cc_cols.mean()], dtype=np.float32
        )

        # Dorsal midline direction
        self.dorsal_vector_yx = dorsal_vector_yx or self._infer_dorsal_vector()
        logger.info("Dorsal unit vector (row, col): %s", self.dorsal_vector_yx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, xy_coords: np.ndarray) -> np.ndarray:
        """
        Map raw (x, y) cell coordinates to anatomical feature vectors.

        Parameters
        ----------
        xy_coords : (N, 2)  float  — Xenium pixel or µm coordinates (col, row)

        Returns
        -------
        features : (N, 6)  float32
            Columns: d_CC, d_GM, θ, cos_θ, sin_θ, hemisphere (+1/−1)
        """
        N = xy_coords.shape[0]
        # Convert (x=col, y=row) → (row, col)
        yx_coords = xy_coords[:, ::-1].astype(np.float32)

        H, W = self.tissue_mask.shape
        row_idx = np.clip(yx_coords[:, 0].astype(int), 0, H - 1)
        col_idx = np.clip(yx_coords[:, 1].astype(int), 0, W - 1)

        d_cc = self.d_cc_map[row_idx, col_idx].astype(np.float32)
        d_gm = self.d_gm_map[row_idx, col_idx].astype(np.float32)

        theta = _dorsal_midline_angle(yx_coords, self.cc_centroid_yx)

        # Hemisphere: positive col delta → right → +1, left → −1
        hemi = np.sign(yx_coords[:, 1] - self.cc_centroid_yx[1]).astype(np.float32)
        hemi[hemi == 0] = 1.0  # midline cells → right by convention

        features = np.stack(
            [d_cc, d_gm, theta.astype(np.float32),
             np.cos(theta).astype(np.float32),
             np.sin(theta).astype(np.float32),
             hemi],
            axis=1,
        )
        return features  # (N, 6)

    def mirror_coordinates(self, xy_coords: np.ndarray) -> np.ndarray:
        """
        Mirror coordinates across the spinal cord midline.

        Bilateral symmetry allows pooling left/right lesion data to increase
        niche coverage and detect lateralization biases by comparing
        original vs mirrored trajectory divergences.

        Parameters
        ----------
        xy_coords : (N, 2)

        Returns
        -------
        mirrored : (N, 2)  — x-axis reflected about canal centroid
        """
        midline_x = self.cc_centroid_yx[1]  # col == x
        mirrored = xy_coords.copy().astype(np.float32)
        mirrored[:, 0] = 2.0 * midline_x - mirrored[:, 0]
        return mirrored

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geodesic_distance(
        source_mask: np.ndarray,
        traversable_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute geodesic (tissue-constrained) distance from every pixel to
        the nearest source pixel.

        Uses skimage.graph.MCP_Geometric which implements Dijkstra on a
        8-connected pixel graph with diagonal edge costs = √2.

        Pixels outside traversable_mask receive np.inf.

        Parameters
        ----------
        source_mask : (H, W) bool
        traversable_mask : (H, W) bool — 1=passable tissue, 0=blocked

        Returns
        -------
        dist_map : (H, W) float32  — distances in pixels
        """
        H, W = traversable_mask.shape

        # Cost map: 1.0 on tissue, 1e9 on background (soft block)
        cost = np.where(traversable_mask > 0, 1.0, 1e9).astype(np.float64)

        source_coords = list(zip(*np.where(source_mask > 0)))
        if not source_coords:
            warnings.warn("Source mask is empty — returning zeros.", stacklevel=3)
            return np.zeros((H, W), dtype=np.float32)

        mcp = MCP_Geometric(cost)
        dist_map, _ = mcp.find_costs(source_coords)
        dist_map = np.where(traversable_mask > 0, dist_map, np.inf)
        return dist_map.astype(np.float32)

    @staticmethod
    def _extract_gm_border(gm_mask: np.ndarray,
                           tissue_mask: np.ndarray) -> np.ndarray:
        """
        Extract the white-matter–facing border of the grey matter.

        This is the interface through which immune cells transmigrate from
        WM lesions into GM territory — a biologically relevant boundary.
        """
        dilated = binary_dilation(gm_mask.astype(bool), iterations=1)
        border = dilated.astype(np.uint8) - gm_mask.astype(np.uint8)
        border = np.clip(border, 0, 1)
        # Restrict border to tissue pixels
        border *= tissue_mask
        if border.sum() == 0:
            # Fallback: use full GM mask
            border = gm_mask.copy()
        return border

    def _infer_dorsal_vector(self) -> Tuple[float, float]:
        """
        Infer dorsal midline direction from GM horn morphology.

        Heuristic: in standard coronal spinal cord sections the dorsal horns
        are the topmost GM protrusions.  The centroid of the top half of the
        GM mask defines the dorsal direction relative to the central canal.
        """
        H, W = self.gm_mask.shape
        top_half = self.gm_mask[: H // 2, :]
        gm_rows, gm_cols = np.where(top_half > 0)

        if len(gm_rows) == 0:
            logger.warning(
                "Could not infer dorsal vector from GM morphology — "
                "defaulting to image-top (row decreasing)."
            )
            return (-1.0, 0.0)

        dorsal_centroid_yx = np.array([gm_rows.mean(), gm_cols.mean()])
        vec = self.cc_centroid_yx - dorsal_centroid_yx   # CC relative to dorsal
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return (-1.0, 0.0)
        return tuple((-vec / norm).tolist())  # point TOWARD dorsal


# ---------------------------------------------------------------------------
# AnatomicalAligner
# ---------------------------------------------------------------------------

class AnatomicalAligner:
    """
    High-level orchestrator that aligns multi-section, multi-timepoint
    Xenium data into a shared Relative Lesion Coordinate System (RLCS).

    For each section the pipeline:
      1. Validates and loads tissue / CC / GM masks.
      2. Constructs a RadialBasisTransform.
      3. Maps all cell coordinates to (d_CC, d_GM, θ) anatomical features.
      4. Optionally mirrors coordinates for bilateral lesion pooling.
      5. Returns an AnnData-compatible feature matrix ready for graph construction.

    Parameters
    ----------
    resolution_um : float
        Xenium pixel resolution in micrometres (default 0.2138 µm/px for
        Xenium 5k panels; often 1.0 after binning).
    mirror_bilateral : bool
        If True, mirror each section across midline and augment the dataset.
        Mirrored cells receive hemisphere = −1, original = +1.
    device : str
        Torch device ('cpu' or 'cuda').
    """

    # Timepoint ordering for MOG and PLP models
    TIMEPOINTS_MOG = [
        "baseline",
        "non_symptomatic",
        "early_onset",
        "chronic_peak",
        "chronic_long",
    ]
    TIMEPOINTS_PLP = [
        "baseline",
        "onset_I",
        "onset_II",
        "monophasic",
        "peak_I",
        "remission_I",
        "peak_II",
        "remission_II",
        "peak_III",
    ]
    ALL_TIMEPOINTS = {
        "MOG": TIMEPOINTS_MOG,
        "PLP": TIMEPOINTS_PLP,
    }

    def __init__(
        self,
        resolution_um: float = 1.0,
        mirror_bilateral: bool = True,
        device: str = "cpu",
    ) -> None:
        self.resolution_um = resolution_um
        self.mirror_bilateral = mirror_bilateral
        self.device = torch.device(device)
        self._transforms: dict[str, RadialBasisTransform] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_section(
        self,
        section_id: str,
        tissue_mask: np.ndarray,
        cc_mask: np.ndarray,
        gm_mask: np.ndarray,
        dorsal_vector_yx: Optional[Tuple[float, float]] = None,
    ) -> "AnatomicalAligner":
        """
        Fit a RadialBasisTransform for one section.

        Parameters
        ----------
        section_id : str
            Unique identifier (e.g. 'MOG_chronic_peak_s1').
        tissue_mask, cc_mask, gm_mask : np.ndarray (H, W)
        dorsal_vector_yx : optional manual override for dorsal direction.

        Returns
        -------
        self  (chainable)
        """
        logger.info("Fitting AlignmentTransform for section: %s", section_id)
        rbt = RadialBasisTransform(
            tissue_mask=tissue_mask,
            cc_mask=cc_mask,
            gm_mask=gm_mask,
            resolution_um=self.resolution_um,
            dorsal_vector_yx=dorsal_vector_yx,
        )
        self._transforms[section_id] = rbt
        return self

    def transform_section(
        self,
        section_id: str,
        xy_coords: np.ndarray,
        include_mirror: Optional[bool] = None,
    ) -> dict:
        """
        Apply RLCS transformation to cell coordinates.

        Parameters
        ----------
        section_id : str
        xy_coords : (N, 2)  raw Xenium pixel coords (x=col, y=row)
        include_mirror : override instance-level mirror_bilateral setting

        Returns
        -------
        dict with keys:
            'features'        : (N, 6) or (2N, 6) anatomical features
            'xy_original'     : (N, 2) original coordinates
            'mirror_flag'     : (N,) or (2N,) int8 — +1 original, −1 mirror
            'n_original'      : int — number of original cells
        """
        if section_id not in self._transforms:
            raise KeyError(
                f"Section '{section_id}' not fitted.  Call fit_section() first."
            )
        rbt = self._transforms[section_id]
        do_mirror = self.mirror_bilateral if include_mirror is None else include_mirror

        feats = rbt.transform(xy_coords)                        # (N, 6)
        mirror_flag = np.ones(len(xy_coords), dtype=np.int8)

        if do_mirror:
            mirrored_xy = rbt.mirror_coordinates(xy_coords)
            feats_mirror = rbt.transform(mirrored_xy)
            # Flip hemisphere sign for mirrored cells
            feats_mirror[:, 5] *= -1
            feats = np.concatenate([feats, feats_mirror], axis=0)
            flag_mirror = -np.ones(len(xy_coords), dtype=np.int8)
            mirror_flag = np.concatenate([mirror_flag, flag_mirror])

        return {
            "features": feats.astype(np.float32),
            "xy_original": xy_coords.astype(np.float32),
            "mirror_flag": mirror_flag,
            "n_original": len(xy_coords),
        }

    def encode_timepoint(self, model_type: str, timepoint: str) -> int:
        """
        Return integer index for a timepoint within its model trajectory.

        Used as ordinal temporal encoding in GNN timepoint embeddings.

        Parameters
        ----------
        model_type : 'MOG' or 'PLP'
        timepoint  : string timepoint name

        Returns
        -------
        int  — 0-based index
        """
        tp_list = self.ALL_TIMEPOINTS.get(model_type.upper())
        if tp_list is None:
            raise ValueError(f"Unknown model type: {model_type}")
        if timepoint not in tp_list:
            raise ValueError(
                f"Timepoint '{timepoint}' not in {model_type} trajectory: {tp_list}"
            )
        return tp_list.index(timepoint)

    def timepoint_tensor(
        self,
        model_type: str,
        timepoint: str,
        n_cells: int,
    ) -> torch.Tensor:
        """
        Return (n_cells,) int64 tensor of timepoint indices.

        Used as input to the GNN timepoint embedding layer.
        """
        idx = self.encode_timepoint(model_type, timepoint)
        return torch.full((n_cells,), idx, dtype=torch.long, device=self.device)

    @property
    def n_sections(self) -> int:
        return len(self._transforms)

    def section_ids(self) -> list[str]:
        return list(self._transforms.keys())
