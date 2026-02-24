"""
SpatialGraphBuilder

Constructs the hybrid spatio-temporal heterogeneous graph used as input to
the SpatioTemporalGNN.

Graph schema
────────────
Nodes:  one per cell (or niche centroid after optional sub-sampling)

Node features:
    • Log-normalised gene expression        (n_genes,)
    • Cell-type embedding                   (n_ct_dims,)
    • Anatomical RLCS features              (6,)       d_CC, d_GM, θ, cos_θ, sin_θ, hemi
    • Timepoint embedding index             (1,)       int  — embedded in GNN
    • Model-type flag                       (1,)       0=MOG, 1=PLP

Edge types
──────────
1. Spatial proximity edges
   Radius = 50 µm (configurable).
   Edge weight = Gaussian kernel of Euclidean distance.
   w_spatial = exp(−d² / (2σ²))

2. Ligand–Receptor (LR) interaction edges
   Drawn between cells where:
     • A known LR pair (ligand in cell_i, receptor in cell_j) is expressed.
     • Expression above detection threshold.
   Edge weight = LR_prior × ligand_expression × receptor_expression

   LR database priority: CellChat > Omnipath > custom TSV.

3. Temporal edges
   Link niche centroids across consecutive timepoints (same region, same model).
   Enable temporal flow propagation in GRU / Neural ODE modules.

Scalability:
    NeighborLoader-compatible edge index format.
    Sparse COO adjacency.
    Option to produce torch_geometric Data or HeteroData objects.

Biological rationale:
    Spatial edges capture local niche micro-environments.
    LR edges encode the signalling capacity between cell types — the molecular
    'wiring' that determines whether an inflamed niche transitions toward
    resolution or relapse priming.
    Temporal edges allow the GNN to track niche state trajectories over time.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from scipy.spatial import cKDTree

try:
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.utils import to_undirected, coalesce
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn(
        "torch_geometric not installed.  SpatialGraphBuilder will return "
        "raw edge_index tensors instead of PyG Data objects.",
        ImportWarning,
        stacklevel=1,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR Database Loader
# ---------------------------------------------------------------------------

class LRDatabase:
    """
    Ligand–Receptor interaction database.

    Supports CellChat, Omnipath, and custom TSV formats.

    Expected TSV columns: ligand  receptor  weight  (tab-separated, header row)

    Parameters
    ----------
    source : 'cellchat' | 'omnipath' | Path
        Data source.  If Path, loads a custom TSV file.
    min_weight : float
        Minimum LR prior weight to retain (filters low-confidence pairs).
    """

    # Curated high-confidence neuroinflammatory LR pairs
    # (used as fallback when no database is provided)
    NEUROIMMUNE_FALLBACK = [
        # Ligand        Receptor    Prior weight
        ("Cxcl10",     "Cxcr3",    0.9),   # Microglia → T-cell chemotaxis
        ("Ccl2",       "Ccr2",     0.9),   # Monocyte recruitment
        ("Ccl5",       "Ccr5",     0.8),
        ("Tgfb1",      "Tgfbr1",   0.85),  # Immunosuppression / fibrosis
        ("Il1b",       "Il1r1",    0.9),   # Microglial IL-1 signalling
        ("Tnf",        "Tnfrsf1a", 0.9),
        ("Ifng",       "Ifngr1",   0.85),  # T-cell → astrocyte activation
        ("Csf1",       "Csf1r",    0.9),   # Colony-stimulating factor microglia
        ("Vegfa",      "Kdr",      0.7),   # Vascular permeability
        ("Spp1",       "Cd44",     0.85),  # Osteopontin, remyelination block
        ("C3",         "Itgam",    0.8),   # Complement – phagocytosis
        ("Cx3cl1",     "Cx3cr1",   0.9),   # Microglia homeostasis
        ("Pdgfb",      "Pdgfrb",   0.75),  # Pericyte stabilisation
        ("Hgf",        "Met",      0.7),   # Oligodendrocyte precursor support
        ("Nrg1",       "Erbb4",    0.75),  # Oligodendrocyte maturation
    ]

    def __init__(
        self,
        source: Union[str, Path] = "fallback",
        min_weight: float = 0.5,
    ) -> None:
        self.min_weight = min_weight
        self._lr_df = self._load(source)
        logger.info(
            "LRDatabase loaded: %d pairs (source=%s)", len(self._lr_df), source
        )

    def _load(self, source) -> pd.DataFrame:
        if source == "fallback" or source is None:
            df = pd.DataFrame(
                self.NEUROIMMUNE_FALLBACK, columns=["ligand", "receptor", "weight"]
            )
        elif source in ("cellchat", "omnipath"):
            # In production: download via API or pre-bundled CSV
            # Here we use the fallback as a structural placeholder
            logger.warning(
                "Live %s download not implemented — using curated fallback LR set.",
                source,
            )
            df = pd.DataFrame(
                self.NEUROIMMUNE_FALLBACK, columns=["ligand", "receptor", "weight"]
            )
        else:
            path = Path(source)
            df = pd.read_csv(path, sep="\t")
            if not {"ligand", "receptor", "weight"}.issubset(df.columns):
                raise ValueError(
                    "Custom LR TSV must have columns: ligand, receptor, weight"
                )

        df = df[df["weight"] >= self.min_weight].reset_index(drop=True)
        # Lowercase all gene names for robust matching
        df["ligand"] = df["ligand"].str.lower()
        df["receptor"] = df["receptor"].str.lower()
        return df

    def get_pairs(self) -> List[Tuple[str, str, float]]:
        return list(zip(
            self._lr_df["ligand"],
            self._lr_df["receptor"],
            self._lr_df["weight"],
        ))

    def ligands(self) -> List[str]:
        return self._lr_df["ligand"].tolist()

    def receptors(self) -> List[str]:
        return self._lr_df["receptor"].tolist()


# ---------------------------------------------------------------------------
# SpatialGraphBuilder
# ---------------------------------------------------------------------------

class SpatialGraphBuilder:
    """
    Build the hybrid interaction graph from processed cell data.

    Parameters
    ----------
    spatial_radius_um : float
        Radius (µm) for spatial proximity edges.  Default 50 µm captures
        ≈3–4 cell diameters, consistent with direct paracrine signalling.
    spatial_sigma_um : float
        Gaussian σ for spatial edge weight kernel.
    lr_database : LRDatabase | None
        LR interaction database.  If None, uses fallback curated set.
    lr_expr_threshold : float
        Minimum normalised expression (log1p scale) for a ligand / receptor
        to be considered active in a cell.
    n_gene_features : int | None
        Number of top HVGs to retain as node features.  None = use all.
    device : str
        Torch device.
    """

    MODEL_FLAG = {"MOG": 0, "PLP": 1}

    def __init__(
        self,
        spatial_radius_um: float = 50.0,
        spatial_sigma_um: float = 25.0,
        lr_database: Optional[LRDatabase] = None,
        lr_expr_threshold: float = 0.1,
        n_gene_features: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self.spatial_radius_um = spatial_radius_um
        self.spatial_sigma_um = spatial_sigma_um
        self.lr_db = lr_database or LRDatabase()
        self.lr_expr_threshold = lr_expr_threshold
        self.n_gene_features = n_gene_features
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(
        self,
        xy_coords: np.ndarray,
        expr_matrix: np.ndarray,
        gene_names: List[str],
        anatomical_features: np.ndarray,
        cell_type_embeddings: np.ndarray,
        timepoint_idx: np.ndarray,
        model_type: str,
        section_id: str,
    ) -> "Data":
        """
        Build a PyG Data object (or dict) for one section/timepoint.

        Parameters
        ----------
        xy_coords : (N, 2)  µm coordinates (x, y)
        expr_matrix : (N, G)  log1p-normalised expression
        gene_names : list of G gene names (lowercase)
        anatomical_features : (N, 6)  RLCS features from AnatomicalAligner
        cell_type_embeddings : (N, D_ct)  learned or one-hot cell-type vectors
        timepoint_idx : (N,)  int  timepoint index
        model_type : 'MOG' or 'PLP'
        section_id : str  informational label

        Returns
        -------
        torch_geometric.data.Data or dict containing:
            x           : (N, F)  node features
            edge_index  : (2, E)  COO edge index
            edge_attr   : (E, 2)  [spatial_weight, lr_weight]
            edge_type   : (E,)    int8  0=spatial, 1=LR
            pos         : (N, 2)  xy coords (for viz)
            timepoint   : (N,)    int
            model_flag  : (N,)    int8
            section_id  : str
        """
        N = xy_coords.shape[0]
        logger.info(
            "Building graph for %s | %s | n_cells=%d", model_type, section_id, N
        )

        # 1. Node feature matrix
        x = self._build_node_features(
            expr_matrix, gene_names,
            anatomical_features, cell_type_embeddings,
            timepoint_idx, model_type, N,
        )

        # 2. Spatial edges
        sp_src, sp_dst, sp_w = self._spatial_edges(xy_coords)

        # 3. LR interaction edges
        lr_src, lr_dst, lr_w = self._lr_edges(xy_coords, expr_matrix, gene_names)

        # 4. Merge edge sets
        edge_index, edge_attr, edge_type = self._merge_edges(
            sp_src, sp_dst, sp_w,
            lr_src, lr_dst, lr_w,
            N,
        )

        model_flag_val = self.MODEL_FLAG.get(model_type.upper(), -1)
        model_flag = torch.full((N,), model_flag_val, dtype=torch.int8,
                                device=self.device)

        if HAS_PYG:
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=torch.tensor(xy_coords, dtype=torch.float32,
                                 device=self.device),
                timepoint=torch.tensor(timepoint_idx, dtype=torch.long,
                                       device=self.device),
                model_flag=model_flag,
                section_id=section_id,
            )
            data.edge_type = edge_type
            data.num_nodes = N
        else:
            data = {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_type": edge_type,
                "pos": torch.tensor(xy_coords, dtype=torch.float32,
                                    device=self.device),
                "timepoint": torch.tensor(timepoint_idx, dtype=torch.long,
                                          device=self.device),
                "model_flag": model_flag,
                "section_id": section_id,
                "num_nodes": N,
            }

        logger.info(
            "Graph built: %d nodes, %d edges (%d spatial + %d LR)",
            N, edge_index.shape[1],
            int((edge_type == 0).sum()), int((edge_type == 1).sum()),
        )
        return data

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------

    def _build_node_features(
        self,
        expr_matrix: np.ndarray,
        gene_names: List[str],
        anatomical_features: np.ndarray,
        cell_type_embeddings: np.ndarray,
        timepoint_idx: np.ndarray,
        model_type: str,
        N: int,
    ) -> Tensor:
        """
        Concatenate all node feature blocks into a single (N, F) tensor.

        Feature blocks (in order):
          1. Gene expression      (N, G)
          2. Anatomical RLCS      (N, 6)
          3. Cell-type embedding  (N, D_ct)
          4. Timepoint (float)    (N, 1)   normalised 0–1
          5. Model flag (float)   (N, 1)
        """
        # Optionally reduce to top HVGs
        if self.n_gene_features is not None and self.n_gene_features < expr_matrix.shape[1]:
            var = expr_matrix.var(axis=0)
            top_idx = np.argsort(var)[-self.n_gene_features:]
            expr_matrix = expr_matrix[:, top_idx]

        model_flag_val = float(self.MODEL_FLAG.get(model_type.upper(), 0))
        model_col = np.full((N, 1), model_flag_val, dtype=np.float32)

        max_tp = max(
            len(v) for v in [
                [0, 1, 2, 3, 4],        # MOG — 5 timepoints
                [0, 1, 2, 3, 4, 5, 6, 7, 8],  # PLP — 9 timepoints
            ]
        )
        tp_norm = timepoint_idx.astype(np.float32) / max(max_tp - 1, 1)
        tp_col = tp_norm.reshape(-1, 1)

        parts = [
            expr_matrix.astype(np.float32),
            anatomical_features.astype(np.float32),
            cell_type_embeddings.astype(np.float32),
            tp_col,
            model_col,
        ]
        x_np = np.concatenate(parts, axis=1)
        return torch.tensor(x_np, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Spatial edges
    # ------------------------------------------------------------------

    def _spatial_edges(
        self, xy_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build radius-neighbourhood spatial edges with Gaussian kernel weights.

        Returns (src, dst, weights) as 1-D np arrays.
        """
        tree = cKDTree(xy_coords)
        pairs = tree.query_pairs(r=self.spatial_radius_um, output_type="ndarray")
        if len(pairs) == 0:
            return (np.empty(0, int), np.empty(0, int), np.empty(0, np.float32))

        src, dst = pairs[:, 0], pairs[:, 1]
        dists = np.linalg.norm(xy_coords[src] - xy_coords[dst], axis=1)
        weights = np.exp(-(dists ** 2) / (2 * self.spatial_sigma_um ** 2)).astype(
            np.float32
        )
        return src, dst, weights

    # ------------------------------------------------------------------
    # LR interaction edges
    # ------------------------------------------------------------------

    def _lr_edges(
        self,
        xy_coords: np.ndarray,
        expr_matrix: np.ndarray,
        gene_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build directed LR interaction edges.

        For each LR pair (L, R) with prior weight w:
          • Find cells expressing L above threshold (ligand cells)
          • Find cells expressing R above threshold (receptor cells)
          • Within spatial radius, link ligand_cell → receptor_cell
          • Edge weight = w × lig_expr × rec_expr

        Returns (src, dst, weights) as directed edge lists.
        """
        lr_pairs = self.lr_db.get_pairs()
        gene_lower = [g.lower() for g in gene_names]
        gene_idx = {g: i for i, g in enumerate(gene_lower)}

        tree = cKDTree(xy_coords)
        srcs, dsts, wts = [], [], []

        for ligand, receptor, prior in lr_pairs:
            l_idx = gene_idx.get(ligand)
            r_idx = gene_idx.get(receptor)
            if l_idx is None or r_idx is None:
                continue

            lig_expr = expr_matrix[:, l_idx]
            rec_expr = expr_matrix[:, r_idx]

            lig_cells = np.where(lig_expr > self.lr_expr_threshold)[0]
            rec_cells = np.where(rec_expr > self.lr_expr_threshold)[0]

            if len(lig_cells) == 0 or len(rec_cells) == 0:
                continue

            # Spatial constraint: LR edges only within radius
            lig_tree = cKDTree(xy_coords[lig_cells])
            rec_tree = cKDTree(xy_coords[rec_cells])
            pairs_set = lig_tree.query_ball_tree(rec_tree, r=self.spatial_radius_um)

            for i_local, j_locals in enumerate(pairs_set):
                if not j_locals:
                    continue
                i_global = lig_cells[i_local]
                j_globals = rec_cells[np.array(j_locals)]
                w = (prior * lig_expr[i_global] *
                     rec_expr[j_globals]).astype(np.float32)
                srcs.append(np.full(len(j_globals), i_global))
                dsts.append(j_globals)
                wts.append(w)

        if not srcs:
            return (np.empty(0, int), np.empty(0, int), np.empty(0, np.float32))

        src = np.concatenate(srcs)
        dst = np.concatenate(dsts)
        wt = np.concatenate(wts)
        return src, dst, wt

    # ------------------------------------------------------------------
    # Edge merging
    # ------------------------------------------------------------------

    def _merge_edges(
        self,
        sp_src, sp_dst, sp_w,
        lr_src, lr_dst, lr_w,
        N: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Combine spatial (undirected) and LR (directed) edge sets.

        Returns:
            edge_index : (2, E_total)
            edge_attr  : (E_total, 2)  [spatial_weight, lr_weight]
            edge_type  : (E_total,)    0=spatial, 1=LR
        """
        # Spatial: make undirected by adding reverse edges
        if len(sp_src) > 0:
            sp_src_full = np.concatenate([sp_src, sp_dst])
            sp_dst_full = np.concatenate([sp_dst, sp_src])
            sp_w_full   = np.concatenate([sp_w, sp_w])
            sp_attr = np.stack(
                [sp_w_full, np.zeros_like(sp_w_full)], axis=1
            )
            sp_type = np.zeros(len(sp_src_full), dtype=np.int8)
        else:
            sp_src_full = sp_dst_full = np.empty(0, int)
            sp_attr = np.empty((0, 2), np.float32)
            sp_type = np.empty(0, np.int8)

        # LR: directed, keep as-is
        if len(lr_src) > 0:
            lr_attr = np.stack(
                [np.zeros_like(lr_w), lr_w], axis=1
            )
            lr_type = np.ones(len(lr_src), dtype=np.int8)
        else:
            lr_src = lr_dst = np.empty(0, int)
            lr_attr = np.empty((0, 2), np.float32)
            lr_type = np.empty(0, np.int8)

        src_all = np.concatenate([sp_src_full, lr_src])
        dst_all = np.concatenate([sp_dst_full, lr_dst])
        attr_all = np.concatenate([sp_attr, lr_attr], axis=0)
        type_all = np.concatenate([sp_type, lr_type])

        edge_index = torch.tensor(
            np.stack([src_all, dst_all], axis=0), dtype=torch.long,
            device=self.device
        )
        edge_attr = torch.tensor(attr_all, dtype=torch.float32,
                                 device=self.device)
        edge_type = torch.tensor(type_all, dtype=torch.int8,
                                 device=self.device)

        return edge_index, edge_attr, edge_type

    # ------------------------------------------------------------------
    # Temporal edge construction
    # ------------------------------------------------------------------

    def build_temporal_edges(
        self,
        graphs: List,
        max_distance_um: float = 100.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Link niche centroids across consecutive timepoints.

        For a sequence of section graphs [t0, t1, t2, …], create temporal
        edges from each cell in t_k to its nearest neighbours in t_{k+1}
        within max_distance_um.

        These edges are used by the GRU / Neural ODE temporal bridge to
        propagate inflammatory state across timepoints.

        Parameters
        ----------
        graphs : list of Data objects (ordered by timepoint)
        max_distance_um : float

        Returns
        -------
        temporal_edge_index : (2, E_t)   COO, with offset node indices
        temporal_edge_attr  : (E_t, 1)   spatial distance weights
        """
        if len(graphs) < 2:
            raise ValueError("Need at least 2 graphs for temporal edges.")

        temporal_srcs, temporal_dsts, temporal_ws = [], [], []
        cumulative_offset = 0
        node_counts = []

        # Collect positions and offsets
        positions = []
        offsets = []
        for g in graphs:
            pos = g.pos.cpu().numpy() if hasattr(g, "pos") else g["pos"].cpu().numpy()
            positions.append(pos)
            offsets.append(cumulative_offset)
            n = pos.shape[0]
            node_counts.append(n)
            cumulative_offset += n

        for t in range(len(graphs) - 1):
            pos_t = positions[t]
            pos_t1 = positions[t + 1]
            off_t = offsets[t]
            off_t1 = offsets[t + 1]

            tree_t1 = cKDTree(pos_t1)
            near_idx = tree_t1.query_ball_point(pos_t, r=max_distance_um)

            for i, neighbours in enumerate(near_idx):
                if not neighbours:
                    continue
                j_arr = np.array(neighbours)
                dists = np.linalg.norm(pos_t[i] - pos_t1[j_arr], axis=1)
                ws = np.exp(-dists / max_distance_um).astype(np.float32)
                temporal_srcs.append(np.full(len(j_arr), off_t + i))
                temporal_dsts.append(off_t1 + j_arr)
                temporal_ws.append(ws)

        if not temporal_srcs:
            return (
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0, 1), dtype=torch.float32, device=self.device),
            )

        src = np.concatenate(temporal_srcs)
        dst = np.concatenate(temporal_dsts)
        ws  = np.concatenate(temporal_ws)

        temporal_edge_index = torch.tensor(
            np.stack([src, dst]), dtype=torch.long, device=self.device
        )
        temporal_edge_attr = torch.tensor(
            ws.reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        return temporal_edge_index, temporal_edge_attr
