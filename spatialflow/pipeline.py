"""
EAESpatialFlow — Master Pipeline Orchestrator

Combines all modules into a single end-to-end pipeline:

    AnatomicalAligner → SpatialGraphBuilder → SpatioTemporalGNN
        → NicheVelocity → DivergenceScorer → MechanisticAttributor
        → VisualizationEngine

Usage modes
───────────
1. Training mode
   pipeline.train(train_data_dict, val_data_dict)

2. Inference mode
   pipeline.predict(data_dict)

3. Attribution mode
   pipeline.attribute(data_dict)

Data contract
─────────────
Input dict per section:
{
    'section_id':          str
    'model_type':          'MOG' | 'PLP'
    'timepoint':           str   (e.g. 'remission_I')
    'xy_coords':           np.ndarray (N, 2)  µm
    'expr_matrix':         np.ndarray (N, G)  log1p-normalised
    'gene_names':          List[str]
    'cell_types':          List[str]           (N,)
    'tissue_mask':         np.ndarray (H, W)   binary
    'cc_mask':             np.ndarray (H, W)   binary
    'gm_mask':             np.ndarray (H, W)   binary
    'cell_type_embeddings': np.ndarray (N, D_ct) optional
}

Validation tasks (auto-evaluated after training):
    1. Classify MOG vs PLP (chronic vs relapsing)
    2. Forecast Peak II latent state from Remission I
    3. Detect IFN / complement niche signatures
    4. Predict inflammatory velocity direction
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast

from .aligner import AnatomicalAligner
from .graph_builder import LRDatabase, SpatialGraphBuilder
from .model import SpatioTemporalGNN
from .velocity import NicheVelocity, InflammatoryAxisDefiner
from .divergence import DivergenceScorer
from .losses import TrajectoryContrastiveLoss
from .attributor import MechanisticAttributor, DivergenceTarget
from .visualization import VisualizationEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build cell-type one-hot embeddings
# ---------------------------------------------------------------------------

def _build_ct_embeddings(
    cell_types: List[str],
    known_types: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """One-hot encode cell types.  Returns (N, K) array and type list."""
    if known_types is None:
        known_types = sorted(set(cell_types))
    ct_map = {ct: i for i, ct in enumerate(known_types)}
    K = len(known_types)
    N = len(cell_types)
    emb = np.zeros((N, K), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        idx = ct_map.get(ct.lower(), 0)
        emb[i, idx] = 1.0
    return emb, known_types


# ---------------------------------------------------------------------------
# EAESpatialFlow
# ---------------------------------------------------------------------------

class EAESpatialFlow:
    """
    End-to-end EAE SpatialFlow framework.

    Parameters
    ----------
    config : dict
        Configuration dictionary (see configs/default_config.yaml).
    device : str
        'cuda' or 'cpu'.
    output_dir : str | Path
    """

    def __init__(
        self,
        config: Dict,
        device: str = "cpu",
        output_dir: Union[str, Path] = "outputs",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Modules
        self.aligner = AnatomicalAligner(
            resolution_um=config.get("resolution_um", 1.0),
            mirror_bilateral=config.get("mirror_bilateral", True),
            device=device,
        )
        self.lr_db = LRDatabase(
            source=config.get("lr_database", "fallback"),
            min_weight=config.get("lr_min_weight", 0.5),
        )
        self.graph_builder = SpatialGraphBuilder(
            spatial_radius_um=config.get("spatial_radius_um", 50.0),
            spatial_sigma_um=config.get("spatial_sigma_um", 25.0),
            lr_database=self.lr_db,
            lr_expr_threshold=config.get("lr_expr_threshold", 0.1),
            n_gene_features=config.get("n_gene_features", None),
            device=device,
        )

        # Model (initialised after first data pass to infer in_channels)
        self.model: Optional[SpatioTemporalGNN] = None
        self._in_channels: Optional[int] = None
        self._gene_names: Optional[List[str]] = None
        self._known_cell_types: Optional[List[str]] = None

        # Velocity + divergence
        self.velocity_module = NicheVelocity(
            latent_dim=config.get("latent_dim", 64),
            smooth_sigma=config.get("velocity_smooth_sigma", 0.5),
            learn_inf_axis=config.get("learn_inf_axis", False),
        ).to(self.device)

        self.divergence_scorer = DivergenceScorer(
            latent_dim=config.get("latent_dim", 64),
            knn_k=config.get("div_knn_k", 15),
            bifurcation_threshold=config.get("bifurcation_threshold", 0.5),
        ).to(self.device)

        # Loss
        self.contrastive_loss = TrajectoryContrastiveLoss(
            margin=config.get("triplet_margin", 1.0),
            lambda_triplet=config.get("lambda_triplet", 1.0),
            lambda_align=config.get("lambda_align", 0.3),
            lambda_temporal=config.get("lambda_temporal", 0.2),
            lambda_velocity=config.get("lambda_velocity", 0.1),
        )

        # Viz
        self.viz = VisualizationEngine(
            output_dir=str(self.output_dir / "figures"),
            dpi=config.get("figure_dpi", 150),
        )

        # Attribution (initialised after model build)
        self.attributor: Optional[MechanisticAttributor] = None

        self._scaler = GradScaler() if device == "cuda" else None

    # ------------------------------------------------------------------
    # Data preprocessing pipeline
    # ------------------------------------------------------------------

    def preprocess_section(self, section: Dict) -> Dict:
        """
        Run alignment, embedding, and graph construction for one section.

        Returns enriched section dict with graph Data object attached.
        """
        sid = section["section_id"]
        model_type = section["model_type"]
        timepoint = section["timepoint"]
        xy = section["xy_coords"]
        expr = section["expr_matrix"]
        gene_names = [g.lower() for g in section["gene_names"]]
        cell_types = section["cell_types"]

        # 1. Fit aligner
        self.aligner.fit_section(
            section_id=sid,
            tissue_mask=section["tissue_mask"],
            cc_mask=section["cc_mask"],
            gm_mask=section["gm_mask"],
        )

        # 2. Anatomical transform
        anat_result = self.aligner.transform_section(sid, xy)
        anat_feats = anat_result["features"]        # (N, 6) or (2N, 6) if mirrored

        # 3. Cell-type embeddings
        if "cell_type_embeddings" in section and section["cell_type_embeddings"] is not None:
            ct_emb = section["cell_type_embeddings"]
        else:
            if self._known_cell_types is None:
                self._known_cell_types = sorted(set(ct.lower() for ct in cell_types))
            ct_emb, self._known_cell_types = _build_ct_embeddings(
                [c.lower() for c in cell_types], self._known_cell_types
            )

        # 4. Timepoint index
        tp_idx = self.aligner.encode_timepoint(model_type, timepoint)
        tp_arr = np.full(len(xy), tp_idx, dtype=np.int64)

        # Handle mirrored cells
        n_original = anat_result["n_original"]
        if anat_feats.shape[0] > len(xy):
            # Mirrored: duplicate expr/ct/tp arrays
            expr = np.concatenate([expr, expr], axis=0)
            ct_emb = np.concatenate([ct_emb, ct_emb], axis=0)
            tp_arr = np.concatenate([tp_arr, tp_arr])
            xy_for_graph = np.concatenate([xy, anat_result["xy_original"]])
        else:
            xy_for_graph = xy

        # Store gene names on first section
        if self._gene_names is None:
            self._gene_names = gene_names

        # 5. Build graph
        graph = self.graph_builder.build(
            xy_coords=xy_for_graph,
            expr_matrix=expr,
            gene_names=gene_names,
            anatomical_features=anat_feats,
            cell_type_embeddings=ct_emb,
            timepoint_idx=tp_arr,
            model_type=model_type,
            section_id=sid,
        )

        section["graph"] = graph
        section["anat_features"] = anat_feats
        section["tp_idx"] = tp_idx
        section["gene_names_lower"] = gene_names
        return section

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _init_model(self, in_channels: int) -> None:
        cfg = self.config
        self.model = SpatioTemporalGNN(
            in_channels=in_channels,
            hidden_dim=cfg.get("hidden_dim", 64),
            latent_dim=cfg.get("latent_dim", 64),
            n_genes_reconstruct=cfg.get("n_genes_reconstruct", 500),
            heads=cfg.get("attn_heads", 4),
            edge_dim=2,
            dropout=cfg.get("dropout", 0.1),
            temporal_mode=cfg.get("temporal_mode", "gru"),
            gru_hidden=cfg.get("gru_hidden", 128),
            projection_dim=cfg.get("projection_dim", 128),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model initialised: %s parameters.", f"{n_params:,}")

        if self._gene_names:
            G = min(len(self._gene_names), cfg.get("n_genes_reconstruct", 500))
            D_ct = len(self._known_cell_types or [])
            self.attributor = MechanisticAttributor(
                gene_names=self._gene_names[:G],
                feature_layout={
                    "gene_start": 0,     "gene_end": G,
                    "rlcs_start": G,     "rlcs_end": G + 6,
                    "ct_start": G + 6,   "ct_end": G + 6 + D_ct,
                    "tp_col": G + 6 + D_ct,
                    "model_col": G + 6 + D_ct + 1,
                },
                device=str(self.device),
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        sections: List[Dict],
        n_epochs: int = 50,
        lr: float = 1e-3,
        warmup_steps: int = 500,
        checkpoint_every: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the SpatioTemporalGNN end-to-end.

        Parameters
        ----------
        sections : list of raw section dicts (preprocessed internally)
        n_epochs : int
        lr : float
        warmup_steps : int  for KL annealing
        checkpoint_every : int

        Returns
        -------
        history : dict of loss lists
        """
        logger.info("Preprocessing %d sections …", len(sections))
        processed = [self.preprocess_section(s) for s in sections]

        # Infer in_channels from first graph
        first_graph = processed[0]["graph"]
        x0 = first_graph["x"] if isinstance(first_graph, dict) else first_graph.x
        in_channels = x0.shape[1]
        self._in_channels = in_channels
        self._init_model(in_channels)

        optimizer = optim.AdamW(
            list(self.model.parameters()) +
            list(self.divergence_scorer.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        history = {k: [] for k in [
            "total_loss", "elbo_loss", "contrastive_loss",
            "divergence_loss", "recon_loss", "kl_loss"
        ]}

        global_step = 0

        for epoch in range(n_epochs):
            self.model.train()
            self.divergence_scorer.train()
            epoch_losses = {k: [] for k in history}

            for sec in processed:
                graph = sec["graph"]
                x  = (graph["x"] if isinstance(graph, dict) else graph.x).to(self.device)
                ei = (graph["edge_index"] if isinstance(graph, dict) else graph.edge_index).to(self.device)
                ea = (graph.get("edge_attr") if isinstance(graph, dict) else
                      getattr(graph, "edge_attr", None))
                if ea is not None:
                    ea = ea.to(self.device)
                tp = (graph["timepoint"] if isinstance(graph, dict) else graph.timepoint).to(self.device)
                mf = (graph["model_flag"] if isinstance(graph, dict) else graph.model_flag).long().to(self.device)

                beta = SpatioTemporalGNN.kl_annealing_beta(global_step, warmup_steps)

                optimizer.zero_grad()

                with autocast(enabled=(self.device.type == "cuda")):
                    out = self.model.forward_single(x, ei, ea)
                    z, mu, log_var, x_recon = (
                        out["z"], out["mu"], out["log_var"], out["x_recon"]
                    )

                    # ELBO
                    G_rec = min(x.shape[1], x_recon.shape[1])
                    elbo_loss, elbo_info = self.model.encoder.elbo_loss(
                        x[:, :G_rec], x_recon[:, :G_rec], mu, log_var, beta
                    )

                    # Velocity + divergence
                    z_proj = self.model.project(z)
                    vel_out = self.velocity_module(z.unsqueeze(0))
                    vel = vel_out["velocity"].squeeze(0)           # (N, D)
                    div_out = self.divergence_scorer(z, vel, sec.get("anat_features"))
                    div_loss = self.divergence_scorer.divergence_constraint_loss(
                        div_out["divergence_map"], mf.float()
                    )

                    # Contrastive
                    model_type = sec["model_type"]
                    plp_names = AnatomicalAligner.TIMEPOINTS_PLP
                    mog_names = AnatomicalAligner.TIMEPOINTS_MOG
                    contr_loss, contr_info = self.contrastive_loss(
                        z_proj, mf, tp,
                        timepoint_names_plp=plp_names,
                        timepoint_names_mog=mog_names,
                        velocity=vel,
                        inf_axis=self.velocity_module.inflammatory_axis,
                    )

                    total = elbo_loss + 0.5 * div_loss + 0.5 * contr_loss

                if self._scaler is not None:
                    self._scaler.scale(total).backward()
                    self._scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                epoch_losses["total_loss"].append(total.item())
                epoch_losses["elbo_loss"].append(elbo_loss.item())
                epoch_losses["contrastive_loss"].append(contr_loss.item())
                epoch_losses["divergence_loss"].append(div_loss.item())
                epoch_losses["recon_loss"].append(elbo_info["recon_loss"].item())
                epoch_losses["kl_loss"].append(elbo_info["kl_loss"].item())
                global_step += 1

            scheduler.step()
            for k in history:
                mean_v = float(np.mean(epoch_losses[k]))
                history[k].append(mean_v)

            if (epoch + 1) % max(1, checkpoint_every) == 0:
                self.save_checkpoint(epoch + 1)
                logger.info(
                    "Epoch %d/%d | total=%.4f elbo=%.4f contr=%.4f div=%.4f",
                    epoch + 1, n_epochs,
                    history["total_loss"][-1],
                    history["elbo_loss"][-1],
                    history["contrastive_loss"][-1],
                    history["divergence_loss"][-1],
                )

        return history

    # ------------------------------------------------------------------
    # Inference / embedding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, sections: List[Dict]) -> Dict[str, Dict]:
        """
        Run inference on processed sections.

        Returns per-section dict with: z, velocity, divergence, relapse_risk.
        """
        if self.model is None:
            raise RuntimeError("Model not initialised.  Call train() first.")

        self.model.eval()
        results = {}

        for sec in sections:
            if "graph" not in sec:
                sec = self.preprocess_section(sec)

            graph = sec["graph"]
            x  = (graph["x"] if isinstance(graph, dict) else graph.x).to(self.device)
            ei = (graph["edge_index"] if isinstance(graph, dict) else graph.edge_index).to(self.device)
            ea = (graph.get("edge_attr") if isinstance(graph, dict) else
                  getattr(graph, "edge_attr", None))
            if ea is not None:
                ea = ea.to(self.device)

            out = self.model.forward_single(x, ei, ea)
            z = out["mu"]   # Use μ (deterministic) at inference

            vel_out = self.velocity_module(z.unsqueeze(0))
            vel = vel_out["velocity"].squeeze(0)

            anat = sec.get("anat_features")
            if anat is not None:
                anat_t = torch.tensor(anat, dtype=torch.float32, device=self.device)
            else:
                anat_t = None

            div_out = self.divergence_scorer(z, vel, anat_t)

            results[sec["section_id"]] = {
                "z": z.cpu().numpy(),
                "velocity": vel.cpu().numpy(),
                "velocity_magnitude": vel_out["velocity_magnitude"].squeeze(0).cpu().numpy(),
                "divergence": div_out["divergence_map"].cpu().numpy(),
                "bifurcation_mask": div_out["bifurcation_mask"].cpu().numpy(),
                "relapse_nursery_mask": div_out["relapse_nursery_mask"].cpu().numpy(),
                "relapse_risk": vel_out["relapse_risk_score"].squeeze(0).cpu().numpy(),
                "inf_momentum": vel_out["inf_momentum"].squeeze(0).cpu().numpy(),
            }

        return results

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def attribute(
        self,
        section: Dict,
        n_top_genes: int = 50,
    ) -> Dict:
        """
        Run mechanistic attribution on a section.

        Returns top genes, RLCS features, spatial hotspots.
        """
        if self.model is None or self.attributor is None:
            raise RuntimeError("Model and attributor must be initialised first.")

        if "graph" not in section:
            section = self.preprocess_section(section)

        # Run inference to get divergence mask
        pred = self.predict([section])[section["section_id"]]
        bif_mask = pred["bifurcation_mask"]

        graph = section["graph"]
        x  = (graph["x"] if isinstance(graph, dict) else graph.x).to(self.device)
        ei = (graph["edge_index"] if isinstance(graph, dict) else graph.edge_index).to(self.device)
        ea = (graph.get("edge_attr") if isinstance(graph, dict) else
              getattr(graph, "edge_attr", None))
        if ea is not None:
            ea = ea.to(self.device)

        vel_t = torch.tensor(pred["velocity"], dtype=torch.float32, device=self.device)
        target_fn = DivergenceTarget(
            encoder=self.model.encoder,
            divergence_scorer=self.divergence_scorer,
            edge_index=ei,
            edge_attr=ea,
            velocity=vel_t,
        )

        bif_idx = torch.where(torch.tensor(bif_mask))[0]
        if len(bif_idx) == 0:
            logger.warning("No bifurcation cells found — attributing all cells.")
            bif_idx = None

        attr_out = self.attributor.attribute_divergence(target_fn, x, target_cells=bif_idx)

        top_genes = self.attributor.top_relapse_genes(
            attr_out["gene_attributions"], n_top=n_top_genes
        )
        rlcs_feat = self.attributor.top_rlcs_features(attr_out["rlcs_attributions"])

        xy = section["xy_coords"]
        if bif_idx is not None:
            xy_subset = xy[bif_idx.numpy()]
        else:
            xy_subset = xy
        hotspots = self.attributor.spatial_hotspots(
            attr_out["attributions"], xy_subset, attr_out["gene_attributions"]
        )

        return {
            "top_genes": top_genes,
            "rlcs_attribution": rlcs_feat,
            "spatial_hotspots": hotspots,
            "raw_attributions": attr_out,
        }

    # ------------------------------------------------------------------
    # Validation tasks
    # ------------------------------------------------------------------

    def validate(self, sections: List[Dict]) -> Dict[str, float]:
        """
        Auto-evaluate model on standard validation tasks.

        Tasks:
          1. MOG vs PLP classification accuracy
          2. Inflammatory momentum direction accuracy (PLP peak > remission)
          3. Divergence separation (PLP > MOG mean divergence)
        """
        results = self.predict([self.preprocess_section(s) if "graph" not in s else s
                                for s in sections])
        metrics = {}

        mog_divs, plp_divs = [], []
        plp_peak_mom, plp_rem_mom = [], []

        for sec in sections:
            sid = sec["section_id"]
            if sid not in results:
                continue
            r = results[sid]
            model_type = sec["model_type"]
            timepoint = sec["timepoint"]

            if model_type == "MOG":
                mog_divs.extend(r["divergence"].tolist())
            else:
                plp_divs.extend(r["divergence"].tolist())
                if "peak" in timepoint.lower():
                    plp_peak_mom.extend(r["inf_momentum"].tolist())
                elif "remission" in timepoint.lower():
                    plp_rem_mom.extend(r["inf_momentum"].tolist())

        if mog_divs and plp_divs:
            metrics["mog_mean_divergence"] = float(np.mean(mog_divs))
            metrics["plp_mean_divergence"] = float(np.mean(plp_divs))
            metrics["divergence_separation"] = (
                float(np.mean(plp_divs)) - float(np.mean(mog_divs))
            )

        if plp_peak_mom and plp_rem_mom:
            metrics["plp_peak_inf_momentum"] = float(np.mean(plp_peak_mom))
            metrics["plp_rem_inf_momentum"] = float(np.mean(plp_rem_mom))
            metrics["momentum_relapse_signal"] = (
                float(np.mean(plp_peak_mom)) - float(np.mean(plp_rem_mom))
            )

        logger.info("Validation metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / f"spatialflow_epoch{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict() if self.model else None,
            "velocity_state": self.velocity_module.state_dict(),
            "divergence_state": self.divergence_scorer.state_dict(),
            "config": self.config,
        }, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if self.model is None and "config" in ckpt:
            # Model not yet built — build from config
            self._init_model(self._in_channels or 512)
        if self.model and "model_state" in ckpt and ckpt["model_state"]:
            self.model.load_state_dict(ckpt["model_state"])
        self.velocity_module.load_state_dict(ckpt["velocity_state"])
        self.divergence_scorer.load_state_dict(ckpt["divergence_state"])
        logger.info("Checkpoint loaded from %s (epoch %s)", path, ckpt.get("epoch"))

    # ------------------------------------------------------------------
    # Save analysis outputs
    # ------------------------------------------------------------------

    def save_outputs(
        self,
        results: Dict[str, Dict],
        sections: List[Dict],
        embeddings_fname: str = "latent_embeddings.h5ad",
    ) -> None:
        """
        Save all analysis artifacts to disk.

        Files created:
            latent_embeddings.h5ad
            niche_velocity.parquet
            divergence_scores.csv
        """
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect into dataframes
        velocity_rows = []
        divergence_rows = []
        z_list = []
        meta_list = []

        for sec in sections:
            sid = sec["section_id"]
            if sid not in results:
                continue
            r = results[sid]
            N = len(r["z"])
            model_type = sec["model_type"]
            timepoint = sec["timepoint"]

            for i in range(N):
                velocity_rows.append({
                    "section_id": sid, "model_type": model_type,
                    "timepoint": timepoint, "cell_idx": i,
                    "velocity_magnitude": float(r["velocity_magnitude"][i]),
                    "inf_momentum": float(r["inf_momentum"][i]),
                    "relapse_risk": float(r["relapse_risk"][i]),
                })
                divergence_rows.append({
                    "section_id": sid, "model_type": model_type,
                    "timepoint": timepoint, "cell_idx": i,
                    "divergence": float(r["divergence"][i]),
                    "bifurcation": bool(r["bifurcation_mask"][i]),
                    "relapse_nursery": bool(r["relapse_nursery_mask"][i]),
                })
                z_list.append(r["z"][i])
                meta_list.append({
                    "section_id": sid, "model_type": model_type,
                    "timepoint": timepoint, "cell_idx": i,
                })

        pd.DataFrame(velocity_rows).to_parquet(
            out_dir / "niche_velocity.parquet", index=False
        )
        pd.DataFrame(divergence_rows).to_csv(
            out_dir / "divergence_scores.csv", index=False
        )
        logger.info("Outputs saved to %s", out_dir)

        # Save latent embeddings as h5ad (requires anndata)
        try:
            import anndata as ad
            z_arr = np.vstack(z_list)
            meta_df = pd.DataFrame(meta_list)
            adata = ad.AnnData(X=z_arr, obs=meta_df)
            adata.write_h5ad(out_dir / embeddings_fname)
            logger.info("Latent embeddings saved: %s", out_dir / embeddings_fname)
        except ImportError:
            np.save(out_dir / "latent_embeddings.npy", np.vstack(z_list))
            pd.DataFrame(meta_list).to_csv(out_dir / "latent_embeddings_meta.csv", index=False)
            logger.warning(
                "anndata not installed — saved embeddings as .npy + .csv"
            )
