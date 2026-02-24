#!/usr/bin/env python3
"""
EAE-SpatialFlow v4 — Evaluation and Visualization Script

Usage
─────
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/spatialflow_epoch0050.pt \
    --data_dir /path/to/xenium_data \
    --config configs/default_config.yaml \
    --output_dir outputs/evaluation \
    --device cuda

Generates all required visualizations:
    1. Niche trajectory UMAP (per model, all timepoints combined)
    2. Velocity vector fields (per timepoint)
    3. Divergence heatmaps (per timepoint)
    4. Relapse probability maps (PLP sections)
    5. Divergence trajectory plot (mean divergence over time)
    6. Niche cluster map (after Leiden clustering in Z-space)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from spatialflow import EAESpatialFlow, VisualizationEngine
from spatialflow.divergence import PAGAGraph
from scripts.train import load_all_sections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Leiden clustering in latent space
# ---------------------------------------------------------------------------

def leiden_cluster_z(z: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    """
    Run Leiden community detection on a KNN graph in Z-space.

    Requires igraph + leidenalg.

    Returns (N,) int array of cluster labels.
    """
    try:
        import igraph as ig
        import leidenalg
        from scipy.spatial import cKDTree

        k = 15
        tree = cKDTree(z)
        _, nn_idx = tree.query(z, k=k + 1)
        nn_idx = nn_idx[:, 1:]   # exclude self

        N = len(z)
        edges = []
        for i in range(N):
            for j in nn_idx[i]:
                if i < j:
                    edges.append((i, int(j)))

        g = ig.Graph(n=N, edges=edges, directed=False)
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=42,
        )
        labels = np.array(partition.membership, dtype=int)
        n_clusters = labels.max() + 1
        logger.info("Leiden: %d clusters found (resolution=%.2f).", n_clusters, resolution)
        return labels

    except ImportError:
        logger.warning(
            "igraph / leidenalg not installed — using KMeans fallback."
        )
        from sklearn.cluster import KMeans
        k = max(2, min(20, len(z) // 100))
        labels = KMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(z)
        return labels.astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EAE-SpatialFlow v4 and generate visualizations"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/default_config.yaml", type=Path)
    parser.add_argument("--output_dir", default="outputs/evaluation", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--leiden_resolution", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["output_dir"] = str(args.output_dir)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    pipeline = EAESpatialFlow(config=config, device=args.device,
                              output_dir=args.output_dir)
    pipeline.load_checkpoint(args.checkpoint)

    logger.info("Loading sections …")
    sections = load_all_sections(args.data_dir)
    if not sections:
        logger.error("No sections found.")
        sys.exit(1)

    logger.info("Running inference …")
    results = pipeline.predict(sections)

    # Collect all Z, metadata, spatial coords
    all_z, all_labels, all_model, all_tp, all_xy = [], [], [], [], []
    all_vel, all_vel_mag, all_div, all_bif, all_risk = [], [], [], [], []

    for sec in sections:
        sid = sec["section_id"]
        if sid not in results:
            continue
        r = results[sid]
        N = len(r["z"])
        all_z.append(r["z"])
        all_model.extend([sec["model_type"]] * N)
        all_tp.extend([sec["timepoint"]] * N)
        all_vel.append(r["velocity"])
        all_vel_mag.append(r["velocity_magnitude"])
        all_div.append(r["divergence"])
        all_bif.append(r["bifurcation_mask"])
        all_risk.append(r["relapse_risk"])
        xy = sec["xy_coords"][:N]
        all_xy.append(xy)

    Z = np.vstack(all_z)
    vel_all = np.vstack(all_vel)
    vel_mag_all = np.concatenate(all_vel_mag)
    div_all = np.concatenate(all_div)
    bif_all = np.concatenate(all_bif)
    risk_all = np.concatenate(all_risk)
    xy_all = np.vstack(all_xy)

    model_arr = np.array(all_model)
    tp_arr = np.array(all_tp)

    logger.info("Total cells: %d | Computing UMAP …", len(Z))

    # --- UMAP ---
    try:
        from umap import UMAP
        reducer = UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        umap_2d = reducer.fit_transform(Z)
    except ImportError:
        from sklearn.decomposition import PCA
        umap_2d = PCA(n_components=2).fit_transform(Z)
        logger.warning("umap-learn not available — using PCA projection.")

    # --- Leiden clustering ---
    cluster_labels = leiden_cluster_z(Z, resolution=args.leiden_resolution)

    # Default niche archetype names (user should annotate from biology)
    n_clusters = cluster_labels.max() + 1
    archetype_names = {
        i: f"Niche {i}" for i in range(n_clusters)
    }
    # Override with biologically meaningful names if K >= 3
    if n_clusters >= 3:
        top_div_clusters = np.argsort(
            [div_all[cluster_labels == k].mean() for k in range(n_clusters)]
        )
        archetype_names[top_div_clusters[-1]] = "Relapse Spark"
        archetype_names[top_div_clusters[0]] = "Resolution Sink"
        if n_clusters >= 4:
            archetype_names[top_div_clusters[-2]] = "Fibrotic Gatekeeper"

    viz = VisualizationEngine(
        output_dir=str(args.output_dir / "figures"),
        dpi=config.get("figure_dpi", 150),
    )

    # --- Trajectory UMAP coloured by model ---
    model_int = (model_arr == "PLP").astype(int)
    viz.plot_trajectory_umap(
        z=Z, labels=model_int,
        label_map={0: "MOG (Chronic)", 1: "PLP (Relapsing-Remitting)"},
        umap_coords=umap_2d,
        title="EAE-SpatialFlow: MOG vs PLP Trajectory",
        save_name="trajectory_umap_model.pdf",
    )

    # --- Trajectory UMAP coloured by niche cluster ---
    viz.plot_trajectory_umap(
        z=Z, labels=cluster_labels,
        label_map=archetype_names,
        umap_coords=umap_2d,
        title="Niche State Clusters",
        save_name="trajectory_umap_clusters.pdf",
    )

    # --- Project velocity to 2D for UMAP arrows ---
    try:
        vel_2d = VisualizationEngine.project_velocity_to_2d(vel_all, umap_2d, Z)
        viz.plot_trajectory_umap(
            z=Z, labels=model_int,
            label_map={0: "MOG", 1: "PLP"},
            umap_coords=umap_2d,
            velocities_2d=vel_2d,
            title="Trajectory UMAP with Velocity Arrows",
            save_name="trajectory_umap_velocity.pdf",
        )
    except Exception as e:
        logger.warning("Velocity 2D projection failed: %s", e)

    # --- Spatial divergence heatmap (all sections combined) ---
    viz.plot_divergence_heatmap(
        xy_coords=xy_all,
        divergence=div_all,
        bifurcation_mask=bif_all,
        title="Global Divergence Heatmap",
        save_name="divergence_heatmap_all.pdf",
    )

    # --- Relapse probability map (PLP only) ---
    plp_mask = model_arr == "PLP"
    if plp_mask.sum() > 0:
        viz.plot_relapse_probability(
            xy_coords=xy_all[plp_mask],
            relapse_risk=risk_all[plp_mask],
            relapse_nursery_mask=bif_all[plp_mask],
            title="PLP Relapse Risk Map",
            save_name="relapse_probability_plp.pdf",
        )

    # --- Niche cluster spatial map ---
    viz.plot_niche_clusters(
        xy_coords=xy_all,
        cluster_labels=cluster_labels,
        cluster_names=archetype_names,
        title="Spatial Niche Cluster Map",
        save_name="niche_cluster_spatial.pdf",
    )

    # --- Divergence trajectory (per model) ---
    from spatialflow.aligner import AnatomicalAligner
    for model_type, tp_list in [
        ("MOG", AnatomicalAligner.TIMEPOINTS_MOG),
        ("PLP", AnatomicalAligner.TIMEPOINTS_PLP),
    ]:
        mean_divs, bif_fracs = [], []
        for tp in tp_list:
            mask = (model_arr == model_type) & (tp_arr == tp)
            if mask.sum() == 0:
                mean_divs.append(0.0)
                bif_fracs.append(0.0)
            else:
                mean_divs.append(float(div_all[mask].mean()))
                bif_fracs.append(float(bif_all[mask].mean()))

        viz.plot_divergence_trajectory(
            mean_div_seq=np.array(mean_divs),
            bif_frac_seq=np.array(bif_fracs),
            timepoint_names=tp_list,
            model_type=model_type,
            save_name=f"divergence_trajectory_{model_type.lower()}.pdf",
        )

    # --- PAGA connectivity ---
    if len(Z) < 50000:   # PAGA is memory-intensive for >50k cells
        try:
            paga = PAGAGraph(cluster_labels=cluster_labels, z=Z, knn_k=15)
            conn = paga.compute_connectivity()
            np.save(str(args.output_dir / "paga_connectivity.npy"), conn)
            pd.DataFrame(conn, index=list(archetype_names.values()),
                         columns=list(archetype_names.values())).to_csv(
                str(args.output_dir / "paga_connectivity.csv")
            )
            logger.info("PAGA connectivity saved.")
        except Exception as e:
            logger.warning("PAGA computation failed: %s", e)

    # Save cluster labels
    pd.DataFrame({
        "section_id": [s["section_id"] for s in sections for _ in range(
            len(results.get(s["section_id"], {}).get("z", [])))],
        "cluster": cluster_labels,
        "niche_archetype": [archetype_names.get(int(c), f"Cluster {c}")
                            for c in cluster_labels],
        "model_type": all_model,
        "timepoint": all_tp,
        "divergence": div_all,
        "relapse_risk": risk_all,
    }).to_csv(args.output_dir / "niche_annotations.csv", index=False)

    logger.info("Evaluation complete.  Figures in: %s", args.output_dir / "figures")


if __name__ == "__main__":
    main()
