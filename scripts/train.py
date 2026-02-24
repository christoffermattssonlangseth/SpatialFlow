#!/usr/bin/env python3
"""
EAE-SpatialFlow v4 — Training Script

Usage
─────
python scripts/train.py \
    --data_dir /path/to/xenium_data \
    --config configs/default_config.yaml \
    --output_dir outputs/ \
    --device cuda \
    --n_epochs 50

Data directory structure expected:
    data_dir/
        MOG/
            baseline/
                section1/
                    cells.parquet        (x, y, cell_type, gene columns)
                    tissue_mask.npy
                    cc_mask.npy
                    gm_mask.npy
            non_symptomatic/
            early_onset/
            chronic_peak/
            chronic_long/
        PLP/
            baseline/
            onset_I/
            ...
            peak_III/

cells.parquet columns:
    x, y: µm coordinates
    cell_type: string
    <gene_name>: log1p-normalised expression (one column per gene)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch

# Ensure package on path
sys.path.insert(0, str(Path(__file__).parent.parent))
from spatialflow import EAESpatialFlow
from spatialflow.aligner import AnatomicalAligner
from spatialflow.visualization import VisualizationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_section(
    section_dir: Path,
    model_type: str,
    timepoint: str,
    section_id: str,
) -> dict:
    """
    Load one Xenium section from disk.

    Returns a section dict compatible with EAESpatialFlow.preprocess_section().
    """
    cells_path = section_dir / "cells.parquet"
    tissue_path = section_dir / "tissue_mask.npy"
    cc_path = section_dir / "cc_mask.npy"
    gm_path = section_dir / "gm_mask.npy"

    for p in [cells_path, tissue_path, cc_path, gm_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df = pd.read_parquet(cells_path)
    gene_cols = [c for c in df.columns if c not in ("x", "y", "cell_type")]

    xy_coords = df[["x", "y"]].values.astype(np.float32)
    expr_matrix = df[gene_cols].values.astype(np.float32)
    gene_names = gene_cols
    cell_types = df["cell_type"].tolist()

    tissue_mask = np.load(tissue_path)
    cc_mask = np.load(cc_path)
    gm_mask = np.load(gm_path)

    return {
        "section_id": section_id,
        "model_type": model_type,
        "timepoint": timepoint,
        "xy_coords": xy_coords,
        "expr_matrix": expr_matrix,
        "gene_names": gene_names,
        "cell_types": cell_types,
        "tissue_mask": tissue_mask,
        "cc_mask": cc_mask,
        "gm_mask": gm_mask,
    }


def load_all_sections(data_dir: Path) -> list:
    """
    Walk data_dir and load all available model/timepoint/section combinations.
    """
    sections = []
    for model_type in ("MOG", "PLP"):
        model_dir = data_dir / model_type
        if not model_dir.exists():
            logger.warning("Model directory not found: %s", model_dir)
            continue

        # Validate timepoints against expected list
        valid_timepoints = (
            AnatomicalAligner.TIMEPOINTS_MOG if model_type == "MOG"
            else AnatomicalAligner.TIMEPOINTS_PLP
        )

        for tp in valid_timepoints:
            tp_dir = model_dir / tp
            if not tp_dir.exists():
                continue

            for sec_dir in sorted(tp_dir.iterdir()):
                if not sec_dir.is_dir():
                    continue
                section_id = f"{model_type}_{tp}_{sec_dir.name}"
                try:
                    sec = load_section(sec_dir, model_type, tp, section_id)
                    sections.append(sec)
                    logger.info("Loaded section: %s (%d cells)",
                                section_id, len(sec["xy_coords"]))
                except FileNotFoundError as e:
                    logger.warning("Skipping %s: %s", section_id, e)

    logger.info("Total sections loaded: %d", len(sections))
    return sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train EAE-SpatialFlow v4 framework"
    )
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/default_config.yaml", type=Path)
    parser.add_argument("--output_dir", default="outputs", type=Path)
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Override n_epochs from config")
    parser.add_argument("--n_top_genes", type=int, default=50)
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after training")
    parser.add_argument("--attribute", action="store_true",
                        help="Run attribution on high-divergence niches")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.n_epochs is not None:
        config["n_epochs"] = args.n_epochs

    # Device check
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        device = "cpu"

    logger.info("=" * 60)
    logger.info("EAE-SpatialFlow v4 — Training")
    logger.info("Device: %s | Config: %s", device, args.config)
    logger.info("=" * 60)

    # Load data
    sections = load_all_sections(args.data_dir)
    if not sections:
        logger.error("No sections loaded.  Check data_dir structure.")
        sys.exit(1)

    # Split train / val (hold out last section per model per timepoint)
    train_sections = sections[:-max(1, len(sections) // 10)]
    val_sections = sections[-max(1, len(sections) // 10):]
    logger.info("Train: %d sections | Val: %d sections",
                len(train_sections), len(val_sections))

    # Build pipeline
    pipeline = EAESpatialFlow(
        config=config,
        device=device,
        output_dir=args.output_dir,
    )

    # Train
    history = pipeline.train(
        train_sections,
        n_epochs=config.get("n_epochs", 50),
        lr=config.get("learning_rate", 1e-3),
        warmup_steps=config.get("warmup_steps", 500),
        checkpoint_every=config.get("checkpoint_every", 10),
    )
    logger.info("Training complete. Final loss: %.4f", history["total_loss"][-1])

    # Validate
    if args.validate and val_sections:
        metrics = pipeline.validate(val_sections)
        logger.info("Validation metrics:")
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)

    # Inference on all sections
    all_processed = pipeline._last_processed if hasattr(pipeline, "_last_processed") else []
    logger.info("Running inference on all sections …")
    results = pipeline.predict(sections)

    # Save outputs
    pipeline.save_outputs(results, sections)

    # Attribution
    if args.attribute:
        logger.info("Running attribution on PLP Remission I sections …")
        for sec in sections:
            if sec["model_type"] == "PLP" and "remission" in sec["timepoint"].lower():
                try:
                    attr = pipeline.attribute(sec, n_top_genes=args.n_top_genes)
                    pipeline.attributor.save_outputs(
                        output_dir=Path(args.output_dir) / "attribution" / sec["section_id"],
                        top_genes_df=attr["top_genes"],
                        rlcs_df=attr["rlcs_attribution"],
                        hotspots_df=attr["spatial_hotspots"],
                    )
                except Exception as e:
                    logger.warning("Attribution failed for %s: %s",
                                   sec["section_id"], e)

    logger.info("Pipeline complete.  Outputs in: %s", args.output_dir)


if __name__ == "__main__":
    main()
