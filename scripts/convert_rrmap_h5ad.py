#!/usr/bin/env python3
"""
Convert RRmap-style .h5ad into SpatialFlow train.py directory layout.

Expected output tree:
    output_dir/
      MOG/<timepoint>/<section>/cells.parquet
      MOG/<timepoint>/<section>/tissue_mask.npy
      MOG/<timepoint>/<section>/cc_mask.npy
      MOG/<timepoint>/<section>/gm_mask.npy
      PLP/<timepoint>/<section>/...

This converter is intentionally conservative:
1) It reads .h5ad using h5py to avoid anndata version mismatches.
2) It maps RRmap metadata labels to SpatialFlow model/timepoint labels.
3) It can downsample each section so you can run a first pass cheaply.
4) It creates simple synthetic masks so the pipeline can execute.

If you have real tissue/CC/GM masks, replace the generated .npy files.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("convert_rrmap_h5ad")

TIMEPOINTS_MOG = {
    "baseline",
    "non_symptomatic",
    "early_onset",
    "chronic_peak",
    "chronic_long",
}
TIMEPOINTS_PLP = {
    "baseline",
    "onset_I",
    "onset_II",
    "monophasic",
    "peak_I",
    "remission_I",
    "peak_II",
    "remission_II",
    "peak_III",
}


def _norm_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def _decode_string_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"O", "S", "U"}:
        return np.array(
            [
                v.decode("utf-8", errors="ignore")
                if isinstance(v, (bytes, bytearray))
                else str(v)
                for v in values
            ],
            dtype=object,
        )
    return values.astype(object, copy=False)


def read_obs_column(obs_group: h5py.Group, column: str) -> np.ndarray:
    if column not in obs_group:
        raise KeyError(f"obs column not found: {column}")

    obj = obs_group[column]
    if isinstance(obj, h5py.Dataset):
        return _decode_string_array(obj[:])

    enc_type = obj.attrs.get("encoding-type", "")
    if enc_type != "categorical":
        raise ValueError(
            f"Unsupported obs[{column}] encoding-type={enc_type!r}; expected categorical or dataset."
        )

    categories = _decode_string_array(obj["categories"][:])
    codes = obj["codes"][:].astype(np.int64, copy=False)

    out = np.empty(codes.shape[0], dtype=object)
    out[:] = None
    valid = codes >= 0
    out[valid] = categories[codes[valid]]
    return out


def read_var_names(var_group: h5py.Group) -> np.ndarray:
    for key in ("gene_symbols", "index", "_index"):
        if key in var_group:
            values = _decode_string_array(var_group[key][:])
            if len(values) > 0:
                return values
    raise KeyError("Could not find var gene names in any of: gene_symbols, index, _index")


def pick_gene_indices(var_group: h5py.Group, n_genes: int) -> np.ndarray:
    n_vars = int(var_group["index"].shape[0]) if "index" in var_group else int(var_group["gene_symbols"].shape[0])
    n_genes = max(1, min(n_genes, n_vars))

    if "highly_variable" in var_group:
        hv = np.asarray(var_group["highly_variable"][:]).astype(bool)
        hv_idx = np.where(hv)[0]
        if hv_idx.size > 0:
            if hv_idx.size > n_genes:
                if "highly_variable_rank" in var_group:
                    ranks = np.asarray(var_group["highly_variable_rank"][:], dtype=np.float64)
                    rank_vals = ranks[hv_idx]
                    rank_vals = np.where(np.isfinite(rank_vals), rank_vals, np.inf)
                    order = np.argsort(rank_vals, kind="stable")
                    return hv_idx[order[:n_genes]]
                return hv_idx[:n_genes]
            return hv_idx

    return np.arange(n_genes, dtype=np.int64)


def map_model_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    norm = _norm_label(str(label))
    mapping = {
        "chronic": "MOG",
        "mog": "MOG",
        "relapse_remitting": "PLP",
        "rr": "PLP",
        "plp": "PLP",
    }
    return mapping.get(norm)


def map_timepoint_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    norm = _norm_label(str(label))
    mapping = {
        "mog_cfa": "baseline",
        "plp_cfa": "baseline",
        "baseline": "baseline",
        "non_symptomatic": "non_symptomatic",
        "non_symptomatics": "non_symptomatic",
        "non_symptomatic_phase": "non_symptomatic",
        "non_symptomatic_stage": "non_symptomatic",
        "early_onset": "early_onset",
        "chronic_peak": "chronic_peak",
        "chronic_long": "chronic_long",
        "onset_i": "onset_I",
        "onset_ii": "onset_II",
        "monophasic": "monophasic",
        "peak_i": "peak_I",
        "peak_ii": "peak_II",
        "peak_iii": "peak_III",
        "remitt_i": "remission_I",
        "remitt_ii": "remission_II",
        "remission_i": "remission_I",
        "remission_ii": "remission_II",
    }
    return mapping.get(norm)


def majority(values: Sequence[Optional[str]]) -> Optional[str]:
    counts = Counter(v for v in values if v is not None and str(v) != "")
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def sanitize_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return out.strip("_") or "section"


def extract_rows_dense(
    x_group: h5py.Group,
    row_indices: np.ndarray,
    selected_gene_idx: np.ndarray,
    n_vars: int,
) -> np.ndarray:
    indptr = x_group["indptr"]
    indices = x_group["indices"]
    data = x_group["data"]

    col_map = np.full(n_vars, -1, dtype=np.int32)
    col_map[selected_gene_idx] = np.arange(selected_gene_idx.size, dtype=np.int32)

    out = np.zeros((row_indices.size, selected_gene_idx.size), dtype=np.float32)
    for i, r in enumerate(row_indices):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        cols = indices[start:end]
        vals = data[start:end]
        mapped = col_map[cols]
        keep = mapped >= 0
        if np.any(keep):
            out[i, mapped[keep]] = vals[keep]
    return out


def build_masks(
    x: np.ndarray,
    y: np.ndarray,
    coord_scale_um: float,
    pad: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_local = (x - np.min(x)) / coord_scale_um + pad
    y_local = (y - np.min(y)) / coord_scale_um + pad

    w = int(np.ceil(float(np.max(x_local)))) + pad + 1
    h = int(np.ceil(float(np.max(y_local)))) + pad + 1
    w = max(w, 32)
    h = max(h, 32)

    tissue = np.ones((h, w), dtype=np.uint8)

    cy, cx = h // 2, w // 2
    rr, cc = np.ogrid[:h, :w]

    r_cc = max(2, min(h, w) // 40)
    cc_mask = (((rr - cy) ** 2 + (cc - cx) ** 2) <= (r_cc ** 2)).astype(np.uint8)

    gm_mask = (
        ((rr - cy) ** 2) / (0.30 * h * h) + ((cc - cx) ** 2) / (0.22 * w * w) <= 1.0
    ).astype(np.uint8)
    gm_mask[cc_mask > 0] = 0

    return x_local.astype(np.float32), y_local.astype(np.float32), tissue, cc_mask, gm_mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RRmap .h5ad to SpatialFlow section directory layout."
    )
    parser.add_argument("--h5ad", required=True, type=Path, help="Input .h5ad file")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output root folder")
    parser.add_argument("--n_genes", type=int, default=500, help="Number of genes to export")
    parser.add_argument(
        "--max_cells_per_section",
        type=int,
        default=20000,
        help="Random downsample per section (0 disables downsampling)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coord_scale_um", type=float, default=8.0)
    parser.add_argument("--mask_pad_px", type=int, default=8)

    parser.add_argument("--section_col", default="output_folder")
    parser.add_argument("--model_col", default="model")
    parser.add_argument("--timepoint_col", default="course")
    parser.add_argument("--cell_type_col", default="annotation")
    parser.add_argument("--x_col", default="centroid_x")
    parser.add_argument("--y_col", default="centroid_y")

    parser.add_argument(
        "--skip_unknown",
        action="store_true",
        help="Skip sections with unmapped model/timepoint instead of raising",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # SpatialFlow expects parquet input; fail early if parquet engine is missing.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Parquet engine not found. Install one of: `pip install pyarrow` "
                "or `pip install fastparquet`."
            ) from exc

    if not args.h5ad.exists():
        raise FileNotFoundError(f"Input file not found: {args.h5ad}")

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.h5ad, "r") as f:
        if "obs" not in f or "var" not in f or "X" not in f:
            raise ValueError("Input file missing one of required groups: obs, var, X")

        obs = f["obs"]
        var = f["var"]
        x_group = f["X"]

        if not isinstance(x_group, h5py.Group) or "indptr" not in x_group:
            raise ValueError("Expected X to be CSR group with data/indices/indptr.")

        n_obs = int(x_group.attrs["shape"][0])
        n_vars = int(x_group.attrs["shape"][1])
        LOGGER.info("Input shape: %d cells x %d genes", n_obs, n_vars)

        section_vals = read_obs_column(obs, args.section_col)
        model_raw = read_obs_column(obs, args.model_col)
        tp_raw = read_obs_column(obs, args.timepoint_col)
        cell_type_raw = read_obs_column(obs, args.cell_type_col)
        x_vals = np.asarray(obs[args.x_col][:], dtype=np.float32)
        y_vals = np.asarray(obs[args.y_col][:], dtype=np.float32)

        model_mapped = np.array([map_model_label(v) for v in model_raw], dtype=object)
        tp_mapped = np.array([map_timepoint_label(v) for v in tp_raw], dtype=object)

        var_names = read_var_names(var)
        selected_gene_idx = pick_gene_indices(var, n_genes=args.n_genes)
        selected_gene_names = var_names[selected_gene_idx]
        LOGGER.info("Selected %d genes", selected_gene_idx.size)

        section_to_rows: Dict[str, List[int]] = defaultdict(list)
        for i, sec in enumerate(section_vals):
            if sec is None:
                continue
            sec_name = str(sec)
            if sec_name == "":
                continue
            section_to_rows[sec_name].append(i)

        LOGGER.info("Found %d section groups", len(section_to_rows))

        manifest_rows = []

        for sec_name in sorted(section_to_rows):
            rows = np.asarray(section_to_rows[sec_name], dtype=np.int64)
            if rows.size == 0:
                continue

            model_type = majority(model_mapped[rows])
            timepoint = majority(tp_mapped[rows])

            if model_type is None or timepoint is None:
                msg = f"Unmapped model/timepoint for section {sec_name!r}"
                if args.skip_unknown:
                    LOGGER.warning("%s; skipping.", msg)
                    continue
                raise ValueError(msg)

            if model_type == "MOG" and timepoint not in TIMEPOINTS_MOG:
                msg = f"Invalid MOG timepoint {timepoint!r} in section {sec_name!r}"
                if args.skip_unknown:
                    LOGGER.warning("%s; skipping.", msg)
                    continue
                raise ValueError(msg)
            if model_type == "PLP" and timepoint not in TIMEPOINTS_PLP:
                msg = f"Invalid PLP timepoint {timepoint!r} in section {sec_name!r}"
                if args.skip_unknown:
                    LOGGER.warning("%s; skipping.", msg)
                    continue
                raise ValueError(msg)

            if args.max_cells_per_section > 0 and rows.size > args.max_cells_per_section:
                rows = np.sort(
                    rng.choice(rows, size=args.max_cells_per_section, replace=False)
                )

            x_sec = x_vals[rows]
            y_sec = y_vals[rows]
            ct_sec = np.array(
                [str(v) if v is not None else "unknown" for v in cell_type_raw[rows]],
                dtype=object,
            )

            expr_sec = extract_rows_dense(
                x_group=x_group,
                row_indices=rows,
                selected_gene_idx=selected_gene_idx,
                n_vars=n_vars,
            )

            x_local, y_local, tissue, cc_mask, gm_mask = build_masks(
                x=x_sec,
                y=y_sec,
                coord_scale_um=args.coord_scale_um,
                pad=args.mask_pad_px,
            )

            section_dir = out_root / model_type / timepoint / sanitize_name(sec_name)
            section_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(expr_sec, columns=selected_gene_names)
            df.insert(0, "cell_type", ct_sec)
            df.insert(0, "y", y_local)
            df.insert(0, "x", x_local)
            df.to_parquet(section_dir / "cells.parquet", index=False)

            np.save(section_dir / "tissue_mask.npy", tissue)
            np.save(section_dir / "cc_mask.npy", cc_mask)
            np.save(section_dir / "gm_mask.npy", gm_mask)

            LOGGER.info(
                "Wrote %s (%s/%s, %d cells)",
                section_dir,
                model_type,
                timepoint,
                rows.size,
            )
            manifest_rows.append(
                {
                    "section": sanitize_name(sec_name),
                    "model_type": model_type,
                    "timepoint": timepoint,
                    "n_cells": int(rows.size),
                    "n_genes": int(selected_gene_idx.size),
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["model_type", "timepoint", "section"], ignore_index=True
    )
    manifest_path = out_root / "conversion_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info(
        "Done. Set config resolution_um to %.3f to match coordinate downscaling.",
        args.coord_scale_um,
    )


if __name__ == "__main__":
    main()
