"""
EAE-SpatialFlow v4 — Component Unit Tests

Tests all major modules with synthetic data:
    - AnatomicalAligner / RadialBasisTransform
    - SpatialGraphBuilder
    - SpatioTemporalGNN / NicheMomentumEncoder
    - NicheVelocity / InflammatoryAxisDefiner
    - DivergenceScorer
    - TrajectoryContrastiveLoss / HardTripletMiner
    - MechanisticAttributor
    - VisualizationEngine (smoke tests)
    - EAESpatialFlow pipeline (end-to-end)

Run with: python -m pytest tests/test_components.py -v
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tissue_masks():
    """Synthetic 100×100 tissue, central canal, and GM masks."""
    H, W = 100, 100
    tissue = np.ones((H, W), dtype=np.uint8)
    tissue[40:60, 45:55] = 0   # hollow lumen (CC region, not CC itself)

    cc = np.zeros((H, W), dtype=np.uint8)
    cc[48:52, 48:52] = 1       # small central canal

    gm = np.zeros((H, W), dtype=np.uint8)
    gm[30:70, 25:75] = 1       # oval GM region
    # Remove overlap with CC
    gm[48:52, 48:52] = 0

    return tissue, cc, gm


@pytest.fixture
def cell_data():
    """Synthetic cell data: 50 cells, 20 genes."""
    np.random.seed(42)
    N, G = 50, 20
    xy = np.random.uniform(0, 100, (N, 2)).astype(np.float32)
    expr = np.abs(np.random.randn(N, G)).astype(np.float32)
    genes = [f"gene_{i:03d}" for i in range(G)]
    cell_types = ["microglia", "astrocyte", "oligodendrocyte",
                  "t_cell", "neuron"] * (N // 5)
    return xy, expr, genes, cell_types


@pytest.fixture
def latent_z():
    """Synthetic latent embeddings."""
    torch.manual_seed(0)
    return torch.randn(30, 64)


# ---------------------------------------------------------------------------
# AnatomicalAligner / RadialBasisTransform
# ---------------------------------------------------------------------------

class TestAnatomicalAligner:

    def test_rbt_transform_shape(self, tissue_masks, cell_data):
        from spatialflow.aligner import RadialBasisTransform

        tissue, cc, gm = tissue_masks
        xy, *_ = cell_data

        rbt = RadialBasisTransform(
            tissue_mask=tissue, cc_mask=cc, gm_mask=gm, resolution_um=1.0
        )
        feats = rbt.transform(xy)

        assert feats.shape == (len(xy), 6), f"Expected (N, 6), got {feats.shape}"
        assert not np.isnan(feats).any(), "NaN in anatomical features"
        assert not np.isinf(feats[:, :2]).any(), "Inf in geodesic distances"

    def test_rbt_theta_range(self, tissue_masks, cell_data):
        from spatialflow.aligner import RadialBasisTransform

        tissue, cc, gm = tissue_masks
        xy, *_ = cell_data
        rbt = RadialBasisTransform(tissue, cc, gm)
        feats = rbt.transform(xy)
        # θ column (index 2)
        theta = feats[:, 2]
        assert (theta >= -np.pi - 1e-6).all()
        assert (theta <= np.pi + 1e-6).all()

    def test_rbt_mirror(self, tissue_masks, cell_data):
        from spatialflow.aligner import RadialBasisTransform

        tissue, cc, gm = tissue_masks
        xy, *_ = cell_data
        rbt = RadialBasisTransform(tissue, cc, gm)
        mirrored = rbt.mirror_coordinates(xy)
        assert mirrored.shape == xy.shape
        # Hemisphere of mirrored should differ on average
        orig_hemi = rbt.transform(xy)[:, 5]
        mirror_hemi = rbt.transform(mirrored)[:, 5]
        # Some cells should change hemisphere
        assert not (orig_hemi == mirror_hemi).all()

    def test_aligner_fit_transform(self, tissue_masks, cell_data):
        from spatialflow.aligner import AnatomicalAligner

        tissue, cc, gm = tissue_masks
        xy, *_ = cell_data

        aligner = AnatomicalAligner(resolution_um=1.0, mirror_bilateral=False)
        aligner.fit_section("test_section", tissue, cc, gm)
        result = aligner.transform_section("test_section", xy)

        assert "features" in result
        assert result["features"].shape[0] == len(xy)

    def test_timepoint_encoding_mog(self):
        from spatialflow.aligner import AnatomicalAligner
        aligner = AnatomicalAligner()
        assert aligner.encode_timepoint("MOG", "baseline") == 0
        assert aligner.encode_timepoint("MOG", "chronic_long") == 4

    def test_timepoint_encoding_plp(self):
        from spatialflow.aligner import AnatomicalAligner
        aligner = AnatomicalAligner()
        assert aligner.encode_timepoint("PLP", "remission_I") == 5
        assert aligner.encode_timepoint("PLP", "peak_III") == 8

    def test_invalid_timepoint_raises(self):
        from spatialflow.aligner import AnatomicalAligner
        aligner = AnatomicalAligner()
        with pytest.raises(ValueError):
            aligner.encode_timepoint("MOG", "not_a_timepoint")


# ---------------------------------------------------------------------------
# SpatialGraphBuilder
# ---------------------------------------------------------------------------

class TestSpatialGraphBuilder:

    def test_build_returns_expected_keys(self, tissue_masks, cell_data):
        from spatialflow.aligner import AnatomicalAligner, RadialBasisTransform
        from spatialflow.graph_builder import SpatialGraphBuilder

        tissue, cc, gm = tissue_masks
        xy, expr, genes, cell_types = cell_data

        rbt = RadialBasisTransform(tissue, cc, gm)
        anat_feats = rbt.transform(xy)

        ct_emb = np.eye(5, dtype=np.float32)[[
            ["microglia", "astrocyte", "oligodendrocyte", "t_cell", "neuron"].index(ct)
            for ct in cell_types
        ]]
        tp_idx = np.zeros(len(xy), dtype=np.int64)

        builder = SpatialGraphBuilder(
            spatial_radius_um=20.0,
            device="cpu",
        )
        graph = builder.build(
            xy_coords=xy,
            expr_matrix=expr,
            gene_names=genes,
            anatomical_features=anat_feats,
            cell_type_embeddings=ct_emb,
            timepoint_idx=tp_idx,
            model_type="MOG",
            section_id="test",
        )

        # Check expected fields (dict or PyG Data)
        if isinstance(graph, dict):
            assert "x" in graph
            assert "edge_index" in graph
        else:
            assert hasattr(graph, "x")
            assert hasattr(graph, "edge_index")

    def test_edge_index_valid(self, tissue_masks, cell_data):
        from spatialflow.aligner import RadialBasisTransform
        from spatialflow.graph_builder import SpatialGraphBuilder

        tissue, cc, gm = tissue_masks
        xy, expr, genes, cell_types = cell_data
        rbt = RadialBasisTransform(tissue, cc, gm)
        anat_feats = rbt.transform(xy)
        ct_emb = np.zeros((len(xy), 5), dtype=np.float32)
        tp_idx = np.zeros(len(xy), dtype=np.int64)

        builder = SpatialGraphBuilder(spatial_radius_um=30.0, device="cpu")
        graph = builder.build(xy, expr, genes, anat_feats, ct_emb, tp_idx, "PLP", "s1")

        ei = graph["edge_index"] if isinstance(graph, dict) else graph.edge_index
        assert ei.shape[0] == 2
        N = len(xy)
        assert (ei >= 0).all()
        assert (ei < N).all()


# ---------------------------------------------------------------------------
# SpatioTemporalGNN
# ---------------------------------------------------------------------------

class TestSpatioTemporalGNN:

    def test_encoder_output_shapes(self):
        from spatialflow.model import NicheMomentumEncoder

        in_ch = 32
        latent_dim = 16
        N = 20
        E = 40

        encoder = NicheMomentumEncoder(
            in_channels=in_ch, hidden_dim=8, latent_dim=latent_dim,
            n_genes_reconstruct=10, heads=2, edge_dim=2
        )
        x = torch.randn(N, in_ch)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, 2)

        out = encoder(x, edge_index, edge_attr)
        assert out["z"].shape == (N, latent_dim)
        assert out["mu"].shape == (N, latent_dim)
        assert out["log_var"].shape == (N, latent_dim)
        assert out["x_recon"].shape == (N, 10)

    def test_elbo_loss_positive(self):
        from spatialflow.model import NicheMomentumEncoder

        N, G, D = 10, 5, 8
        enc = NicheMomentumEncoder(in_channels=10, hidden_dim=4, latent_dim=D,
                                   n_genes_reconstruct=G, heads=2, edge_dim=2)
        x = torch.randn(N, 10)
        ei = torch.randint(0, N, (2, 20))
        out = enc(x, ei)

        loss, info = enc.elbo_loss(
            x[:, :G], out["x_recon"], out["mu"], out["log_var"], beta=1.0
        )
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_gru_temporal_bridge(self):
        from spatialflow.model import GRUTemporalBridge

        T, N, D = 5, 10, 16
        bridge = GRUTemporalBridge(latent_dim=D, gru_hidden=32)
        z_seq = torch.randn(T, N, D)
        out = bridge(z_seq)
        assert out.shape == (T, N, D)

    def test_neural_ode_bridge(self):
        from spatialflow.model import LatentNeuralODEBridge

        T, N, D = 4, 8, 16
        bridge = LatentNeuralODEBridge(latent_dim=D, hidden_dim=32, n_steps=5)
        z_seq = torch.randn(T, N, D)
        out = bridge(z_seq)
        assert out.shape == (T, N, D)

    def test_full_gnn_forward(self):
        from spatialflow.model import SpatioTemporalGNN

        N, F, E, D = 15, 20, 30, 16
        model = SpatioTemporalGNN(
            in_channels=F, hidden_dim=8, latent_dim=D,
            n_genes_reconstruct=8, heads=2, edge_dim=2,
        )
        x = torch.randn(N, F)
        ei = torch.randint(0, N, (2, E))
        ea = torch.randn(E, 2)
        out = model.forward_single(x, ei, ea)
        assert "z" in out
        assert out["z"].shape == (N, D)


# ---------------------------------------------------------------------------
# NicheVelocity
# ---------------------------------------------------------------------------

class TestNicheVelocity:

    def test_velocity_shape(self, latent_z):
        from spatialflow.velocity import NicheVelocity

        T, D = 5, 64
        vel_module = NicheVelocity(latent_dim=D)
        z_seq = torch.randn(T, 10, D)
        out = vel_module(z_seq)

        assert out["velocity"].shape == (T, 10, D)
        assert out["acceleration"].shape == (T, 10, D)
        assert out["velocity_magnitude"].shape == (T, 10)
        assert out["accel_magnitude"].shape == (T, 10)

    def test_velocity_zero_at_t0(self):
        from spatialflow.velocity import NicheVelocity

        vel_module = NicheVelocity(latent_dim=8)
        z_seq = torch.randn(4, 5, 8)
        vel = vel_module.compute_velocity(z_seq)
        # First timepoint velocity should be zero (no prior)
        assert vel[0].abs().max().item() < 1e-6

    def test_set_inflammatory_axis(self, latent_z):
        from spatialflow.velocity import NicheVelocity

        vel_module = NicheVelocity(latent_dim=64)
        act = torch.randn(10, 64)
        qui = torch.randn(10, 64)
        vel_module.set_inflammatory_axis(act, qui)
        axis = vel_module.inflammatory_axis
        assert axis.shape == (64,)
        # Should be unit vector
        assert abs(axis.norm().item() - 1.0) < 1e-5

    def test_inflammatory_momentum(self, latent_z):
        from spatialflow.velocity import NicheVelocity

        vel_module = NicheVelocity(latent_dim=64)
        vel_module.set_inflammatory_axis(torch.randn(5, 64), torch.randn(5, 64))
        velocity = torch.randn(1, 10, 64)
        mom = vel_module.inflammatory_momentum(velocity)
        assert mom.shape == (1, 10)

    def test_axis_definer_matches_free_text_labels(self):
        from spatialflow.velocity import InflammatoryAxisDefiner, NicheVelocity

        vel_module = NicheVelocity(latent_dim=16)
        z = torch.randn(6, 16)
        cell_types = [
            "Activated Microglia/Macrophage",
            "IFN-γ-Activated Microglia",
            "Reactive Astrocyte",
            "Mature Oligodendrocyte",
            "CNS Endothelial Cell (spinal)",
            "Spinal Interneuron",
        ]

        InflammatoryAxisDefiner().extract_axis(z, cell_types, vel_module)
        assert vel_module._axis_set
        assert torch.isfinite(vel_module.inflammatory_axis).all()


# ---------------------------------------------------------------------------
# DivergenceScorer
# ---------------------------------------------------------------------------

class TestDivergenceScorer:

    def test_knn_divergence_chunked_large_n(self):
        from spatialflow.divergence import knn_divergence

        torch.manual_seed(0)
        N, D = 1000, 16
        z = torch.randn(N, D)
        velocity = torch.randn(N, D)

        div = knn_divergence(z, velocity, k=15)

        assert div.shape == (N,)
        assert torch.isfinite(div).all()

    def test_divergence_output_shape(self, latent_z):
        from spatialflow.divergence import DivergenceScorer

        N, D = latent_z.shape
        scorer = DivergenceScorer(latent_dim=D, knn_k=5, learn_correction=False)
        velocity = torch.randn(N, D)
        out = scorer(latent_z, velocity)

        assert out["divergence_map"].shape == (N,)
        assert out["bifurcation_mask"].shape == (N,)
        assert out["bifurcation_mask"].dtype == torch.bool

    def test_divergence_constraint_loss(self, latent_z):
        from spatialflow.divergence import DivergenceScorer

        N, D = latent_z.shape
        scorer = DivergenceScorer(latent_dim=D, knn_k=5, learn_correction=False)
        velocity = torch.randn(N, D)
        out = scorer(latent_z, velocity)
        model_flags = torch.randint(0, 2, (N,)).float()
        loss = scorer.divergence_constraint_loss(out["divergence_map"], model_flags)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# TrajectoryContrastiveLoss
# ---------------------------------------------------------------------------

class TestTrajectoryContrastiveLoss:

    def test_triplet_loss_positive(self):
        from spatialflow.losses import TrajectoryContrastiveLoss
        from spatialflow.aligner import AnatomicalAligner

        loss_fn = TrajectoryContrastiveLoss(margin=1.0)
        N, D = 60, 32

        # Create balanced PLP + MOG batch
        z = torch.randn(N, D)
        model_flags = torch.tensor([1] * 40 + [0] * 20, dtype=torch.int8)

        # PLP: timepoints 0..8 → need remission_I (5) and peak_II (6)
        tp = torch.zeros(N, dtype=torch.long)
        # Give first 15 PLP cells remission_I
        tp[:15] = AnatomicalAligner.TIMEPOINTS_PLP.index("remission_I")
        # Next 15 peak_II
        tp[15:30] = AnatomicalAligner.TIMEPOINTS_PLP.index("peak_II")
        # MOG: chronic_long
        tp[40:] = AnatomicalAligner.TIMEPOINTS_MOG.index("chronic_long")

        total, info = loss_fn(
            z, model_flags, tp,
            timepoint_names_plp=AnatomicalAligner.TIMEPOINTS_PLP,
            timepoint_names_mog=AnatomicalAligner.TIMEPOINTS_MOG,
        )
        assert not torch.isnan(total)
        assert total.item() >= 0

    def test_hard_miner_returns_indices(self):
        from spatialflow.losses import HardTripletMiner
        from spatialflow.aligner import AnatomicalAligner

        miner = HardTripletMiner(margin=1.0)
        N, D = 50, 16
        z = torch.randn(N, D)
        model_flags = torch.tensor([1] * 30 + [0] * 20, dtype=torch.int8)

        tp = torch.zeros(N, dtype=torch.long)
        tp[:10] = AnatomicalAligner.TIMEPOINTS_PLP.index("remission_I")
        tp[10:20] = AnatomicalAligner.TIMEPOINTS_PLP.index("peak_II")
        tp[30:] = AnatomicalAligner.TIMEPOINTS_MOG.index("chronic_long")

        a, p, n = miner.mine(
            z, tp, model_flags,
            AnatomicalAligner.TIMEPOINTS_PLP,
            AnatomicalAligner.TIMEPOINTS_MOG,
        )
        if len(a) > 0:
            assert len(a) == len(p) == len(n)
            assert (a >= 0).all()
            assert (p >= 0).all()
            assert (n >= 0).all()


# ---------------------------------------------------------------------------
# MechanisticAttributor
# ---------------------------------------------------------------------------

class TestMechanisticAttributor:

    def test_attribute_divergence_shape(self):
        from spatialflow.attributor import MechanisticAttributor
        from spatialflow.model import NicheMomentumEncoder
        from spatialflow.divergence import DivergenceScorer
        from spatialflow.attributor import DivergenceTarget

        N, G, D = 15, 8, 16
        gene_names = [f"gene_{i}" for i in range(G)]
        layout = {
            "gene_start": 0, "gene_end": G,
            "rlcs_start": G, "rlcs_end": G + 6,
            "ct_start": G + 6, "ct_end": G + 6 + 3,
            "tp_col": G + 6 + 3, "model_col": G + 6 + 3 + 1,
        }
        F = G + 6 + 3 + 2

        attributor = MechanisticAttributor(gene_names=gene_names, feature_layout=layout)
        encoder = NicheMomentumEncoder(F, hidden_dim=4, latent_dim=D,
                                       n_genes_reconstruct=G, heads=2, edge_dim=2)
        scorer = DivergenceScorer(latent_dim=D, knn_k=3, learn_correction=False)
        ei = torch.randint(0, N, (2, 20))
        ea = torch.randn(20, 2)
        vel = torch.randn(N, D)

        target_fn = DivergenceTarget(encoder, scorer, ei, ea, vel)
        x = torch.randn(N, F)

        out = attributor.attribute_divergence(target_fn, x)
        assert "gene_attributions" in out
        assert out["gene_attributions"].shape == (N, G)

    def test_top_relapse_genes_df(self):
        from spatialflow.attributor import MechanisticAttributor

        G = 10
        gene_names = [f"gene_{i}" for i in range(G)]
        attributor = MechanisticAttributor(
            gene_names=gene_names,
            feature_layout={"gene_start": 0, "gene_end": G,
                            "rlcs_start": G, "rlcs_end": G + 6,
                            "ct_start": G + 6, "ct_end": G + 6,
                            "tp_col": G + 6, "model_col": G + 7},
        )
        gene_attrs = torch.randn(5, G)
        df = attributor.top_relapse_genes(gene_attrs, n_top=5)
        assert "gene" in df.columns
        assert "attribution_score" in df.columns
        assert len(df) == 5


# ---------------------------------------------------------------------------
# VisualizationEngine
# ---------------------------------------------------------------------------

class TestVisualizationEngine:

    def test_divergence_heatmap_no_crash(self, tmp_path):
        try:
            from spatialflow.visualization import VisualizationEngine
        except ImportError:
            pytest.skip("matplotlib not installed")

        viz = VisualizationEngine(output_dir=str(tmp_path))
        xy = np.random.uniform(0, 100, (50, 2))
        div = np.random.randn(50)
        bif = div > 1.0

        fig = viz.plot_divergence_heatmap(xy, div, bif, save_name="test_div.pdf")
        # Function should not raise; file saved

    def test_relapse_probability_no_crash(self, tmp_path):
        try:
            from spatialflow.visualization import VisualizationEngine
        except ImportError:
            pytest.skip("matplotlib not installed")

        viz = VisualizationEngine(output_dir=str(tmp_path))
        xy = np.random.uniform(0, 100, (50, 2))
        risk = np.random.rand(50)
        viz.plot_relapse_probability(xy, risk, save_name="test_risk.pdf")


# ---------------------------------------------------------------------------
# End-to-end pipeline (smoke test)
# ---------------------------------------------------------------------------

class TestEAESpatialFlowPipeline:

    @pytest.fixture
    def minimal_section(self, tissue_masks, cell_data):
        tissue, cc, gm = tissue_masks
        xy, expr, genes, cell_types = cell_data
        return {
            "section_id": "MOG_baseline_s1",
            "model_type": "MOG",
            "timepoint": "baseline",
            "xy_coords": xy,
            "expr_matrix": expr,
            "gene_names": genes,
            "cell_types": cell_types,
            "tissue_mask": tissue,
            "cc_mask": cc,
            "gm_mask": gm,
        }

    def test_preprocess_section(self, minimal_section):
        from spatialflow import EAESpatialFlow

        config = {
            "resolution_um": 1.0, "mirror_bilateral": False,
            "spatial_radius_um": 20.0, "hidden_dim": 8,
            "latent_dim": 8, "n_genes_reconstruct": 10,
            "attn_heads": 2, "dropout": 0.0, "temporal_mode": "gru",
            "gru_hidden": 16, "projection_dim": 16,
        }
        pipeline = EAESpatialFlow(config=config, device="cpu")
        sec = pipeline.preprocess_section(minimal_section)
        assert "graph" in sec

    def test_train_smoke(self, minimal_section, tmp_path):
        from spatialflow import EAESpatialFlow

        config = {
            "resolution_um": 1.0, "mirror_bilateral": False,
            "spatial_radius_um": 15.0, "spatial_sigma_um": 8.0,
            "hidden_dim": 8, "latent_dim": 8, "n_genes_reconstruct": 10,
            "attn_heads": 2, "dropout": 0.0, "temporal_mode": "gru",
            "gru_hidden": 16, "projection_dim": 16,
            "triplet_margin": 1.0, "lambda_triplet": 1.0,
            "lambda_align": 0.1, "lambda_temporal": 0.1, "lambda_velocity": 0.0,
            "div_knn_k": 3, "bifurcation_threshold": 0.5,
            "velocity_smooth_sigma": 0.0, "learn_inf_axis": False,
            "output_dir": str(tmp_path),
        }
        pipeline = EAESpatialFlow(config=config, device="cpu",
                                   output_dir=str(tmp_path))

        history = pipeline.train([minimal_section], n_epochs=2,
                                  checkpoint_every=999)
        assert "total_loss" in history
        assert len(history["total_loss"]) == 2
        assert not np.isnan(history["total_loss"]).any()

    def test_predict_returns_results(self, minimal_section, tmp_path):
        from spatialflow import EAESpatialFlow

        config = {
            "resolution_um": 1.0, "mirror_bilateral": False,
            "spatial_radius_um": 15.0, "spatial_sigma_um": 8.0,
            "hidden_dim": 8, "latent_dim": 8, "n_genes_reconstruct": 10,
            "attn_heads": 2, "dropout": 0.0, "temporal_mode": "gru",
            "gru_hidden": 16, "projection_dim": 16,
            "triplet_margin": 1.0, "lambda_triplet": 1.0,
            "lambda_align": 0.1, "lambda_temporal": 0.1, "lambda_velocity": 0.0,
            "div_knn_k": 3, "bifurcation_threshold": 0.5,
            "velocity_smooth_sigma": 0.0, "learn_inf_axis": False,
            "output_dir": str(tmp_path),
        }
        pipeline = EAESpatialFlow(config=config, device="cpu",
                                   output_dir=str(tmp_path))
        pipeline.train([minimal_section], n_epochs=1, checkpoint_every=999)
        results = pipeline.predict([minimal_section])

        assert "MOG_baseline_s1" in results
        r = results["MOG_baseline_s1"]
        assert "z" in r
        assert "divergence" in r
        assert "relapse_risk" in r
        assert not np.isnan(r["z"]).any()
