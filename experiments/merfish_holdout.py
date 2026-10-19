"""MERFISH conversion and independent leave-one-slice-out STADiffuser runs.

Conversion needs anndata/numpy/pandas/scipy only. Model imports are delayed until
run_fold(), so data preparation can run on a laptop without PyG or CUDA.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import rankdata


METRIC_PROTOCOL = "weekly_report_2026_10_18_v1"
EVALUATION_KEYS = {"cell_type_key", "rare_train_fraction", "coverage_radii_um", "xy_unit_um"}


def dense(x):
    return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float32)


def load_merfish(data_dir):
    """Preserve supplied expression/XY; append the numeric source slice_id as Z."""
    data_dir = Path(data_dir).expanduser()
    files = sorted(data_dir.glob("merfish_*.h5ad"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(files) < 3:
        raise FileNotFoundError(f"Expected >=3 merfish_*.h5ad files directly in {data_dir}")
    slices = []
    for path in files:
        a = ad.read_h5ad(path)
        if not a.var_names.is_unique or not a.obs_names.is_unique:
            raise ValueError(f"Nonunique gene/cell names in {path.name}")
        xy = np.asarray(a.obsm["spatial"], dtype=np.float64)
        if xy.shape != (a.n_obs, 2) or not np.isfinite(xy).all():
            raise ValueError(f"Expected finite N x 2 spatial coordinates: {path.name}")
        z_values = pd.to_numeric(a.obs["slice_id"].astype(str), errors="raise").unique()
        if len(z_values) != 1 or not np.isfinite(z_values).all():
            raise ValueError(f"Expected one numeric slice_id in {path.name}")
        x = dense(a.X)
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError(f"Expression must be finite and nonnegative: {path.name}")
        # Construct a clean object; discard stale graphs and unrelated source uns.
        obs = a.obs.copy()
        obs["source_obs_name"] = a.obs_names.astype(str)
        obs["slice_ID"] = path.stem
        obs["z"] = float(z_values[0])
        obs.index = pd.Index([f"{path.stem}::{name}" for name in a.obs_names])
        clean = ad.AnnData(x, obs=obs, var=a.var.copy())
        clean.obsm["spatial_xy"] = xy
        clean.obsm["spatial"] = np.column_stack((xy, np.full(a.n_obs, z_values[0])))
        slices.append(clean)
    slices.sort(key=lambda a: float(a.obs["z"].iloc[0]))
    z_values = [float(a.obs["z"].iloc[0]) for a in slices]
    if len(set(z_values)) != len(z_values):
        raise ValueError("Different source files have the same Z; check slice ordering")
    # Gene identities are panel metadata, not expression-based feature selection.
    genes = slices[0].var_names
    for a in slices[1:]:
        genes = genes[genes.isin(a.var_names)]
    if not len(genes):
        raise ValueError("No shared genes across slices")
    merged = ad.concat([a[:, genes].copy() for a in slices], merge="same", join="inner")
    merged.uns["merfish_conversion"] = {
        "expression": "source X unchanged; no normalization before splitting",
        "xy": "source obsm['spatial'] unchanged; common XY frame assumed",
        "z": "numeric obs['slice_id']; source units unspecified",
        "source_files": [p.name for p in files],
        "slice_order": [str(a.obs["slice_ID"].iloc[0]) for a in slices],
    }
    return merged


def slice_table(adata):
    return (adata.obs.groupby("slice_ID", observed=True)
            .agg(z=("z", "first"), n_cells=("z", "size"))
            .sort_values("z").reset_index())


def default_config(smoke=False):
    config = {
        "seed": 2024, "expression_transform": "log1p",
        "condition": "spatial_knn", "label_key": "annotation", "label_neighbors": 15,
        "xy_division": 20.0, "z_division": 0.01, "coordinate_margin": 2,
        "graph_neighbors": 8, "batch_size": 128, "sample_batch_size": 512,
        "pretrain_epochs": 200, "align_epochs": 300, "diffusion_epochs": 500,
        "align_update_interval": 50, "mnn_neighbors": 15,
        "inference_steps": 1000, "generation_seeds": [2024, 2025, 2026],
        "cell_type_key": "cell_class", "rare_train_fraction": 0.01,
        "coverage_radii_um": [25, 50, 100], "xy_unit_um": 1.0,
        "max_cells_per_slice": None,
        "device": "cuda:0", "smoke": bool(smoke),
    }
    if smoke:
        config.update(pretrain_epochs=1, align_epochs=1, diffusion_epochs=1,
                      inference_steps=10, generation_seeds=[2024],
                      max_cells_per_slice=48, batch_size=16, sample_batch_size=32)
    return config


def transform_expression(x, method):
    x = dense(x).copy()
    if method == "log1p":
        return np.log1p(x)
    if method == "none":
        return x
    raise ValueError("expression_transform must be 'log1p' or 'none'")


def build_spatial_graph(adata, neighbors=8):
    """Within-slice coordinate-only directed kNN, without cross-split edges."""
    edges = []
    for sid in adata.obs["slice_ID"].astype(str).unique():
        mask = adata.obs["slice_ID"].astype(str).to_numpy() == sid
        names = adata.obs_names.to_numpy()[mask]
        xy = np.asarray(adata.obsm["spatial_xy"])[mask]
        k = min(int(neighbors), len(names) - 1)
        if k < 1:
            raise ValueError(f"Slice {sid} needs at least two cells")
        distance, index = cKDTree(xy).query(xy, k=k + 1)
        src = np.repeat(names, k + 1)
        dst = names[index.ravel()]
        keep = src != dst
        edges.append(pd.DataFrame({"Cell1": src[keep], "Cell2": dst[keep],
                                   "Distance": distance.ravel()[keep], "EdgeType": "within"}))
    adata.uns["spatial_net"] = pd.concat(edges, ignore_index=True)
    return adata


def make_fold(adata, heldout, config):
    """Create training data and a zero-expression query with no reference labels."""
    table = slice_table(adata)
    order = table["slice_ID"].astype(str).tolist()
    if heldout not in order[1:-1]:
        raise ValueError(f"Hold out an interior slice, one of {order[1:-1]}")
    source = adata
    limit = config["max_cells_per_slice"]
    if limit is not None:
        rng = np.random.default_rng(config["seed"])
        keep = []
        for sid in order:
            positions = np.flatnonzero(adata.obs["slice_ID"].astype(str).to_numpy() == sid)
            keep.extend(np.sort(rng.choice(positions, min(int(limit), len(positions)), replace=False)))
        source = adata[np.asarray(keep)].copy()
    mask = source.obs["slice_ID"].astype(str).to_numpy() == heldout
    truth = source[mask].copy()  # Retained separately; never passed to a model.
    train = source[~mask].copy()
    train.X = transform_expression(train.X, config["expression_transform"])
    query_obs = truth.obs[["slice_ID", "z", "source_obs_name"]].copy()
    query = ad.AnnData(np.zeros(truth.shape, dtype=np.float32), obs=query_obs, var=train.var.copy())
    for key in ("spatial", "spatial_xy"):
        query.obsm[key] = np.asarray(truth.obsm[key]).copy()
    steps = np.array([config["xy_division"], config["xy_division"], config["z_division"]])
    if (steps <= 0).any():
        raise ValueError("Coordinate divisions must be positive")
    # One train-fitted origin for train AND query; never recenter each slice.
    origin = np.floor(train.obsm["spatial"].min(axis=0) / steps) * steps
    origin -= config["coordinate_margin"] * steps
    for a in (train, query):
        coords = np.rint((a.obsm["spatial"] - origin) / steps).astype(np.int64)
        if coords.min() < 0 or coords.max() >= 1000:
            raise ValueError("Coordinates exceed positional encoding [0, 999]; adjust divisions/margin")
        a.obsm["new_spatial"] = coords
        build_spatial_graph(a, config["graph_neighbors"])
    classes = []
    if config["condition"] == "spatial_knn":
        from sklearn.neighbors import KNeighborsClassifier
        key = config["label_key"]
        if key not in train.obs or train.obs[key].isna().any():
            raise ValueError(f"Missing training labels in {key}")
        labels = train.obs[key].astype(str).to_numpy()
        classes = sorted(np.unique(labels).tolist())
        mapping = {name: i for i, name in enumerate(classes)}
        train.obs["label_"] = np.array([mapping[name] for name in labels], dtype=np.int64)
        clf = KNeighborsClassifier(n_neighbors=min(config["label_neighbors"], train.n_obs), weights="distance")
        clf.fit((train.obsm["spatial"] - origin) / steps, labels)
        predicted = clf.predict((query.obsm["spatial"] - origin) / steps)
        query.obs["predicted_condition"] = predicted
        query.obs["label_"] = np.array([mapping[name] for name in predicted], dtype=np.int64)
    elif config["condition"] != "none":
        raise ValueError("condition must be 'spatial_knn' or 'none'; true held-out labels are not an input")
    assert not set(train.obs_names).intersection(query.obs_names)
    assert not np.any(query.X)
    assert config["label_key"] not in query.obs
    return train, query, truth, {"origin": origin.tolist(), "divisions": steps.tolist(), "classes": classes}


def _pearson_columns(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide((a * b).sum(axis=0), denom, out=np.full_like(denom, np.nan), where=denom > 1e-12)


def metric_expression(x, expression_scale):
    """Weekly-report scale: log1p(max(source X, 0)), without a second log."""
    values = dense(x)
    if not np.isfinite(values).all():
        raise ValueError("Expression contains NaN/Inf")
    values = np.maximum(values, 0)
    if expression_scale == "none":
        values = np.log1p(values)
    elif expression_scale != "log1p":
        raise ValueError("expression_scale must be 'log1p' or 'none'")
    return values.astype(np.float64)


def evaluate(truth_x, prediction, genes, cells, expression_scale="log1p"):
    """Primary MSE/MAE/PCC compare the mean log-expression gene vectors."""
    y, p = metric_expression(truth_x, expression_scale), metric_expression(prediction, expression_scale)
    if y.shape != p.shape or y.shape != (len(cells), len(genes)):
        raise ValueError("Prediction must be finite and match the reference shape")
    residual = p - y
    truth_mean, prediction_mean = y.mean(axis=0), p.mean(axis=0)
    mean_error = prediction_mean - truth_mean
    pcc = float(_pearson_columns(truth_mean[:, None], prediction_mean[:, None])[0])
    gene_corr = _pearson_columns(y, p)
    gene_rank = _pearson_columns(rankdata(y, axis=0), rankdata(p, axis=0))
    cell_corr = _pearson_columns(y.T, p.T)
    finite_mean = lambda x: float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")
    summary = {
        "mse": float(np.mean(mean_error ** 2)), "mae": float(np.mean(np.abs(mean_error))), "pcc": pcc,
        "pointwise_mse": float(np.mean(residual ** 2)), "pointwise_mae": float(np.mean(np.abs(residual))),
        "gene_pearson_mean": finite_mean(gene_corr), "gene_spearman_mean": finite_mean(gene_rank),
        "cell_pearson_mean": finite_mean(cell_corr), "valid_gene_pearson": int(np.isfinite(gene_corr).sum()),
        "valid_gene_spearman": int(np.isfinite(gene_rank).sum()),
        "valid_cell_pearson": int(np.isfinite(cell_corr).sum()),
    }
    by_gene = pd.DataFrame({"gene": genes, "truth_mean_log_expression": truth_mean,
                            "prediction_mean_log_expression": prediction_mean,
                            "squared_mean_error": mean_error ** 2, "absolute_mean_error": np.abs(mean_error),
                            "pearson": gene_corr, "spearman": gene_rank,
                            "rmse": np.sqrt(np.mean(residual ** 2, axis=0)),
                            "mae": np.mean(np.abs(residual), axis=0)})
    by_cell = pd.DataFrame({"cell": cells, "pearson": cell_corr,
                            "rmse": np.sqrt(np.mean(residual ** 2, axis=1))})
    return summary, by_gene, by_cell


def fit_cell_type_readout(train, config):
    """Annotate generated expression using training-only expression 1-NN."""
    from sklearn.neighbors import KNeighborsClassifier
    key = config["cell_type_key"]
    if key not in train.obs or train.obs[key].isna().any():
        raise ValueError(f"Missing training cell-type annotations: {key}")
    classifier = KNeighborsClassifier(n_neighbors=1, algorithm="brute", n_jobs=1)
    classifier.fit(metric_expression(train.X, config["expression_transform"]), train.obs[key].astype(str).to_numpy())
    return classifier


def _nearest_distances(query, reference):
    if len(query) == 0:
        return np.empty(0)
    if len(reference) == 0:
        return np.full(len(query), np.inf)
    return cKDTree(reference).query(query, k=1)[0]


def evaluate_cell_types(truth, prediction, train_counts, config):
    """Training-defined rare types; same-type XY coverage, as in the weekly report."""
    key, scale = config["cell_type_key"], config["expression_transform"]
    if not truth.var_names.equals(prediction.var_names) or not truth.n_obs or not prediction.n_obs:
        raise ValueError("Nonempty truth/prediction with identical gene order required")
    if not 0 < config["rare_train_fraction"] < 1 or config["xy_unit_um"] <= 0:
        raise ValueError("Invalid rare fraction or XY unit conversion")
    if (not {50, 100}.issubset(config["coverage_radii_um"]) or
            any(r <= 0 for r in config["coverage_radii_um"])):
        raise ValueError("Coverage radii must be positive and include 50 and 100 um")
    if truth.obs[key].isna().any() or prediction.obs[key].isna().any():
        raise ValueError("Cell-type labels must be nonmissing")
    tlabels, plabels = truth.obs[key].astype(str).to_numpy(), prediction.obs[key].astype(str).to_numpy()
    txy = np.asarray(truth.obsm["spatial_xy"], dtype=float) * config["xy_unit_um"]
    pxy = np.asarray(prediction.obsm["spatial_xy"], dtype=float) * config["xy_unit_um"]
    if not np.isfinite(txy).all() or not np.isfinite(pxy).all() or int(train_counts.sum()) <= 0:
        raise ValueError("Finite coordinates and nonempty training counts required")
    tg, pg = metric_expression(truth.X, scale), metric_expression(prediction.X, scale)
    classes = sorted(set(train_counts.index) | set(tlabels) | set(plabels))
    rows = []
    for label in classes:
        tm, pm = tlabels == label, plabels == label
        nt, npred, ntrain = int(tm.sum()), int(pm.sum()), int(train_counts.get(label, 0))
        fraction = ntrain / int(train_counts.sum())
        t2p, p2t = _nearest_distances(txy[tm], pxy[pm]), _nearest_distances(pxy[pm], txy[tm])
        row = dict(cell_class=label, train_n=ntrain, train_fraction=fraction,
                   is_rare_train=bool(ntrain > 0 and fraction < config["rare_train_fraction"]),
                   truth_n=nt, predicted_n=npred, truth_fraction=nt / truth.n_obs,
                   predicted_fraction=npred / prediction.n_obs,
                   fraction_error_pp=100 * (npred / prediction.n_obs - nt / truth.n_obs),
                   missing_in_prediction=bool(nt > 0 and npred == 0),
                   unseen_in_training=bool(nt > 0 and ntrain == 0),
                   truth_to_pred_nn_mean_um=float(t2p.mean()) if nt and npred else float("nan"),
                   pred_to_truth_nn_mean_um=float(p2t.mean()) if nt and npred else float("nan"),
                   mean_log_expression_pcc=float(_pearson_columns(tg[tm].mean(0)[:, None], pg[pm].mean(0)[:, None])[0])
                   if nt and npred else float("nan"))
        for radius in config["coverage_radii_um"]:
            suffix = f"{radius:g}um"
            row[f"coverage_recall_{suffix}"] = float(np.mean(t2p <= radius)) if nt else float("nan")
            row[f"coverage_precision_{suffix}"] = float(np.mean(p2t <= radius)) if npred else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    rare = table[table.is_rare_train & (table.truth_n > 0)]
    summary = dict(truth_n=truth.n_obs, predicted_n=prediction.n_obs,
                   count_error_pct=100 * (prediction.n_obs / truth.n_obs - 1),
                   truth_classes=len(set(tlabels)), predicted_classes=len(set(plabels)),
                   missing_classes=int(table.missing_in_prediction.sum()),
                   unseen_training_classes=int(table.unseen_in_training.sum()),
                   rare_classes_present_in_truth=len(rare), rare_missing_classes=int(rare.missing_in_prediction.sum()),
                   symmetric_mean_nn_um=float((_nearest_distances(txy, pxy).mean() + _nearest_distances(pxy, txy).mean()) / 2))
    for radius in config["coverage_radii_um"]:
        suffix = f"{radius:g}um"
        value = float(rare[f"coverage_recall_{suffix}"].mean())
        summary[f"rare_recall_{suffix}"] = value
        summary[f"rare_macro_coverage_recall_{suffix}"] = value
        summary[f"rare_macro_coverage_precision_{suffix}"] = float(rare[f"coverage_precision_{suffix}"].mean())
    return summary, table


def _evaluation_identity(config):
    return {"metric_protocol": METRIC_PROTOCOL, "cell_type_readout": "training_expression_1nn",
            **{key: config[key] for key in EVALUATION_KEYS}}


def _training_config(config):
    # The removed interpolation setting never affected model fitting/generation.
    return {k: v for k, v in config.items() if k not in EVALUATION_KEYS | {"baseline_neighbors"}}


def score_prediction(train, query, truth, values, config, classifier, output, seed):
    """No reference labels are attached until after the generated values exist."""
    summary, by_gene, by_cell = evaluate(transform_expression(truth.X, config["expression_transform"]), values,
                                         truth.var_names, truth.obs_names, config["expression_transform"])
    result = query.copy()
    result.X = np.asarray(values, dtype=np.float32)
    result.obs[config["cell_type_key"]] = classifier.predict(metric_expression(values, config["expression_transform"]))
    result.obs["cell_type_source"] = "training_expression_1nn"
    # evaluate_cell_types expects truth and prediction in the same model scale.
    reference = truth.copy()
    reference.X = transform_expression(truth.X, config["expression_transform"])
    counts = train.obs[config["cell_type_key"]].astype(str).value_counts()
    spatial, by_type = evaluate_cell_types(reference, result, counts, config)
    summary.update(spatial)
    development = (config["smoke"] or config["max_cells_per_slice"] is not None or
                   config["pretrain_epochs"] < 200 or config["align_epochs"] < 300 or
                   config["diffusion_epochs"] < 500 or config["inference_steps"] < 1000)
    summary.update(heldout=str(query.obs["slice_ID"].iloc[0]),
                   method="STADiffuser_predicted_condition" if "label_" in query.obs else "STADiffuser_spatial_only",
                   seed=seed, run_kind="development" if development else "full_budget",
                   expression_scale="log1p_nonnegative_source_X", metric_protocol=METRIC_PROTOCOL,
                   cell_type_readout="training_expression_1nn", query_geometry_fixed=True,
                   negative_prediction_fraction=float(np.mean(np.asarray(values) < 0)),
                   n_cells=truth.n_obs, n_genes=truth.n_vars)
    if seed == "mean":
        by_gene.to_csv(output / "gene_metrics.csv", index=False)
        by_cell.to_csv(output / "cell_metrics.csv", index=False)
        by_type.to_csv(output / "cell_type_metrics.csv", index=False)
        by_type[by_type.is_rare_train | by_type.unseen_in_training].to_csv(output / "rare_cell_metrics.csv", index=False)
        counts.rename_axis("cell_class").rename("train_n").to_csv(output / "train_cell_counts.csv")
        result.uns.clear()
        result.uns["evaluation"] = {"expression_scale": config["expression_transform"],
                                    "metric_protocol": METRIC_PROTOCOL, "cell_type_source": "training_expression_1nn"}
        result.write_h5ad(output / "prediction.h5ad", compression="gzip")
        reference.uns.clear()
        reference.uns["evaluation"] = {"expression_scale": config["expression_transform"]}
        reference.write_h5ad(output / "truth.h5ad", compression="gzip")
    return summary


def reevaluate_saved_fold(adata, heldout, config, output_root):
    """Recompute weekly metrics from a saved mean prediction; no GPU/retraining."""
    output = Path(output_root) / heldout
    saved = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    if saved["dataset_sha256"] != dataset_digest(adata) or _training_config(saved["config"]) != _training_config(config):
        raise ValueError("Saved prediction was produced with different input data/training configuration")
    prediction = ad.read_h5ad(output / "prediction.h5ad")
    train, query, truth, _ = make_fold(adata, heldout, config)
    if not prediction.obs_names.equals(query.obs_names) or not prediction.var_names.equals(query.var_names):
        raise ValueError("Saved prediction gene/cell order differs from the requested fold")
    np.testing.assert_array_equal(prediction.obsm["spatial_xy"], query.obsm["spatial_xy"])
    classifier = fit_cell_type_readout(train, config)
    row = score_prediction(train, query, truth, dense(prediction.X), config, classifier, output, "mean")
    row["generation_metrics_available"] = "saved_mean_prediction_only"
    frame = pd.DataFrame([row])
    _write_json(output / "evaluation_config.json", _evaluation_identity(config))
    frame.to_csv(output / "metrics.csv", index=False)
    return frame


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_digest(adata):
    """Bind completed folds to the input matrices, geometry, and annotations."""
    digest = hashlib.sha256()
    for x in (dense(adata.X), np.asarray(adata.obsm["spatial"], dtype=np.float64)):
        digest.update(np.ascontiguousarray(x).tobytes())
    for names in (adata.obs_names, adata.var_names):
        digest.update(json.dumps(names.tolist(), ensure_ascii=False).encode("utf-8"))
    digest.update(adata.obs.astype(str).to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def _generate(denoiser, autoencoder, query, normalizer, config, seed):
    import torch
    from diffusers import DDPMScheduler
    from stadiffuser import pipeline
    device = config["device"]
    denoiser.eval()
    autoencoder.eval()
    latent = []
    for start in range(0, query.n_obs, config["sample_batch_size"]):
        stop = min(start + config["sample_batch_size"], query.n_obs)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(config["inference_steps"])
        labels = query.obs["label_"].to_numpy()[start:stop] if "label_" in query.obs else None
        z = pipeline.simulate(denoiser=denoiser, spatial_coord=query.obsm["new_spatial"][start:stop],
                              labels=labels, noise_scheduler=scheduler, seed=seed + start,
                              progress=False, device=device)
        latent.append(z.detach().cpu().numpy())
        print(f"Generate seed={seed}, cells={stop}/{query.n_obs}", flush=True)
    latent = normalizer.denormalize(np.concatenate(latent)).astype(np.float32)
    # Decode the complete query graph once. Query expression never enters decode.
    edge_index = torch.as_tensor(query.uns["edge_list"], dtype=torch.long, device=device)
    with torch.no_grad():
        prediction = autoencoder.decode(torch.as_tensor(latent, device=device), edge_index).cpu().numpy()
    return np.asarray(prediction, dtype=np.float32)


def run_fold(adata, heldout, config, output_root):
    """Train a fresh T4 architecture on the other slices, generate and then score."""
    output = Path(output_root) / heldout
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "heldout": heldout, "dataset_sha256": dataset_digest(adata)}
    identity_path = output / "run_config.json"
    if identity_path.exists():
        saved = json.loads(identity_path.read_text(encoding="utf-8"))
        if (saved["dataset_sha256"] != identity["dataset_sha256"] or saved["heldout"] != heldout or
                _training_config(saved["config"]) != _training_config(config)):
            raise ValueError(f"Existing run has different data/configuration: {output}. Change RUN_NAME.")
        if (output / "prediction.h5ad").exists():
            evaluation_path = output / "evaluation_config.json"
            if ((output / "metrics.csv").exists() and evaluation_path.exists() and
                    json.loads(evaluation_path.read_text(encoding="utf-8")) == _evaluation_identity(config)):
                print(f"Reuse completed fold: {heldout}", flush=True)
                return pd.read_csv(output / "metrics.csv")
            print(f"Update weekly-report metrics without retraining: {heldout}", flush=True)
            return reevaluate_saved_fold(adata, heldout, config, output_root)
        raise FileExistsError(f"Incomplete run exists at {output}; choose a new RUN_NAME to preserve its checkpoints")

    import torch
    from diffusers import DDPMScheduler
    from torch_geometric.loader import NeighborLoader
    from stadiffuser import pipeline, utils
    from stadiffuser.dataset import get_slice_loader
    from stadiffuser.models import SpaUNet1DModel
    from stadiffuser.vae import SpaAE

    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select the server environment or set device='cpu' for a smoke check")
    for key in ("pretrain_epochs", "align_epochs", "diffusion_epochs", "inference_steps", "sample_batch_size"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if config["inference_steps"] > 1000 or not config["generation_seeds"]:
        raise ValueError("Use 1..1000 inference steps and at least one generation seed")
    _write_json(identity_path, identity)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    train, query, truth, coordinate_info = make_fold(adata, heldout, config)
    classifier = fit_cell_type_readout(train, config)
    batch_list = slice_table(train)["slice_ID"].astype(str).tolist()
    if train.obs.groupby("slice_ID", observed=True).size().min() < config["mnn_neighbors"]:
        raise ValueError("Each training slice must contain at least mnn_neighbors cells")
    coordinate_info.update(training_slices=batch_list, heldout_slice=heldout,
                           train_cells=train.n_obs, query_cells=query.n_obs,
                           expression_transform=config["expression_transform"],
                           heldout_expression_used_for_training=False,
                           heldout_annotations_used_for_conditioning=False)
    _write_json(output / "split.json", coordinate_info)
    train.obs_names.to_series().to_csv(output / "train_cells.csv", index=False, header=["cell"])
    query.obs_names.to_series().to_csv(output / "query_cells.csv", index=False, header=["cell"])
    versions = {}
    for name in ("torch", "torch-geometric", "diffusers", "numpy", "scanpy", "anndata", "scipy"):
        versions[name] = importlib.metadata.version(name)
    versions["device"] = torch.cuda.get_device_name(config["device"]) if config["device"].startswith("cuda") else "cpu"
    _write_json(output / "environment.json", versions)

    print(f"Hold out {heldout}; train={batch_list}; cells={train.n_obs}/{query.n_obs}", flush=True)
    data = pipeline.prepare_dataset(train)
    autoencoder = SpaAE(input_dim=train.n_vars, block_list=["AttnBlock"], gat_dim=[512, 32], block_out_dims=[32, 32])
    loaders = [get_slice_loader(train, data, sid, use_batch="slice_ID", batch_size=config["batch_size"])
               for sid in batch_list]
    autoencoder, pre_loss = pipeline.pretrain_autoencoder_multi(
        loaders, autoencoder, pretrain_epochs=config["pretrain_epochs"], device=config["device"])
    torch.save({"state_dict": autoencoder.state_dict(), "input_dim": train.n_vars, "stage": "pretrain"}, output / "autoencoder.pt")
    autoencoder, align_loss = pipeline.train_autoencoder_multi(
        train, autoencoder, use_batch="slice_ID", batch_list=batch_list,
        n_epochs=config["align_epochs"], margin=1, lr=1e-4,
        update_interval=config["align_update_interval"], mnn_neighbors=config["mnn_neighbors"], device=config["device"])
    torch.save({"state_dict": autoencoder.state_dict(), "input_dim": train.n_vars, "stage": "aligned"}, output / "autoencoder.pt")
    recon = pipeline.get_recon(train, autoencoder, device=config["device"], apply_normalize=False,
                               batch_mode=True, batch_size=config["batch_size"], show_progress=False)
    normalizer = utils.MinMaxNormalize(recon.obsm["latent"], dim=0)
    normalizer.max = np.maximum(normalizer.max, normalizer.min + 1e-8)
    recon.obsm["normalized_latent"] = normalizer.normalize(recon.obsm["latent"]).astype(np.float32)
    np.savez_compressed(output / "latent_normalizer.npz", minimum=normalizer.min, maximum=normalizer.max)
    n_classes = len(coordinate_info["classes"]) or None
    denoiser = SpaUNet1DModel(in_channels=18, out_channels=1, spatial_encoding="sinusoidal3d",
                              spatial3d_concat=True, num_class_embeds=n_classes).to(config["device"])
    latent_data = pipeline.prepare_dataset(recon, use_rep="normalized_latent", use_spatial="new_spatial",
                                           use_label="label_" if n_classes else None)
    loader = NeighborLoader(latent_data, num_neighbors=[5, 3], batch_size=config["batch_size"])
    denoiser, diffusion_loss = pipeline.train_denoiser(
        loader, denoiser, DDPMScheduler(num_train_timesteps=1000), lr=1e-4, weight_decay=1e-6,
        n_epochs=config["diffusion_epochs"], num_class_embeds=n_classes, device=config["device"])
    torch.save({"state_dict": denoiser.state_dict(), "num_class_embeds": n_classes}, output / "denoiser.pt")
    np.savez_compressed(output / "losses.npz", pretrain=pre_loss, align=align_loss, diffusion=diffusion_loss)
    del recon, latent_data, loader, loaders, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pipeline.prepare_dataset(query, use_spatial="new_spatial")
    rows = []
    prediction_sum = np.zeros(query.shape, dtype=np.float64)
    for seed in config["generation_seeds"]:
        prediction = _generate(denoiser, autoencoder, query, normalizer, config, int(seed))
        rows.append(score_prediction(train, query, truth, prediction, config, classifier, output, int(seed)))
        prediction_sum += prediction
    prediction_mean = np.asarray(prediction_sum / len(config["generation_seeds"]), dtype=np.float32)
    rows.append(score_prediction(train, query, truth, prediction_mean, config, classifier, output, "mean"))
    if n_classes:
        label_key = config["label_key"]
        acc = float(np.mean(query.obs["predicted_condition"].astype(str).to_numpy() == truth.obs[label_key].astype(str).to_numpy()))
        _write_json(output / "condition_evaluation.json", {"label_key": label_key, "accuracy": acc,
                                                          "source": "training-coordinate kNN; truth used for evaluation only"})
    for row in rows:
        row["generation_metrics_available"] = "individual_seeds_and_mean"
    frame = pd.DataFrame(rows)
    _write_json(output / "evaluation_config.json", _evaluation_identity(config))
    frame.to_csv(output / "metrics.csv", index=False)  # Completion marker, written last.
    del autoencoder, denoiser
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame


def main():
    parser = argparse.ArgumentParser(description="Merge MERFISH slices into a portable 3D h5ad; expression is unchanged")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="data/merfish_mouse_hypothalamus_3d.h5ad")
    args = parser.parse_args()
    merged = load_merfish(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(output, compression="gzip")
    print(slice_table(merged).to_string(index=False))
    print(f"Saved {merged.shape}: {output.resolve()}")


if __name__ == "__main__":
    main()
