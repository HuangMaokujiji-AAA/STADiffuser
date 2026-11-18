"""Train STADiffuser once and predict four jointly held-out OpenST-3D sections."""

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

try:
    from anndata.io import read_elem, sparse_dataset
except ImportError:  # AnnData 0.10, used by the original environment.
    from anndata.experimental import read_elem, sparse_dataset


SECTIONS = (2, 3, 4, 5, 6, 7, 9, 11, 17, 18, 19, 23, 24, 25, 26, 28, 33, 34, 36)
TEST = (4, 9, 18, 24)
TRAIN = tuple(section for section in SECTIONS if section not in TEST)
N_HVG = 2000
RADIUS = 150
XY_STEP = 125
NEIGHBORS = [5, 3]
AE_BATCH = 256
DIFFUSION_BATCH = 512
TIMESTEPS = 1000
SEED = 2024
LR = 1e-4
WEIGHT_DECAY = 1e-6
DEFAULT_INPUT = Path("data/Open-ST/GSE251926_metastatic_lymph_node_3d.h5ad")


def read_rows(matrix, rows):
    first, last = int(rows[0]), int(rows[-1])
    return matrix[first:last + 1] if last - first + 1 == len(rows) else matrix[rows]


def select_hvgs(matrix, rows_by_section, var, log_base, n_top=N_HVG):
    """Scanpy Seurat dispersion ranking using pooled, streamed training moments."""
    sums, squares = np.zeros((2, len(var)), dtype=np.float64)
    n_cells = 0
    for section in TRAIN:
        rows = rows_by_section[section]
        print(f"HVG: reading training section #{section} ({len(rows):,} cells)", flush=True)
        for start in range(0, len(rows), 4096):
            x = read_rows(matrix, rows[start:start + 4096]).tocsr()
            x.sum_duplicates()
            if log_base is not None:
                x.data *= np.log(log_base)
            np.expm1(x.data, out=x.data)
            values = x.data.astype(np.float64)
            sums += np.bincount(x.indices, weights=values, minlength=len(var))
            squares += np.bincount(x.indices, weights=values ** 2, minlength=len(var))
        n_cells += len(rows)

    means = sums / n_cells
    variance = np.maximum(squares / n_cells - means ** 2, 0) * n_cells / (n_cells - 1)
    means[means == 0] = 1e-12
    dispersion = variance / means
    dispersion[dispersion == 0] = np.nan
    dispersions = pd.Series(np.log(dispersion), index=var.index)
    bins = pd.cut(np.log1p(means), 20)
    grouped = dispersions.groupby(bins, observed=True)
    center, spread = grouped.transform("mean"), grouped.transform("std")
    normalized = (dispersions - center.where(spread.notna(), 0)) / spread.fillna(center)
    scores = normalized.to_numpy()
    eligible = np.flatnonzero(np.isfinite(scores))
    if len(eligible) < n_top:
        raise ValueError(f"Only {len(eligible)} genes have finite dispersion scores; need {n_top}")
    ranked = eligible[np.argsort(-scores[eligible], kind="stable")[:n_top]]
    indices = np.sort(ranked)
    var = var.iloc[indices].copy()
    var["highly_variable"] = True
    var["means"] = np.log1p(means[indices])
    var["dispersions"] = dispersions.iloc[indices].to_numpy()
    var["dispersions_norm"] = scores[indices]
    var["hvg_rank"] = np.argsort(np.argsort(ranked)) + 1
    return indices, var


def spatial_graph(xy):
    """CSR rows are destinations, columns are sources. Include one self-edge per cell."""
    tree = cKDTree(xy)
    pairs = tree.sparse_distance_matrix(tree, RADIUS, output_type="coo_matrix")
    return sparse.csr_matrix((np.ones(pairs.nnz, dtype=bool), (pairs.row, pairs.col)), shape=pairs.shape)


def quantize(xyz, origin, steps):
    coord = np.rint((xyz - origin) / steps).astype(np.int64)
    if coord.min() < 0 or coord.max() >= 1000:
        raise ValueError("Coordinates are outside the model's positional encoding range [0, 999]")
    return coord


def decode_section(autoencoder, latent, graph, device):
    """Apply the complete section decoder with bounded attention and GAT memory."""
    import torch

    autoencoder.eval()
    with torch.no_grad():
        state = torch.as_tensor(latent, device=device)
        for layer in autoencoder.decoder:
            state = layer(state, query_batch_size=DIFFUSION_BATCH)
        for layer in autoencoder.out_gat:
            state = state.to(device)
            result = torch.empty((graph.shape[0], layer.out_channels), dtype=state.dtype)
            for start in range(0, graph.shape[0], AE_BATCH):
                stop = min(start + AE_BATCH, graph.shape[0])
                edges = graph[start:stop].tocoo()
                sources, local_source = np.unique(edges.col, return_inverse=True)
                edge_index = torch.as_tensor(
                    np.vstack((local_source, edges.row)), dtype=torch.long, device=device
                )
                features = state[torch.as_tensor(sources, dtype=torch.long, device=device)]
                result[start:stop] = layer((features, state[start:stop]), edge_index).cpu()
            state = result
    return state.numpy()


def run_models(matrix, rows, xyz, coord, names, genes, var, protocol, log_base, output, device, smoke):
    import torch
    from diffusers import DDPMScheduler
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from stadiffuser import SpaAE, SpaUNet1DModel, pipeline
    from stadiffuser.utils import MinMaxNormalize

    torch.manual_seed(SEED)
    model_rows = rows
    if smoke:
        # Use a compact patch per section to exercise real graph edges on CPU.
        rng = np.random.default_rng(SEED)
        model_rows = {}
        for section, positions in rows.items():
            xy = xyz[positions, :2]
            _, local = cKDTree(xy).query(xy[rng.integers(len(xy))], k=min(48, len(xy)))
            model_rows[section] = positions[np.sort(np.atleast_1d(local))]

    train_rows = np.concatenate([model_rows[section] for section in TRAIN])
    x = np.empty((len(train_rows), len(genes)), dtype=np.float32)
    graphs = []
    offset = 0
    for section in TRAIN:
        positions = model_rows[section]
        print(f"Loading training section #{section} ({len(positions):,} cells)", flush=True)
        for start in range(0, len(positions), 4096):
            chunk = positions[start:start + 4096]
            read_rows(matrix, chunk)[:, genes].astype(np.float32, copy=False).toarray(
                out=x[offset:offset + len(chunk)]
            )
            offset += len(chunk)
        graphs.append(spatial_graph(xyz[positions, :2]))
    graph = sparse.block_diag(graphs, format="csr")
    dest = np.repeat(np.arange(graph.shape[0], dtype=graph.indices.dtype), np.diff(graph.indptr))
    edges = torch.from_numpy(np.vstack((graph.indices, dest))).long()
    del graph, dest, graphs
    data = Data(x=torch.from_numpy(x), edge_index=edges, spatial=torch.from_numpy(coord[train_rows]))
    del x, edges

    autoencoder = SpaAE(input_dim=len(genes), gat_dim=[512, 32], block_out_dims=[32, 32]).to(device)
    ae_loader = NeighborLoader(data, num_neighbors=NEIGHBORS, batch_size=AE_BATCH, is_sorted=True)
    print("Training the spatial autoencoder", flush=True)
    pipeline.train_autoencoder(ae_loader, autoencoder, n_epochs=protocol["autoencoder_epochs"],
                               lr=LR, weight_decay=WEIGHT_DECAY, device=device)

    autoencoder.eval()
    latent = np.empty((len(train_rows), 32), dtype=np.float32)
    with torch.no_grad():
        for batch in tqdm(ae_loader, desc="Encoding training cells"):
            batch = batch.to(device)
            state = batch.x
            for layer in autoencoder.in_gat:
                state = layer(state, batch.edge_index)
            for layer in autoencoder.encoder:
                state = layer(state)
            latent[batch.n_id[:batch.batch_size].cpu().numpy()] = state[:batch.batch_size].cpu().numpy()
    del ae_loader, batch
    normalizer = MinMaxNormalize(latent, dim=0)
    normalizer.max = np.maximum(normalizer.max, normalizer.min + 1e-8)
    data.x = torch.from_numpy(normalizer.normalize(latent).astype(np.float32))
    del latent

    denoiser = SpaUNet1DModel(in_channels=18, out_channels=1, spatial_encoding="sinusoidal3d",
                              spatial3d_concat=True).to(device)
    diffusion_loader = NeighborLoader(data, num_neighbors=NEIGHBORS, batch_size=DIFFUSION_BATCH, is_sorted=True)
    print("Training the 3D diffusion model", flush=True)
    pipeline.train_denoiser(diffusion_loader, denoiser, DDPMScheduler(num_train_timesteps=TIMESTEPS),
                            n_epochs=protocol["diffusion_epochs"], lr=LR, weight_decay=WEIGHT_DECAY,
                            device=device)
    del diffusion_loader, data
    denoiser.eval()
    torch.manual_seed(SEED)

    for section in TEST:
        positions = model_rows[section]
        z = np.empty((len(positions), denoiser.sample_size), dtype=np.float32)
        batches = range(0, len(positions), DIFFUSION_BATCH)
        for start in tqdm(batches, desc=f"Sampling section #{section}"):
            stop = min(start + DIFFUSION_BATCH, len(positions))
            scheduler = DDPMScheduler(num_train_timesteps=TIMESTEPS)
            scheduler.set_timesteps(protocol["inference_steps"])
            z[start:stop] = pipeline.simulate(
                denoiser=denoiser, spatial_coord=coord[positions[start:stop]],
                noise_scheduler=scheduler, normarlizer=normalizer, progress=False, device=device,
            ).cpu().numpy()
        section_xyz = xyz[positions]
        values = decode_section(autoencoder, z, spatial_graph(section_xyz[:, :2]), device)
        for start in range(0, len(values), 4096):
            if not np.isfinite(values[start:start + 4096]).all():
                raise ValueError(f"Nonfinite predicted expression for section #{section}")
        pred = ad.AnnData(X=values, obs=pd.DataFrame({"n_section": section}, index=names[positions]), var=var)
        pred.obsm["spatial"] = section_xyz[:, :2]
        pred.obsm["spatial_3d_aligned"] = section_xyz
        pred.obsm["latent"] = z
        pred.uns["openst_holdout"] = protocol
        pred.uns["log1p"] = {"base": log_base}
        path = output / f"predicted_section_{section:02d}.h5ad"
        temporary = path.with_suffix(".tmp.h5ad")
        pred.write_h5ad(temporary, compression="gzip")
        temporary.replace(path)
        print(f"Saved {path} ({pred.n_obs:,} cells x {pred.n_vars:,} genes)", flush=True)
        del pred, values


def run(args):
    if not args.prepare_only:
        import torch

        if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a GPU environment, or --device cpu --smoke for a short check.")

    with h5py.File(args.input, "r") as handle:
        obs = handle["obs"]
        ids = read_elem(obs["n_section"]).astype(int)
        if set(ids) != set(SECTIONS):
            raise ValueError(f"Expected physical section IDs {SECTIONS}; found {np.unique(ids)}")
        names = pd.Index(read_elem(obs[obs.attrs["_index"]]))
        var_node = handle["var"]
        var = pd.DataFrame(index=pd.Index(read_elem(var_node[var_node.attrs["_index"]])))
        if not names.is_unique or not var.index.is_unique:
            raise ValueError("Unique cell and gene identifiers are required")
        rows = {section: np.flatnonzero(ids == section) for section in SECTIONS}
        xyz = read_elem(handle["obsm/spatial_3d_aligned"])
        if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
            raise ValueError("Expected finite registered XYZ in obsm['spatial_3d_aligned']")
        heights = np.array([xyz[rows[section][0], 2] for section in SECTIONS])
        if not np.allclose(xyz[:, 2], heights[np.searchsorted(SECTIONS, ids)], rtol=0, atol=1e-6):
            raise ValueError("Every physical section must have one physical Z")
        if (np.diff(heights) <= 0).any():
            raise ValueError("Physical Z must increase with section number")
        train_xyz = xyz[np.isin(ids, TRAIN)]
        origin = train_xyz.min(axis=0)
        train_z = heights[np.isin(SECTIONS, TRAIN)]
        steps = np.array([XY_STEP, XY_STEP, np.median(np.diff(train_z) / np.diff(TRAIN))])
        coord = quantize(xyz, origin, steps)
        del train_xyz

        log_base = read_elem(handle["uns/log1p"]).get("base")
        matrix = sparse_dataset(handle["X"])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        genes, var = select_hvgs(matrix, rows, var, log_base)
        var.to_csv(args.output_dir / "hvg.csv", index_label="gene")
        pd.DataFrame({
            "n_section": SECTIONS, "z_um": heights, "n_cells": [len(rows[s]) for s in SECTIONS],
            "split": ["test" if s in TEST else "train" for s in SECTIONS],
        }).to_csv(args.output_dir / "split.csv", index=False)
        protocol = {
            "train_sections": list(TRAIN), "test_sections": list(TEST),
            "prediction_coordinates": "observed aligned XYZ",
            "expression_scale": "source log1p X", "hvg_flavor": "seurat",
            "hvg_training_cells": sum(len(rows[s]) for s in TRAIN), "n_hvg": len(genes),
            "origin_um": origin.tolist(), "coordinate_step_um": steps.tolist(),
            "graph_radius_um": RADIUS, "neighbor_sampling": NEIGHBORS, "seed": SEED,
            "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "autoencoder_epochs": 1 if args.smoke else 500, "autoencoder_batch_size": AE_BATCH,
            "diffusion_epochs": 1 if args.smoke else 1000, "diffusion_batch_size": DIFFUSION_BATCH,
            "inference_steps": 3 if args.smoke else TIMESTEPS, "diffusion_timesteps": TIMESTEPS,
            "smoke": args.smoke, "device": args.device,
        }
        (args.output_dir / "run.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        print(f"Selected {len(genes):,} HVGs from {protocol['hvg_training_cells']:,} training cells", flush=True)
        if not args.prepare_only:
            run_models(matrix, rows, xyz, coord, names, genes, var, protocol, log_base,
                       args.output_dir, args.device, args.smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the run and predictions")
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Check the split and write the training HVG panel")
    mode.add_argument("--smoke", action="store_true", help="Run one epoch per stage on 48 cells per section")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
