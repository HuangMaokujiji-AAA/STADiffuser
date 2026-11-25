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
VALIDATION = (6, 34)
TRAIN = tuple(section for section in SECTIONS if section not in TEST + VALIDATION)
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
CHECKPOINT_INTERVAL = 10
VALIDATION_CELLS = 4096
EARLY_STOP_WARMUP = 10
EARLY_STOP_PATIENCE = {"autoencoder": 10, "denoiser": 12}
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


def write_protocol(output, protocol):
    temporary = output / "run.json.tmp"
    temporary.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "run.json")


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


def graph_data(matrix, rows, sections, xyz, coord, genes):
    import torch
    from torch_geometric.data import Data

    selected = np.concatenate([rows[section] for section in sections])
    x = np.empty((len(selected), len(genes)), dtype=np.float32)
    graphs = []
    offset = 0
    for section in sections:
        positions = rows[section]
        print(f"Loading section #{section} ({len(positions):,} cells)", flush=True)
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
    return Data(x=torch.from_numpy(x), edge_index=edges, spatial=torch.from_numpy(coord[selected]))


def encode_batch(model, batch):
    state = batch.x
    for layer in model.in_gat:
        state = layer(state, batch.edge_index)
    for layer in model.encoder:
        state = layer(state)
    return state[:batch.batch_size]


class StageValidation:
    """Validate at epoch boundaries and retain the minimum-loss model."""

    def __init__(self, output, stage, score):
        self.stage = stage
        self.score = score
        self.directory = output / "checkpoints"
        self.directory.mkdir(exist_ok=True)
        self.metrics = output / f"{stage}_validation.jsonl"
        self.best_path = self.directory / f"{stage}_best.pt"
        self.patience = EARLY_STOP_PATIENCE[stage]
        self.history = []
        self.best = float("inf")
        self.min_delta = None
        self.stale = 0

    def __call__(self, epoch, model):
        import torch

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                by_section = self.score(model)
        finally:
            model.train(was_training)
        loss = float(np.mean(list(by_section.values())))
        if not np.isfinite(loss):
            raise ValueError(f"Nonfinite {self.stage} validation loss at epoch {epoch}")
        self.history.append(loss)
        paths = []
        if epoch % CHECKPOINT_INTERVAL == 0:
            paths.append(self.directory / f"{self.stage}_epoch_{epoch:04d}.pt")
        if loss < self.best:
            self.best = loss
            self.best_epoch = epoch
            paths.append(self.best_path)
        for path in paths:
            temporary = path.with_suffix(".pt.tmp")
            torch.save({"epoch": epoch, "validation_mse": loss, "model": model.state_dict()}, temporary)
            temporary.replace(path)

        if len(self.history) == EARLY_STOP_WARMUP:
            self.anchor = self.best
            jitter = float(np.median(np.abs(np.diff(self.history, n=2)))) / 2
            self.min_delta = float(np.clip(jitter, 0.001 * self.anchor, 0.005 * self.anchor))
        elif self.min_delta is not None:
            if self.anchor - loss > self.min_delta:
                self.anchor = loss
                self.stale = 0
            else:
                self.stale += 1
        stop = self.stale >= self.patience
        record = dict(epoch=epoch, validation_mse=loss, by_section=by_section, best_epoch=self.best_epoch,
                      best_validation_mse=self.best, min_delta=self.min_delta, patience=self.patience,
                      stale_epochs=self.stale, early_stop=stop)
        with self.metrics.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"{self.stage} validation {json.dumps(record)}", flush=True)
        return stop

    def restore(self, model):
        import torch

        model.load_state_dict(torch.load(self.best_path, map_location="cpu")["model"])
        record = {"last_epoch": len(self.history), "best_epoch": self.best_epoch, "best_mse": self.best,
                  "patience": self.patience, "early_stop": self.stale >= self.patience}
        if self.min_delta is not None:
            record["min_delta"] = self.min_delta
        return record


def run_models(matrix, rows, xyz, coord, names, genes, var, protocol, log_base, output, device, smoke,
               *, prediction_protocol_key="openst_holdout", resume_predictions=False):
    import torch
    from diffusers import DDPMScheduler
    from torch_geometric.loader import NeighborLoader
    from stadiffuser import SpaAE, SpaUNet1DModel, pipeline
    from stadiffuser.utils import MinMaxNormalize

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model_rows = rows
    if smoke:
        # Use a compact patch per section to exercise real graph edges on CPU.
        model_rows = {}
        for section, positions in rows.items():
            xy = xyz[positions, :2]
            _, local = cKDTree(xy).query(xy[rng.integers(len(xy))], k=min(48, len(xy)))
            model_rows[section] = positions[np.sort(np.atleast_1d(local))]

    data = graph_data(matrix, model_rows, TRAIN, xyz, coord, genes)
    validation = {}
    for section in VALIDATION:
        graph = graph_data(matrix, model_rows, [section], xyz, coord, genes)
        seeds = np.sort(rng.choice(graph.num_nodes, min(VALIDATION_CELLS, graph.num_nodes), replace=False))
        validation[section] = NeighborLoader(graph, input_nodes=torch.from_numpy(seeds.astype(np.int64)),
                                             num_neighbors=NEIGHBORS, batch_size=AE_BATCH, is_sorted=True)
    del graph
    torch_device = torch.device(device)
    cuda_devices = [torch_device.index if torch_device.index is not None else torch.cuda.current_device()] \
        if torch_device.type == "cuda" else []

    def ae_score(model):
        scores = {}
        with torch.random.fork_rng(devices=cuda_devices):
            for section, loader in validation.items():
                torch.manual_seed(SEED + section)
                total, count = 0.0, 0
                for batch in loader:
                    batch = batch.to(device)
                    _, recon = model(batch.x, batch.edge_index)
                    n = batch.batch_size
                    total += (recon[:n] - batch.x[:n]).square().mean(dim=1).sum().item()
                    count += n
                scores[section] = total / count
        return scores

    autoencoder = SpaAE(input_dim=len(genes), gat_dim=[512, 32], block_out_dims=[32, 32]).to(device)
    ae_loader = NeighborLoader(data, num_neighbors=NEIGHBORS, batch_size=AE_BATCH, is_sorted=True)
    ae_validation = StageValidation(output, "autoencoder", ae_score)
    print("Training the spatial autoencoder", flush=True)
    pipeline.train_autoencoder(ae_loader, autoencoder, n_epochs=protocol["autoencoder_epochs"],
                               lr=LR, weight_decay=WEIGHT_DECAY, device=device, epoch_end=ae_validation)
    protocol["autoencoder_result"] = ae_validation.restore(autoencoder)
    write_protocol(output, protocol)

    autoencoder.eval()
    latent = np.empty((data.num_nodes, 32), dtype=np.float32)
    with torch.no_grad():
        for batch in tqdm(ae_loader, desc="Encoding training cells"):
            batch = batch.to(device)
            latent[batch.n_id[:batch.batch_size].cpu().numpy()] = encode_batch(autoencoder, batch).cpu().numpy()
    del ae_loader, batch
    normalizer = MinMaxNormalize(latent, dim=0)
    normalizer.max = np.maximum(normalizer.max, normalizer.min + 1e-8)
    data.x = torch.from_numpy(normalizer.normalize(latent).astype(np.float32))
    np.savez(output / "checkpoints/latent_normalizer.npz", minimum=normalizer.min, maximum=normalizer.max)
    del latent

    valid_noise = {}
    val_scheduler = DDPMScheduler(num_train_timesteps=TIMESTEPS)
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        for section, loader in validation.items():
            torch.manual_seed(SEED + section)
            z, coords = [], []
            for batch in loader:
                batch = batch.to(device)
                z.append(encode_batch(autoencoder, batch).cpu().numpy())
                coords.append(batch.spatial[:batch.batch_size])
            clean = torch.as_tensor(normalizer.normalize(np.concatenate(z)), dtype=torch.float32, device=device)
            clean = clean.unsqueeze(1)
            noise = torch.randn_like(clean)
            time = torch.randint(TIMESTEPS, (len(clean),), device=device)
            valid_noise[section] = val_scheduler.add_noise(clean, noise, time), time, torch.cat(coords), noise
    del validation, loader, batch

    def denoiser_score(model):
        scores = {}
        for section, (noisy, time, coords, noise) in valid_noise.items():
            total = 0.0
            for i in range(0, len(noise), DIFFUSION_BATCH):
                batch = slice(i, i + DIFFUSION_BATCH)
                pred = model(noisy[batch], time[batch], coords[batch]).sample
                total += (pred - noise[batch]).square().mean(dim=(1, 2)).sum().item()
            scores[section] = total / len(noise)
        return scores

    denoiser = SpaUNet1DModel(in_channels=18, out_channels=1, spatial_encoding="sinusoidal3d",
                              spatial3d_concat=True).to(device)
    diffusion_loader = NeighborLoader(data, num_neighbors=NEIGHBORS, batch_size=DIFFUSION_BATCH, is_sorted=True)
    diffusion_validation = StageValidation(output, "denoiser", denoiser_score)
    print("Training the 3D diffusion model", flush=True)
    pipeline.train_denoiser(diffusion_loader, denoiser, DDPMScheduler(num_train_timesteps=TIMESTEPS),
                            n_epochs=protocol["diffusion_epochs"], lr=LR, weight_decay=WEIGHT_DECAY,
                            device=device, epoch_end=diffusion_validation)
    protocol["denoiser_result"] = diffusion_validation.restore(denoiser)
    write_protocol(output, protocol)
    valid_noise.clear()
    del diffusion_loader, data
    denoiser.eval()
    if not resume_predictions:
        torch.manual_seed(SEED)

    for section in TEST:
        path = output / f"predicted_section_{section:02d}.h5ad"
        if resume_predictions:
            if path.is_file():
                continue
            torch.manual_seed(SEED)
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
        pred.uns[prediction_protocol_key] = protocol
        pred.uns["log1p"] = {"base": log_base}
        temporary = path.with_suffix(".tmp.h5ad")
        pred.write_h5ad(temporary, compression="lzf" if resume_predictions else "gzip")
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
            "split": ["test" if s in TEST else "validation" if s in VALIDATION else "train" for s in SECTIONS],
        }).to_csv(args.output_dir / "split.csv", index=False)
        protocol = {
            "train_sections": list(TRAIN), "validation_sections": list(VALIDATION), "test_sections": list(TEST),
            "prediction_coordinates": "observed aligned XYZ",
            "expression_scale": "source log1p X", "hvg_flavor": "seurat",
            "hvg_training_cells": sum(len(rows[s]) for s in TRAIN), "n_hvg": len(genes),
            "origin_um": origin.tolist(), "coordinate_step_um": steps.tolist(),
            "graph_radius_um": RADIUS, "neighbor_sampling": NEIGHBORS, "seed": SEED,
            "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "checkpoint_interval_epochs": CHECKPOINT_INTERVAL,
            "validation_cells_per_section": min(48, VALIDATION_CELLS) if args.smoke else VALIDATION_CELLS,
            "early_stop_warmup_epochs": EARLY_STOP_WARMUP,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_delta_fraction_bounds": [0.001, 0.005],
            "autoencoder_validation_metric": "reconstruction MSE on the original log1p scale",
            "denoiser_validation_metric": "fixed-timestep noise prediction MSE in the normalized latent space",
            "autoencoder_epochs": 1 if args.smoke else 500, "autoencoder_batch_size": AE_BATCH,
            "diffusion_epochs": 1 if args.smoke else 1000, "diffusion_batch_size": DIFFUSION_BATCH,
            "inference_steps": 3 if args.smoke else TIMESTEPS, "diffusion_timesteps": TIMESTEPS,
            "smoke": args.smoke, "device": args.device,
        }
        write_protocol(args.output_dir, protocol)
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
