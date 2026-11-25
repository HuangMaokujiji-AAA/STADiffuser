"""Run the previous STADiffuser holdout protocol on registered Zhuang sections."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch_geometric.data import Data

from experiments import openst_holdout as holdout
from .zhuang import SEED, SECTIONS, TEST, TRAIN, VALIDATION, Zhuang, atomic_json


class ResumeValidation(holdout.StageValidation):
    """Commit weights, validation state, optimizer, scheduler and RNG after every epoch."""

    def __init__(self, output, stage, score):
        super().__init__(output, stage, score)
        self.last = self.directory / f"{stage}_last.pt"
        self.canonical_best = self.best_path
        self.fields = ("history", "best", "best_epoch", "min_delta", "stale", "anchor")

    def resume(self, model, optimizer, scheduler):
        self.optimizer, self.scheduler = optimizer, scheduler
        if not self.last.is_file():
            return 0, False
        saved = torch.load(self.last, map_location="cpu")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        for key, value in saved["validation"].items():
            setattr(self, key, value)
        self.best_path = self.directory / saved["best_path"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(saved["cuda_rng"])
        print(f"Resuming {self.stage} after epoch {saved['epoch']}", flush=True)
        return saved["epoch"], self.stale >= self.patience

    def __call__(self, epoch, model):
        previous = self.best_path
        self.best_path = self.directory / f".{self.stage}_best_{epoch:04d}.pt"
        stopped = super().__call__(epoch, model)
        if self.best_epoch != epoch:
            self.best_path = previous
        state = dict(epoch=epoch, model=model.state_dict(), optimizer=self.optimizer.state_dict(),
                     scheduler=self.scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                     best_path=self.best_path.name, validation={key: getattr(self, key) for key in self.fields
                                                                if hasattr(self, key)})
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state()
        temp = self.last.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(self.last)
        if previous != self.best_path and previous != self.canonical_best:
            previous.unlink(missing_ok=True)
        return stopped

    def restore(self, model):
        import shutil
        result = super().restore(model)
        temp = self.canonical_best.with_suffix(".tmp")
        shutil.copyfile(self.best_path, temp)
        temp.replace(self.canonical_best)
        return result


def run(data, output, device, smoke):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
    output.mkdir(parents=True, exist_ok=True)
    holdout.SEED, holdout.TRAIN, holdout.VALIDATION, holdout.TEST = SEED, TRAIN, VALIDATION, TEST
    holdout.StageValidation = ResumeValidation
    meta = {s: data.metadata(s) for s in SECTIONS}
    xyz = np.concatenate([meta[s]["xyz"] for s in SECTIONS])
    names = pd.Index(np.concatenate([meta[s]["names"] for s in SECTIONS]))
    rows = {}
    offset = 0
    for section, info in meta.items():
        size = len(info["rows"])
        rows[section] = np.arange(offset, offset + size)
        offset += size
    origin = np.zeros(3)
    train_pos = data.split.loc[list(TRAIN), "idx"].to_numpy()
    train_depth = np.array([data.z[s] for s in TRAIN])
    steps = np.array([holdout.XY_STEP, holdout.XY_STEP, np.median(np.diff(train_depth) / np.diff(train_pos))])
    coord = holdout.quantize(xyz, origin, steps)

    def graph_data(matrix, active_rows, sections, xyz, coord, genes):
        sections_rows = [active_rows[s] for s in sections]
        expression = np.empty((sum(map(len, sections_rows)), len(genes)), dtype=np.float32)
        graphs = []
        offset = 0
        for section, positions in zip(sections, sections_rows):
            print(f"Loading section {section} ({len(positions):,} cells)", flush=True)
            local = positions - rows[section][0]
            expression[offset:offset + len(local)] = data.expression(section, meta[section]["rows"][local])
            offset += len(local)
            graphs.append(holdout.spatial_graph(xyz[positions, :2]))
        graph = sparse.block_diag(graphs, format="csr")
        dest = np.repeat(np.arange(graph.shape[0], dtype=graph.indices.dtype), np.diff(graph.indptr))
        edge_index = torch.from_numpy(np.vstack((graph.indices, dest))).long()
        return Data(x=torch.from_numpy(expression), edge_index=edge_index,
                    spatial=torch.from_numpy(coord[np.concatenate(sections_rows)]))

    holdout.graph_data = graph_data
    protocol = dict(data.protocol, model="STADiffuser", train_sections=list(TRAIN),
                    validation_sections=list(VALIDATION), test_sections=list(TEST),
                    prediction_coordinates="observed registered CCF",
                    origin_um=origin.tolist(), coordinate_step_um=steps.tolist(),
                    graph_radius_um=holdout.RADIUS, neighbor_sampling=holdout.NEIGHBORS,
                    learning_rate=holdout.LR, weight_decay=holdout.WEIGHT_DECAY,
                    checkpoint_interval_epochs=1, validation_cells_per_section=holdout.VALIDATION_CELLS,
                    early_stop_warmup_epochs=holdout.EARLY_STOP_WARMUP,
                    early_stop_patience=holdout.EARLY_STOP_PATIENCE,
                    early_stop_delta_fraction_bounds=[0.001, 0.005],
                    autoencoder_validation_metric="expression reconstruction MSE",
                    denoiser_validation_metric="fixed-timestep normalized-latent noise MSE",
                    autoencoder_epochs=1 if smoke else 500, autoencoder_batch_size=holdout.AE_BATCH,
                    diffusion_epochs=1 if smoke else 1000, diffusion_batch_size=holdout.DIFFUSION_BATCH,
                    inference_steps=3 if smoke else holdout.TIMESTEPS, diffusion_timesteps=holdout.TIMESTEPS,
                    smoke=smoke, device=device)
    run_path = output / "run.json"
    if run_path.is_file():
        saved = json.loads(run_path.read_text())
        if any(saved.get(key) != value for key, value in protocol.items()):
            raise ValueError("Existing STADiffuser run configuration differs")
        protocol.update(saved)
    else:
        atomic_json(run_path, protocol)
    holdout.run_models(None, rows, xyz, coord, names, np.arange(len(data.var)), data.var,
                       protocol, None, output, device, smoke,
                       prediction_protocol_key="zhuang_abca2", resume_predictions=True)
    atomic_json(output / "completed.json", dict(test_sections=list(TEST),
                                                autoencoder=protocol["autoencoder_result"],
                                                denoiser=protocol["denoiser_result"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(Zhuang(args.data, args.prepared), args.output, args.device, args.smoke)
