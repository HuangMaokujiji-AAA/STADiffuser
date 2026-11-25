"""Common CCF coordinates, split, and expression for Zhuang-ABCA-2."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SEED = 2025
SECTIONS = tuple(i for i in range(4, 62) if i not in (24, 29, 38, 43))
TEST_INDEX = (2, 4, 6, 8, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32, 36, 38, 40, 42, 44, 48, 50, 52)
TRAIN = (SECTIONS[0], *SECTIONS[1::2])
VALIDATION = tuple(SECTIONS[i] for i in (10, 22, 34, 46))
TEST = tuple(SECTIONS[i] for i in TEST_INDEX)
EXPRESSION = "natural log1p of raw counts, all 1122 measured genes"
assert len(SECTIONS) == 54 and VALIDATION == (14, 27, 41, 54)
assert len(set(TRAIN + VALIDATION + TEST)) == 54 and len(TRAIN + VALIDATION + TEST) == 54


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_h5ad(adata, path):
    temp = path.with_suffix(".tmp.h5ad")
    adata.write_h5ad(temp, compression="lzf")
    temp.replace(path)


def reference_pair(section):
    idx = SECTIONS.index(section)
    return SECTIONS[idx - 1], SECTIONS[idx + 1]


def prepare(root, output):
    ccf = pd.read_csv(root / "ccf_coordinates.csv", dtype={"cell_label": str}).set_index(
        "cell_label", verify_integrity=True)
    meta = pd.read_csv(root / "cell_metadata_with_cluster_annotation.csv",
                       usecols=["cell_label", "brain_section_label", "subclass"],
                       dtype=str).set_index("cell_label", verify_integrity=True)
    registered = ccf.join(meta, how="left", validate="1:1")
    output.mkdir(parents=True, exist_ok=True)
    split, genes = [], None
    for path in sorted(root.glob("sec_*.h5ad")):
        section = int(path.stem[4:])
        with h5py.File(path) as f:
            obs, var = f["obs"], f["var"]
            ids = pd.Index(obs[obs.attrs["_index"]].asstr()[:])
            mapped = registered.reindex(ids)
            xyz = mapped[["z", "y", "x"]].to_numpy(dtype=np.float64) * 1000
            keep = np.isfinite(xyz).all(axis=1)
            if keep.any() != (section in SECTIONS):
                raise ValueError(f"Unexpected CCF coverage for {path.name}")
            if section not in SECTIONS:
                continue
            label = f"Zhuang-ABCA-2.{section:03d}"
            groups = mapped.loc[keep]
            if not ids.is_unique or not (groups.brain_section_label == label).all() or groups.subclass.isna().any():
                raise ValueError(f"Invalid keys, section assignment or subclass for {path.name}")
            this_panel = pd.DataFrame({"ensembl": var["ensembl"].asstr()[:]},
                                      index=pd.Index(var["gene_symbol"].asstr()[:], name="gene_symbol"))
            if genes is None:
                genes = this_panel
            if (not this_panel.equals(genes) or not genes.index.is_unique or len(genes) != 1122
                    or f["X"].shape != (len(ids), 1122)):
                raise ValueError(f"Inconsistent gene panel: {path.name}")
            idx = SECTIONS.index(section)
            depth = float(np.median(xyz[keep, 2]))
            fold = "train" if section in TRAIN else "validation" if section in VALIDATION else "test"
            split.append(dict(idx=idx, section=section, section_label=label, split=fold, z_um=depth,
                              raw_cells=len(ids), cells=int(keep.sum())))
            temp = output / f"section_{section:03d}.tmp"
            with temp.open("wb") as stream:
                np.savez_compressed(stream, rows=np.flatnonzero(keep), names=ids[keep].to_numpy(dtype=str),
                                    xyz=xyz[keep].astype(np.float32),
                                    subclass=groups.subclass.to_numpy(dtype=str))
            temp.replace(output / f"section_{section:03d}.npz")
    split = pd.DataFrame(split)
    if (len(split) != len(SECTIONS) or not split.z_um.is_monotonic_increasing
            or split.z_um.duplicated().any() or split.cells.sum() != len(registered)):
        raise ValueError("CCF cells or AP ordering differ from the frozen 54 sections")
    genes.to_csv(output / "genes.csv")
    split.to_csv(output / "split.csv", index=False)
    atomic_json(output / "prepared.json", dict(
        seed=SEED, sections=len(split), cells=int(split.cells.sum()),
        cells_without_ccf=int((split.raw_cells - split.cells).sum()), genes=len(genes),
        coordinates="CCF (ML, DV, AP) in micrometers from CSV (z, y, x) in millimeters",
        slice_depth="median registered AP", annotation="Allen ABC Atlas subclass",
        expression=EXPRESSION, data_root=str(root.resolve())))
    print(split.groupby("split")[["cells", "raw_cells"]].sum().to_string(), flush=True)


class Zhuang:
    def __init__(self, root, prepared):
        self.root = root
        self.prepared = prepared
        self.protocol = json.loads((prepared / "prepared.json").read_text())
        self.split = pd.read_csv(prepared / "split.csv").set_index("section")
        self.var = pd.read_csv(prepared / "genes.csv", index_col=0)
        self.z = self.split.z_um.to_dict()

    def metadata(self, section):
        with np.load(self.prepared / f"section_{section:03d}.npz", allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def expression(self, section, rows):
        with h5py.File(self.root / f"sec_{section:03d}.h5ad") as handle:
            x = handle["X"]
            values = np.empty((len(rows), len(self.var)), dtype=np.float32)
            for start in range(0, len(rows), 4096):
                chunk = rows[start:start + 4096]
                raw = x[chunk[0]:chunk[-1] + 1] if chunk[-1] - chunk[0] + 1 <= 2 * len(chunk) else x[chunk]
                if len(raw) != len(chunk):
                    raw = raw[chunk - chunk[0]]
                if not np.isfinite(raw).all() or (raw < 0).any():
                    raise ValueError(f"Invalid counts in section {section}")
                np.log1p(raw, out=values[start:start + len(chunk)])
        return values

    def section(self, section, for_uot=False):
        import anndata as ad

        data = self.metadata(section)
        x = self.expression(section, data["rows"])
        mask = x.any(axis=1) if for_uot else slice(None)
        annotation = pd.Categorical(data["subclass"][mask])
        obs = pd.DataFrame(dict(subclass=annotation, annotation=annotation, n_section=section,
                                z_coord=self.z[section]), index=data["names"][mask])
        xyz = data["xyz"][mask]
        return ad.AnnData(x[mask] if for_uot else x, obs=obs, var=self.var.copy(),
                          obsm={"spatial": xyz[:, :2].copy(), "spatial_3d_aligned": xyz})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.root, args.output)
