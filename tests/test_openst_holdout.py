import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from experiments import openst_holdout as oh


class OpenSTHoldoutChecks(unittest.TestCase):
    def test_streamed_hvgs_match_scanpy_and_ignore_test_expression(self):
        import scanpy as sc

        rng = np.random.default_rng(2024)
        counts = rng.negative_binomial(2, rng.uniform(0.15, 0.9, 100), size=(380, 100))
        x = sparse.csr_matrix(np.log1p(counts.astype(np.float32)) / np.log(2))
        var = pd.DataFrame(index=[f"g{i}" for i in range(x.shape[1])])
        rows = {section: np.arange(i, len(counts), 19) for i, section in enumerate(oh.SECTIONS)}
        train = np.concatenate([rows[section] for section in oh.TRAIN])
        reference = ad.AnnData(X=x[train], var=var)
        reference.uns["log1p"] = {"base": 2}
        sc.pp.highly_variable_genes(reference, flavor="seurat", n_top_genes=20)
        indices, hvgs = oh.select_hvgs(x, rows, var, log_base=2, n_top=20)
        np.testing.assert_array_equal(indices, np.flatnonzero(reference.var["highly_variable"]))
        np.testing.assert_allclose(hvgs["dispersions_norm"], reference.var.iloc[indices]["dispersions_norm"],
                                   rtol=1e-6)
        changed = x.toarray()
        changed[np.concatenate([rows[section] for section in oh.TEST + oh.VALIDATION])] = 40
        new_indices, new_hvgs = oh.select_hvgs(sparse.csr_matrix(changed), rows, var, log_base=2, n_top=20)
        np.testing.assert_array_equal(new_indices, indices)
        pd.testing.assert_frame_equal(new_hvgs, hvgs)

    def test_chunked_decoder_matches_original_full_graph(self):
        import torch
        from stadiffuser.vae import SpaAE

        torch.manual_seed(2024)
        model = SpaAE(input_dim=16, gat_dim=[64, 32])
        latent = torch.randn(31, 32)
        xy = np.random.default_rng(2024).uniform(0, 350, size=(len(latent), 2))
        graph = oh.spatial_graph(xy).tocoo()
        edges = torch.tensor(np.vstack((graph.col, graph.row)), dtype=torch.long)
        expected = model.decode(latent, edges).numpy()
        with patch.multiple(oh, AE_BATCH=7, DIFFUSION_BATCH=11):
            actual = oh.decode_section(model, latent.numpy(), graph.tocsr(), "cpu")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_periodic_checkpoint_early_stop_and_best_restore(self):
        import torch

        for stage, patience in oh.EARLY_STOP_PATIENCE.items():
            with self.subTest(stage=stage), TemporaryDirectory() as directory:
                model = torch.nn.Linear(1, 1, bias=False)
                losses = iter([0.5] + [0.6] * (oh.EARLY_STOP_WARMUP + patience - 1))
                output = Path(directory)
                validation = oh.StageValidation(output, stage, lambda _: {6: next(losses)})
                for epoch in range(1, oh.EARLY_STOP_WARMUP + patience + 1):
                    with torch.no_grad():
                        model.weight.fill_(epoch)
                    stop = validation(epoch, model)
                    self.assertEqual(stop, epoch == oh.EARLY_STOP_WARMUP + patience)
                    self.assertTrue(model.training)
                checkpoint = torch.load(output / "checkpoints" / f"{stage}_epoch_0010.pt", map_location="cpu")
                self.assertEqual(checkpoint["model"]["weight"].item(), 10)
                result = validation.restore(model)
                self.assertEqual(result["best_epoch"], 1)
                self.assertEqual(model.weight.item(), 1)
