"""Run: python -m unittest discover -s tests -p test_merfish_holdout.py"""

import tempfile
import unittest
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from experiments import merfish_holdout as mh


class HoldoutChecks(unittest.TestCase):
    def setUp(self):
        self.work_root = Path(__file__).resolve().parents[1] / "ccfa-workfiles/experiments/merfish-holdout/build"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=self.work_root)
        for sid in range(5):
            xy = np.column_stack((np.arange(6) * 20.0, np.arange(6) * 7.0))
            obs = pd.DataFrame({"slice_id": [str(0.04 + sid * 0.05)] * 6,
                                "annotation": ["A", "B"] * 3,
                                "cell_class": ["type_a", "type_b"] * 3},
                               index=[f"cell{i}" for i in range(6)])
            values = np.column_stack((np.arange(6) + sid + 1,
                                      np.arange(6) * 2 + sid * 3 + 2,
                                      np.arange(6) * 3 + sid * 2 + 1)).astype(np.float32)
            a = ad.AnnData(values, obs=obs, var=pd.DataFrame(index=["g1", "g2", "g3"]))
            a.obsm["spatial"] = xy
            a.write_h5ad(Path(self.tmp.name) / f"merfish_{sid}.h5ad")
        self.adata = mh.load_merfish(self.tmp.name)
        self.config = mh.default_config()
        self.config["label_neighbors"] = 1

    def tearDown(self):
        assert Path(self.tmp.name).resolve().is_relative_to(self.work_root.resolve())
        self.tmp.cleanup()

    def test_conversion_and_heldout_expression_label_isolation(self):
        self.assertEqual(self.adata.shape, (30, 3))
        self.assertTrue(self.adata.obs_names.is_unique)
        self.assertEqual(self.adata.obsm["spatial"].shape, (30, 3))
        train, query, truth, params = mh.make_fold(self.adata, "merfish_2", self.config)
        changed = self.adata.copy()
        mask = changed.obs["slice_ID"].to_numpy() == "merfish_2"
        changed.X[mask] = 12345
        changed.obs["annotation"] = changed.obs["annotation"].astype(str)
        changed.obs.loc[mask, "annotation"] = "UNSEEN_ORACLE_LABEL"
        changed.obs["cell_class"] = changed.obs["cell_class"].astype(str)
        changed.obs.loc[mask, "cell_class"] = "UNSEEN_ORACLE_TYPE"
        other_train, other_query, _, other_params = mh.make_fold(changed, "merfish_2", self.config)
        np.testing.assert_array_equal(train.X, other_train.X)
        np.testing.assert_array_equal(query.X, 0)
        np.testing.assert_array_equal(query.obs["label_"], other_query.obs["label_"])
        self.assertEqual(params, other_params)
        query_values = mh.metric_expression(query.X, self.config["expression_transform"])
        np.testing.assert_array_equal(mh.fit_cell_type_readout(train, self.config).predict(query_values),
                                      mh.fit_cell_type_readout(other_train, self.config).predict(query_values))
        self.assertNotIn("annotation", query.obs)
        self.assertFalse(set(train.obs_names).intersection(truth.obs_names))
        for a in (train, query):
            edges = a.uns["spatial_net"]
            self.assertTrue(set(edges.Cell1).issubset(a.obs_names))
            self.assertTrue(set(edges.Cell2).issubset(a.obs_names))

    def test_shared_coordinate_frame(self):
        self.config["expression_transform"] = "none"
        train, query, truth, params = mh.make_fold(self.adata, "merfish_2", self.config)
        expected = np.rint((query.obsm["spatial"] - params["origin"]) / params["divisions"])
        np.testing.assert_array_equal(query.obsm["new_spatial"], expected)

    def test_global_mean_metrics_do_not_become_pointwise_metrics(self):
        truth = np.array([[0, 0, 2], [2, 4, 4]], dtype=np.float32)
        prediction = truth[::-1].copy()
        summary, genes, _ = mh.evaluate(truth, prediction, ["a", "b", "c"], ["x", "y"])
        self.assertEqual(summary["mse"], 0)
        self.assertEqual(summary["mae"], 0)
        self.assertAlmostEqual(summary["pcc"], 1)
        self.assertEqual(summary["pointwise_mse"], 8)
        np.testing.assert_array_equal(genes.truth_mean_log_expression, [1, 2, 3])
        raw = np.array([[0, 1, 8], [2, 3, 4]], dtype=np.float32)
        np.testing.assert_allclose(mh.metric_expression(raw, "none"), mh.metric_expression(np.log1p(raw), "log1p"))

    def test_rare_same_type_coverage_counts_missing_predictions_as_zero(self):
        obs = pd.DataFrame({"cell_class": ["rare", "rare", "missing"]}, index=["x", "y", "z"])
        truth = ad.AnnData(np.array([[0, 1], [1, 2], [2, 3]], dtype=np.float32),
                          obs=obs, var=pd.DataFrame(index=["g1", "g2"]))
        truth.obsm["spatial_xy"] = np.array([[0, 0], [100, 0], [500, 0]], dtype=float)
        prediction = truth.copy()
        prediction.obs["cell_class"] = ["common", "rare", "common"]
        counts = pd.Series({"common": 991, "rare": 4, "missing": 3, "absent_truth": 2})
        summary, table = mh.evaluate_cell_types(truth, prediction, counts, self.config)
        by_class = table.set_index("cell_class")
        self.assertEqual(summary["rare_classes_present_in_truth"], 2)
        self.assertEqual(summary["rare_missing_classes"], 1)
        self.assertEqual(summary["rare_recall_50um"], 0.25)
        self.assertEqual(summary["rare_recall_100um"], 0.5)
        self.assertEqual(by_class.loc["missing", "coverage_recall_50um"], 0)
        self.assertTrue(np.isnan(by_class.loc["missing", "coverage_precision_50um"]))
        self.assertEqual(summary["symmetric_mean_nn_um"], 0)
        self.assertEqual(summary["count_error_pct"], 0)

    def test_expression_readout_uses_training_labels_only(self):
        train, query, truth, _ = mh.make_fold(self.adata, "merfish_2", self.config)
        readout = mh.fit_cell_type_readout(train, self.config)
        values = mh.metric_expression(train.X, self.config["expression_transform"])
        prediction = readout.predict(values)
        np.testing.assert_array_equal(prediction, train.obs.cell_class.astype(str))
        self.assertNotIn("cell_class", query.obs)

    def test_metrics_identity_and_constant_columns(self):
        y = np.array([[1, 2, 4], [3, 4, 4], [5, 8, 4]], dtype=np.float32)
        summary, genes, _ = mh.evaluate(y, y, ["a", "b", "constant"], ["x", "y", "z"])
        self.assertEqual(summary["mse"], 0)
        self.assertEqual(summary["mae"], 0)
        self.assertAlmostEqual(summary["gene_pearson_mean"], 1)
        self.assertEqual(summary["valid_gene_pearson"], 2)
        self.assertTrue(np.isnan(genes.iloc[2].pearson))

    def test_saved_prediction_gets_new_metrics_without_training(self):
        train, query, truth, _ = mh.make_fold(self.adata, "merfish_2", self.config)
        output_root = Path(self.tmp.name) / "results"
        output = output_root / "merfish_2"
        output.mkdir(parents=True)
        prediction = query.copy()
        prediction.X = mh.transform_expression(truth.X, self.config["expression_transform"])
        prediction.write_h5ad(output / "prediction.h5ad")
        legacy_config = dict(self.config, baseline_neighbors=5)
        identity = {"config": legacy_config, "heldout": "merfish_2", "dataset_sha256": mh.dataset_digest(self.adata)}
        (output / "run_config.json").write_text(json.dumps(identity), encoding="utf-8")
        frame = mh.run_fold(self.adata, "merfish_2", self.config, output_root)
        self.assertEqual(frame.mse.iloc[0], 0)
        self.assertEqual(frame.generation_metrics_available.iloc[0], "saved_mean_prediction_only")
        self.assertTrue((output / "rare_cell_metrics.csv").is_file())
        self.assertEqual(json.loads((output / "evaluation_config.json").read_text())["metric_protocol"], mh.METRIC_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
