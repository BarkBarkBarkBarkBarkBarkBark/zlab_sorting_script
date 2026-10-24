"""Deterministic inference contracts plus real analyzer persistence and GUI tables."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import spikeinterface as si
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sorting_scripts import auto_curate, model_labeling


def make_model():
    model = Pipeline([("classifier", DecisionTreeClassifier(random_state=0))])
    model.fit(pd.DataFrame({"snr": [0.0, 10.0]}), [0, 1])
    return model


class ModelLabelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording, cls.sorting = si.generate_ground_truth_recording(
            durations=[2.0], sampling_frequency=30000, num_channels=4, num_units=3, seed=12,
            generate_sorting_kwargs={"firing_rates": 20, "refractory_period_ms": 4},
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.analyzer = si.create_sorting_analyzer(self.sorting, self.recording, sparse=False)
        self.config = auto_curate.AutoCurationConfig(
            thresholds={}, quality_metric_names=("snr",), apply_quality_filtering=False,
            auto_merge=False, remove_redundant_units=False,
            job_kwargs={"n_jobs": 1, "progress_bar": False},
            extension_params={"random_spikes": {"max_spikes_per_unit": 30, "seed": 0}},
        )
        self.calls = []
        self.all_noise = False

    def load(self, repo_id, revision, trust):
        model = make_model()
        model.classifier_name = "noise" if "noise_neural" in repo_id else "sua_mua"
        return model, {"metric_params": {}}, {"repo_id": repo_id, "revision": "test-revision",
                                             "required_metrics": ["snr"]}

    def predict(self, analyzer, model, info, config):
        ids = analyzer.unit_ids
        self.calls.append((model.classifier_name, list(ids)))
        if model.classifier_name == "noise":
            labels = ["noise" if self.all_noise or index == 1 else "neural" for index in range(len(ids))]
        else:
            labels = ["sua"] * len(ids)
        return pd.DataFrame({"label": labels, "probability": [0.95] * len(ids)}, index=ids)

    def run_pipeline(self, folder="run"):
        with patch.object(model_labeling, "_load_classifier", side_effect=self.load) as loader:
            with patch.object(model_labeling, "_predict", side_effect=self.predict):
                result = auto_curate.AutoCurator(self.config).run(self.analyzer, self.root / folder)
        return result, loader

    def test_default_dual_labeling_alignment_persistence_and_gui(self):
        self.analyzer.sorting.set_property("human_label", ["keep", "inspect", "keep"])
        result, loader = self.run_pipeline()
        self.assertEqual(loader.call_count, 2)
        self.assertTrue(loader.call_args.args[2])
        self.assertEqual(result.analyzer.get_num_units(), 3)
        self.assertEqual(self.calls, [("noise", ["0", "1", "2"]), ("sua_mua", ["0", "2"])])
        self.assertEqual(result.ml_classifications.loc["1", "sua_mua_status"], "not_evaluated")
        self.assertEqual(result.ml_classifications.loc["1", "sua_mua_label"], "")
        self.assertFalse(result.ml_classifications["filter_applied"].any())
        saved = si.load_sorting_analyzer(result.analyzer_folder)
        np.testing.assert_array_equal(saved.sorting.get_property("ml_noise_label"), ["neural", "noise", "neural"])
        np.testing.assert_array_equal(saved.sorting.get_property("ml_sua_label"), ["sua", "", "sua"])
        np.testing.assert_array_equal(saved.sorting.get_property("human_label"), ["keep", "inspect", "keep"])
        self.assertNotIn("classifier_label", saved.sorting.get_property_keys())
        self.assertNotIn("ml_noise_label", self.analyzer.sorting.get_property_keys())
        from spikeinterface.widgets.utils import make_units_table_from_analyzer

        options = result.gui_kwargs()
        gui_table = make_units_table_from_analyzer(saved, extra_properties=options["extra_unit_properties"])
        self.assertEqual(gui_table.loc["1", "ml_noise_probability"], 0.95)
        self.assertTrue(np.isnan(gui_table.loc["1", "ml_sua_probability"]))
        report = pd.read_csv(result.output_folder / "unit_ml_classifications.csv", dtype={"unit_id": str})
        self.assertEqual(report["unit_id"].tolist(), list(saved.unit_ids))

    def test_disabled_skips_loading_and_inference(self):
        self.config.enable_model_labeling = False
        result, loader = self.run_pipeline()
        loader.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertTrue(result.ml_classifications["model_labeling_status"].eq("disabled").all())

    def test_all_noise_skips_sua_and_keeps_all(self):
        self.all_noise = True
        result, loader = self.run_pipeline()
        self.assertEqual(loader.call_count, 2)  # Both checked before processing; SUA inference skipped.
        self.assertEqual(result.analyzer.get_num_units(), 3)
        self.assertTrue(result.ml_classifications["sua_mua_status"].eq("not_evaluated").all())

    def test_sua_without_noise_evaluates_all(self):
        self.config.enable_noise_classifier = False
        result, loader = self.run_pipeline()
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(self.calls, [("sua_mua", ["0", "1", "2"])])

    def test_filtering_is_explicit_and_independent(self):
        self.config.model_filtering = {"noise": {"exclude_labels": ["noise"], "min_probability": 0.9}}
        result, _ = self.run_pipeline("review")
        self.assertTrue(result.ml_classifications.at["1", "filter_candidate"])
        self.assertEqual(result.analyzer.get_num_units(), 3)
        self.config.apply_model_filtering = True
        result, _ = self.run_pipeline("filtered")
        self.assertEqual(result.analyzer.get_num_units(), 2)
        self.assertTrue(result.ml_classifications.at["1", "filter_applied"])
        self.config.model_filtering = {"sua_mua": {"exclude_labels": ["sua"], "min_probability": 0.9}}
        result, _ = self.run_pipeline("sua_filtered")
        np.testing.assert_array_equal(result.analyzer.unit_ids, ["1"])

    def test_loading_failure_is_reported_and_raised(self):
        with patch.object(model_labeling, "_load_classifier", side_effect=OSError("network/cache unavailable")), \
             patch.object(auto_curate, "_compute_metrics") as compute, \
             patch.object(auto_curate.curation, "remove_duplicated_spikes") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "noise preflight failed.*network/cache"):
                auto_curate.AutoCurator(self.config).run(self.analyzer, self.root / "run")
        compute.assert_not_called()
        cleanup.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_one_line_preflight_checks_both_models_without_metrics(self):
        with patch.object(model_labeling, "_load_classifier", side_effect=self.load) as loader, \
             patch.object(auto_curate, "_compute_metrics") as compute:
            result = auto_curate.preflight(self.config)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(set(result["models"]), {"noise", "sua_mua"})
        self.assertEqual(loader.call_count, 2)
        compute.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_preflight_disabled_models_does_not_load(self):
        self.config.enable_model_labeling = False
        with patch.object(model_labeling, "_load_classifier") as loader:
            self.assertEqual(auto_curate.preflight(self.config), {"status": "passed", "models": {}})
        loader.assert_not_called()

    def test_sua_preflight_failure_also_precedes_processing(self):
        with patch.object(model_labeling, "_load_classifier", side_effect=[self.load("noise_neural", None, True),
                                                                                 RuntimeError("sklearn mismatch")]), \
             patch.object(auto_curate, "_compute_metrics") as compute:
            with self.assertRaisesRegex(RuntimeError, "sua_mua preflight failed.*sklearn mismatch"):
                auto_curate.AutoCurator(self.config).run(self.analyzer, self.root / "run")
        compute.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_post_merge_units_are_reclassified(self):
        self.config.auto_merge = True
        with patch.object(auto_curate.curation, "compute_merge_unit_groups", return_value=[["0", "1"]]):
            result, _ = self.run_pipeline()
        self.assertEqual(result.analyzer.get_num_units(), 2)
        merged_id = result.merge_groups[0]["new_unit_id"]
        self.assertIn(merged_id, self.calls[0][1])
        self.assertNotIn("0", self.calls[0][1])
        self.assertIn(merged_id, result.ml_classifications.index)

    def test_rerun_uses_new_outputs_and_integer_unit_ids(self):
        self.analyzer = si.create_sorting_analyzer(self.sorting.rename_units([10, 20, 30]), self.recording, sparse=False)
        first, _ = self.run_pipeline("first")
        second, _ = self.run_pipeline("second")
        pd.testing.assert_frame_equal(first.ml_classifications, second.ml_classifications)
        np.testing.assert_array_equal(second.analyzer.unit_ids, [10, 20, 30])

    def test_zero_units_skip_models(self):
        self.analyzer = self.analyzer.select_units([])
        result, loader = self.run_pipeline()
        loader.assert_not_called()
        self.assertEqual(result.analyzer.get_num_units(), 0)
        self.assertTrue(result.ml_classifications.empty)

    def test_missing_metrics_are_computed_and_unknown_features_rejected(self):
        auto_curate._compute_metrics(self.analyzer, self.config)
        metrics, _ = model_labeling._prepare_metrics(self.analyzer, ["num_spikes", "half_width"], {"metric_params": {}}, self.config)
        self.assertIn("num_spikes", metrics)
        self.assertIn("half_width", metrics)
        with self.assertRaisesRegex(ValueError, "unsupported metric"):
            model_labeling._prepare_metrics(self.analyzer, ["fictional"], {"metric_params": {}}, self.config)
        self.config.quality_metric_params = {"snr": {"peak_sign": "pos"}}
        info = {"metric_params": {"quality_metric_params": {"metric_params": {"snr": {"peak_sign": "neg"}}}}}
        with self.assertRaisesRegex(ValueError, "conflict"):
            model_labeling._prepare_metrics(self.analyzer, ["snr"], info, self.config)

    def test_stock_loaded_model_api_and_prediction_id_validation(self):
        auto_curate._compute_metrics(self.analyzer, self.config)
        selected = self.analyzer.select_units(["2", "0"])
        info = {"metric_params": {}, "label_conversion": {"0": "noise", "1": "neural"}}
        predictions = model_labeling._predict(selected, make_model(), info, self.config)
        self.assertEqual(list(predictions.index), ["2", "0"])
        self.assertNotIn("classifier_label", self.analyzer.sorting.get_property_keys())

    def test_filtering_requires_policy(self):
        self.config.apply_model_filtering = True
        with self.assertRaisesRegex(ValueError, "explicit model_filtering"):
            auto_curate.AutoCurator(self.config)

    def test_model_version_mismatch_fails_before_deserializing(self):
        model_folder = self.root / "snapshot-revision"
        model_folder.mkdir()
        (model_folder / "model_info.json").write_text(json.dumps({
            "metric_params": {}, "requirements": {"scikit-learn": "1.4.2"},
        }))
        (model_folder / "model.skops").touch()
        with patch("huggingface_hub.snapshot_download", return_value=str(model_folder)):
            with patch.object(model_labeling, "version", return_value="1.9.0"):
                with patch.object(model_labeling.curation, "load_model") as loader:
                    with self.assertRaisesRegex(RuntimeError, "1.4.2.*1.9.0"):
                        model_labeling._load_classifier("example/model", None, True)
                    loader.assert_not_called()

    def test_cached_model_loading_and_download_fallback(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        folder = self.root / "commit-id"
        folder.mkdir()
        (folder / "model_info.json").write_text(json.dumps({"metric_params": {}}))
        (folder / "model.skops").write_bytes(b"mock artifact")
        for cached in (True, False):
            outcomes = [str(folder)] if cached else [LocalEntryNotFoundError("cache miss"), str(folder)]
            with self.subTest(cached=cached), patch("huggingface_hub.snapshot_download", side_effect=outcomes) as download:
                with patch.object(model_labeling.curation, "load_model", return_value=(make_model(), {"metric_params": {}})):
                    _, _, provenance = model_labeling._load_classifier("example/model", "pinned", True)
                self.assertTrue(download.call_args_list[0].kwargs["local_files_only"])
                self.assertEqual(download.call_count, 1 if cached else 2)
                self.assertEqual(provenance["revision"], "commit-id")
                self.assertFalse(provenance["training_parameters_available"])

    @unittest.skipUnless(os.environ.get("ZLAB_RUN_HF_INTEGRATION") == "1", "Opt-in real HF integration")
    def test_actual_huggingface_models(self):
        recording, sorting = si.generate_ground_truth_recording(
            durations=[4.0], sampling_frequency=30000, num_channels=16, num_units=4, seed=42,
            generate_sorting_kwargs={"firing_rates": 20},
        )
        analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
        result = auto_curate.AutoCurator(self.config).run(analyzer, self.root / "hf_noise")
        self.assertTrue(result.ml_classifications["noise_status"].eq("evaluated").all())
        self.config.enable_noise_classifier = False
        result = auto_curate.AutoCurator(self.config).run(analyzer, self.root / "hf_sua")
        saved = si.load_sorting_analyzer(result.analyzer_folder)
        self.assertTrue(result.ml_classifications["sua_mua_status"].eq("evaluated").all())
        self.assertEqual(len(saved.sorting.get_property("ml_sua_label")), 4)


if __name__ == "__main__":
    unittest.main()
