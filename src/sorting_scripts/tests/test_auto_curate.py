import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import spikeinterface as si

from sorting_scripts import auto_curate


class ThresholdTests(unittest.TestCase):
    def test_inclusive_bounds_nan_inf_and_absolute_values(self):
        metrics = pd.DataFrame(
            {"snr": [5, 4, np.nan, np.inf, 6], "amplitude": [-30, -50, -50, -50, -20]},
            index=["edge", "low", "nan", "inf", "amplitude"],
        )
        labels = auto_curate._threshold_labels(metrics, {
            "snr": {"greater": 5}, "amplitude": {"greater": 30, "abs": True},
        })
        self.assertEqual(list(labels["label"]), ["good", "noise", "noise", "noise", "noise"])
        self.assertEqual(labels.loc["amplitude", "reason"], "amplitude")

    def test_missing_metrics_fail_instead_of_accepting_units(self):
        with self.assertRaisesRegex(ValueError, "Required metric columns"):
            auto_curate._threshold_labels(pd.DataFrame(index=[1]), {"snr": {"greater": 5}})

    def test_invalid_policy_fails_before_running(self):
        for kwargs in (
            {"label_method": "unknown"}, {"keep_labels": ()},
            {"thresholds": {"snr": {"greater": 5, "less": 3}}},
            {"thresholds": {"snr": {"greater": np.nan}}},
            {"extension_params": {"quality_metrics": {}}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                auto_curate.AutoCurator(auto_curate.AutoCurationConfig(**kwargs))

    def test_unavailable_optional_labeler_fails_with_version_guidance(self):
        with patch.object(auto_curate.curation, "bombcell_label_units", None, create=True):
            with self.assertRaisesRegex(RuntimeError, "label_method='threshold'"):
                auto_curate.AutoCurator(auto_curate.AutoCurationConfig(label_method="bombcell"))


class AutoCurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording, cls.sorting = si.generate_ground_truth_recording(
            durations=[2.0], sampling_frequency=30000, num_channels=4,
            num_units=3, seed=12,
            generate_sorting_kwargs={"firing_rates": 20, "refractory_period_ms": 4},
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.analyzer = si.create_sorting_analyzer(self.sorting, self.recording, sparse=False)
        self.config = auto_curate.AutoCurationConfig(
            enable_model_labeling=False,
            apply_quality_filtering=True,
            thresholds={"num_spikes": {"greater": 1}},
            quality_metric_names=("num_spikes", "firing_rate", "snr"),
            remove_redundant_units=False, auto_merge=False,
            job_kwargs={"n_jobs": 1, "progress_bar": False},
            extension_params={"random_spikes": {"max_spikes_per_unit": 30, "seed": 0}},
        )

    def run_curator(self):
        return auto_curate.AutoCurator(self.config).run(self.analyzer, self.root / "run")

    def test_stock_pipeline_persists_metrics_and_does_not_change_input(self):
        self.config.export_report = True
        self.config.export_phy = True
        result = self.run_curator()
        self.assertEqual(result.summary()["kept_units"], 3)
        self.assertEqual(list(result.decisions["decision"]), ["kept"] * 3)
        self.assertEqual(self.analyzer.get_loaded_extension_names(), [])
        reloaded = si.load_sorting_analyzer(result.analyzer_folder)
        np.testing.assert_array_equal(reloaded.unit_ids, self.sorting.unit_ids)
        self.assertIn("auto_curation_label", reloaded.sorting.get_property_keys())
        self.assertTrue((result.output_folder / "report").is_dir())
        self.assertTrue((result.output_folder / "phy" / "spike_times.npy").is_file())
        manifest = json.loads((result.output_folder / "curation.json").read_text())
        self.assertEqual(manifest["duplicate_spikes_removed"], 0)
        self.assertEqual(manifest["spikeinterface_version"], si.__version__)
        for filename in ("config.json", "decisions.csv", "metrics_before.csv", "metrics_after.csv", "labels.csv"):
            self.assertTrue((result.output_folder / filename).exists())
        with self.assertRaises(FileExistsError):
            self.run_curator()

    def test_every_unit_can_be_rejected_and_saved(self):
        self.config.thresholds = {"num_spikes": {"greater": 100000}}
        result = self.run_curator()
        self.assertEqual(result.analyzer.get_num_units(), 0)
        self.assertTrue((result.decisions["decision"] == "rejected").all())
        self.assertEqual(si.load_sorting_analyzer(result.analyzer_folder).get_num_units(), 0)
        self.assertEqual(len(result.labels), 3)

    def test_stock_duplicate_spikes_and_sparse_empty_unit_cleanup(self):
        spike_times = np.array([100, 101, 500, 900])
        sorting = si.NumpySorting.from_unit_dict(
            {"unit": spike_times, "empty": np.array([], dtype="int64")}, 30000,
        )
        sparsity = si.ChannelSparsity(np.ones((2, 4), dtype=bool), sorting.unit_ids, self.recording.channel_ids)
        self.analyzer = si.create_sorting_analyzer(sorting, self.recording, sparsity=sparsity)
        result = self.run_curator()
        self.assertEqual(result.decisions.at["empty", "decision"], "empty")
        self.assertEqual(result.analyzer.get_num_units(), 1)
        np.testing.assert_array_equal(
            result.analyzer.sorting.get_unit_spike_train("unit"), [100, 500, 900],
        )
        self.assertEqual(len(self.analyzer.sorting.get_unit_spike_train("unit")), 4)

    def test_stock_redundancy_removal(self):
        times = self.sorting.get_unit_spike_train(self.sorting.unit_ids[0])
        sorting = si.NumpySorting.from_unit_dict({"first": times, "duplicate": times}, 30000)
        self.analyzer = si.create_sorting_analyzer(sorting, self.recording, sparse=False)
        self.config.remove_redundant_units = True
        result = self.run_curator()
        self.assertEqual(result.analyzer.get_num_units(), 1)
        self.assertEqual(result.decisions["decision"].value_counts().to_dict(), {"kept": 1, "redundant": 1})

    def test_hard_merge_uses_stock_analyzer_and_records_lineage(self):
        self.config.auto_merge = True
        first_id, second_id = self.sorting.unit_ids[:2]
        with patch.object(auto_curate.curation, "compute_merge_unit_groups", return_value=[(first_id, second_id)]) as detect:
            result = self.run_curator()
        self.assertEqual(result.analyzer.get_num_units(), 2)
        self.assertEqual(result.decisions.at[first_id, "decision"], "merged_kept")
        self.assertEqual(result.decisions.at[first_id, "final_unit_id"], result.decisions.at[second_id, "final_unit_id"])
        merged_id = result.decisions.at[first_id, "final_unit_id"]
        self.assertEqual(result.merge_groups[0]["source_unit_ids"], [first_id, second_id])
        self.assertEqual(
            result.metrics_after.at[merged_id, "num_spikes"],
            len(result.analyzer.sorting.get_unit_spike_train(merged_id)),
        )
        self.assertEqual(detect.call_args.kwargs["preset"], "similarity_correlograms")

    def test_stock_merge_detection_runs_with_default_preset(self):
        self.config.auto_merge = True
        result = self.run_curator()
        self.assertGreater(result.analyzer.get_num_units(), 0)

    def test_default_metrics_and_integer_ids(self):
        sorting = self.sorting.rename_units([10, 20, 30])
        self.analyzer = si.create_sorting_analyzer(sorting, self.recording, sparse=False)
        self.config.quality_metric_names = auto_curate.AutoCurationConfig().quality_metric_names
        result = self.run_curator()
        np.testing.assert_array_equal(result.analyzer.unit_ids, [10, 20, 30])
        self.assertIn("isi_violations_ratio", result.metrics_after)
        self.assertIn("amplitude_cutoff", result.metrics_after)

    def test_bombcell_requires_all_enabled_metrics(self):
        self.config.label_method = "bombcell"
        self.config.bombcell_thresholds = {"noise": {"snr": {"greater": 5}}}
        metrics = pd.DataFrame({"snr": [6]}, index=["first"])
        predictions = pd.DataFrame({"label": ["good"]}, index=["first"])
        with patch.object(auto_curate.curation, "bombcell_label_units", return_value=predictions, create=True) as labeler:
            labels = auto_curate._label_units(self.analyzer, metrics, self.config)
            self.assertEqual(labels.at["first", "label"], "good")
            labeler.assert_called_once_with(self.analyzer, thresholds=self.config.bombcell_thresholds)
            self.config.bombcell_thresholds["noise"]["missing_metric"] = {"less": 1}
            with self.assertRaisesRegex(ValueError, "missing_metric"):
                auto_curate._label_units(self.analyzer, metrics, self.config)

if __name__ == "__main__":
    unittest.main()
