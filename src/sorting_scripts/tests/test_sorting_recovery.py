"""Regression tests for KS4 degenerate CCGs, scoped hooks, and checkpoints."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sorting_scripts import kilosort_runtime as recovery, zsort


class OutputProtectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name) / "existing"
        self.folder.mkdir()
        self.sentinel = self.folder / "valuable_output.dat"
        self.sentinel.write_bytes(b"keep me")

    def test_unreadable_sorter_is_preserved_without_preprocessing_or_rerun(self):
        with patch.object(zsort.ss, "read_sorter_folder", side_effect=ValueError("failed run")), \
             patch.object(zsort, "run_sorter") as run, \
             patch.object(zsort, "_ensure_signed_recording") as preprocess:
            with self.assertRaisesRegex(RuntimeError, "preserved") as caught:
                zsort.sort(object(), {"sorter_output_folder": self.folder})
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        run.assert_not_called()
        preprocess.assert_not_called()
        self.assertEqual(self.sentinel.read_bytes(), b"keep me")

    def test_readable_sorter_is_reused(self):
        with patch.object(zsort.ss, "read_sorter_folder", return_value="sorting"), \
             patch.object(zsort, "run_sorter") as run:
            self.assertEqual(zsort.sort(object(), {"sorter_output_folder": self.folder}), "sorting")
        run.assert_not_called()

    def test_memory_settings_are_forwarded_without_running_kilosort(self):
        from contextlib import nullcontext

        for settings in ({}, {"batch_size": 15000, "clear_cache": True}):
            with self.subTest(settings=settings), \
                 patch.object(zsort, "_ensure_signed_recording", return_value="signed"), \
                 patch.object(zsort.si, "bandpass_filter", return_value="filtered"), \
                 patch.object(zsort.si, "center", return_value="centered"), \
                 patch.object(zsort.si, "whiten", return_value="whitened"), \
                 patch.object(recovery, "kilosort_runtime", return_value=nullcontext()), \
                 patch.object(zsort, "run_sorter", return_value="sorting") as run:
                zsort.sort(object(), {"sorter_output_folder": self.folder / "new"}, **settings)
            self.assertEqual(run.call_args.kwargs["batch_size"], settings.get("batch_size", 60000))
            self.assertEqual(run.call_args.kwargs["clear_cache"], settings.get("clear_cache", False))
            self.assertFalse(run.call_args.kwargs["remove_existing_folder"])

    def test_invalid_memory_settings_fail_before_preprocessing(self):
        settings = [{"batch_size": value} for value in (0, -1, 1.5, True, "15000")]
        settings.append({"clear_cache": "True"})
        with patch.object(zsort, "_ensure_signed_recording") as preprocess:
            for kwargs in settings:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    zsort.sort(object(), {"sorter_output_folder": self.folder / "new"}, **kwargs)
        preprocess.assert_not_called()

    def test_unreadable_analyzer_is_preserved(self):
        with patch.object(zsort.si, "load_sorting_analyzer", side_effect=ValueError("damaged")), \
             patch.object(zsort.si, "create_sorting_analyzer") as create:
            with self.assertRaisesRegex(RuntimeError, "preserved"):
                zsort.analyze(object(), object(), {"analyzer_folder": self.folder})
        create.assert_not_called()
        self.assertEqual(self.sentinel.read_bytes(), b"keep me")

    def test_readable_analyzer_is_reused(self):
        with patch.object(zsort.si, "load_sorting_analyzer", return_value="analyzer"):
            self.assertEqual(zsort.analyze(object(), object(), {"analyzer_folder": self.folder}), "analyzer")

    def test_new_analyzer_disables_overwrite(self):
        analyzer = Mock()
        with patch.object(zsort, "_ensure_signed_recording", return_value="signed"), \
             patch.object(zsort.si, "bandpass_filter", return_value="filtered"), \
             patch.object(zsort.si, "create_sorting_analyzer", return_value=analyzer) as create:
            zsort.analyze(object(), object(), {"analyzer_folder": self.folder / "new"})
        self.assertFalse(create.call_args.kwargs["overwrite"])
        analyzer.compute.assert_called_once()


@unittest.skipUnless(find_spec("kilosort") and find_spec("torch"), "Requires optional sorter extra")
class KilosortRecoveryTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.swarm = import_module("kilosort.swarmsplitter")
        self.ccg = import_module("kilosort.CCG")
        self.runtime = import_module("kilosort.run_kilosort")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.st = np.array([[30., 0., 1.], [60., 0., 1.]])
        self.features = torch.ones((2, 1, 1))
        self.ops = {"settings": {}, "nested": {"tensor": torch.arange(2)}}
        self.kwargs = dict(st=self.st, tF=self.features, ops=self.ops,
                           device=torch.device("cpu"), bfile=SimpleNamespace(imin=7))

    def test_reproduces_original_zero_division_and_guards_zero_and_empty(self):
        original = self.swarm.check_CCG
        with self.assertRaises(ZeroDivisionError):
            original(np.array([10.]), np.array([10.]))
        with recovery.kilosort_runtime():
            for first, second in [([10.], [10.]), ([10., 10.], [10.]), ([], [1.]), ([1.], [])]:
                with self.subTest(first=first, second=second):
                    a, b = np.array(first), np.array(second)
                    self.assertEqual(self.swarm.check_CCG(a, b), (False, False))
                    result = self.ccg.check_CCG(a, b)
                    self.assertEqual(result[:2], (False, False))
                    self.assertTrue(np.isnan(result[2]))
                    self.assertEqual(self.swarm.refractoriness(a, b), 0)
        self.assertIs(self.swarm.check_CCG, original)

    def test_regular_ccgs_are_identical_to_upstream(self):
        rng = np.random.default_rng(13)
        first = np.sort(rng.uniform(0, 30, 100))
        second = np.sort(rng.uniform(0, 30, 80))
        swarm_expected = self.swarm.check_CCG(first, second)
        ccg_expected = self.ccg.check_CCG(first, second, acg_threshold=.1, ccg_threshold=.2)
        with recovery.kilosort_runtime():
            self.assertEqual(self.swarm.check_CCG(first, second), swarm_expected)
            self.assertEqual(self.ccg.check_CCG(first, second, acg_threshold=.1, ccg_threshold=.2), ccg_expected)
            with self.assertRaisesRegex(ValueError, "finite"):
                self.swarm.check_CCG(np.array([0., np.nan]), second)
            with self.assertRaisesRegex(ValueError, "finite"):
                self.swarm.check_CCG(np.array([0.]), np.array([np.nan]))

    def test_all_hooks_restored_on_failure_and_nested_run_rejected(self):
        originals = (self.swarm.check_CCG, self.ccg.check_CCG, self.runtime.cluster_spikes)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with recovery.kilosort_runtime():
                with self.assertRaisesRegex(RuntimeError, "Concurrent or nested"):
                    with recovery.kilosort_runtime():
                        pass
                raise RuntimeError("deliberate")
        self.assertEqual(originals, (self.swarm.check_CCG, self.ccg.check_CCG, self.runtime.cluster_spikes))

    def test_checkpoint_precedes_crash_and_resume_replays_inputs_and_rng(self):
        draws = []
        def cluster(st, tF, ops, device, bfile, tic0=0.):
            draws.append((random.random(), np.random.random(), self.torch.rand(1).item()))
            if bfile is not None:
                self.assertTrue(self.checkpoint.is_file())
                raise RuntimeError("simulated final clustering failure")
            np.testing.assert_array_equal(st, self.st)
            self.torch.testing.assert_close(tF, self.features)
            return np.zeros(2, dtype="int32"), self.torch.ones((1, 1, 1)), st, tF

        with patch.object(self.runtime, "cluster_spikes", cluster):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with recovery.kilosort_runtime(self.checkpoint):
                    self.runtime.cluster_spikes(**self.kwargs)
            # The real torch serializer loads the complete arrays/tensors/state.
            payload = self.torch.load(self.checkpoint, weights_only=False)
            self.assertEqual(payload["imin"], 7)
            self.torch.testing.assert_close(payload["ops"]["nested"]["tensor"], self.ops["nested"]["tensor"])
            before = recovery._rng_state(self.torch)
            with patch.object(self.runtime, "save_sorting") as save:
                result = recovery.resume_clustering(self.checkpoint, self.root / "resumed")
            after = recovery._rng_state(self.torch)
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(before["python"], after["python"])
        np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
        self.torch.testing.assert_close(before["torch"], after["torch"])
        self.assertEqual(save.call_args.kwargs["imin"], 7)
        self.assertTrue((result / "zlab_resume_complete.json").exists())
        self.assertTrue(self.checkpoint.exists())
        with self.assertRaises(FileExistsError):
            recovery.resume_clustering(self.checkpoint, result)

    def test_checkpoint_write_failure_prevents_clustering_and_cleans_partial(self):
        called = Mock()
        def cluster(st, tF, ops, device, bfile):
            called()
        with patch.object(self.runtime, "cluster_spikes", cluster), \
             patch.object(self.torch, "save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                with recovery.kilosort_runtime(self.checkpoint):
                    self.runtime.cluster_spikes(**self.kwargs)
        called.assert_not_called()
        self.assertFalse(self.checkpoint.exists())
        self.assertFalse(self.checkpoint.with_suffix(".pt.writing").exists())

    def test_existing_checkpoint_is_not_replaced(self):
        self.checkpoint.write_bytes(b"previous checkpoint")
        def cluster(st, tF, ops, device, bfile):
            self.fail("Should fail before clustering")
        with patch.object(self.runtime, "cluster_spikes", cluster):
            with self.assertRaises(FileExistsError):
                with recovery.kilosort_runtime(self.checkpoint):
                    self.runtime.cluster_spikes(**self.kwargs)
        self.assertEqual(self.checkpoint.read_bytes(), b"previous checkpoint")

    def test_resume_failure_keeps_checkpoint_and_no_success_marker(self):
        def cluster(st, tF, ops, device, bfile, tic0=0.):
            raise RuntimeError("clustering failed")
        with patch.object(self.runtime, "cluster_spikes", cluster):
            with self.assertRaises(RuntimeError):
                with recovery.kilosort_runtime(self.checkpoint):
                    self.runtime.cluster_spikes(**self.kwargs)
            with self.assertRaisesRegex(RuntimeError, "clustering failed"):
                recovery.resume_clustering(self.checkpoint, self.root / "partial")
        self.assertTrue(self.checkpoint.exists())
        self.assertFalse((self.root / "partial/zlab_resume_complete.json").exists())

    def test_unknown_checkpoint_schema_and_package_version_fail_closed(self):
        self.torch.save({"schema": 2, "versions": {}}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "schema"):
            recovery.resume_clustering(self.checkpoint, self.root / "new")
        self.assertFalse((self.root / "new").exists())
        with patch.object(recovery, "version", return_value="5.0.0"):
            with self.assertRaisesRegex(RuntimeError, "require"):
                with recovery.kilosort_runtime():
                    pass

    def test_new_sort_uses_scoped_adapter_and_retains_binary(self):
        destination = self.root / "sorter"
        with patch.object(zsort, "_ensure_signed_recording", return_value="rec"), \
             patch.object(zsort.si, "bandpass_filter", return_value="filtered"), \
             patch.object(zsort.si, "center", return_value="centered"), \
             patch.object(zsort.si, "whiten", return_value="whitened"), \
             patch.object(zsort, "run_sorter", return_value="sorting") as run:
            self.assertEqual(zsort.sort(object(), {"sorter_output_folder": destination}), "sorting")
        self.assertFalse(run.call_args.kwargs["remove_existing_folder"])
        self.assertFalse(run.call_args.kwargs["delete_recording_dat"])
