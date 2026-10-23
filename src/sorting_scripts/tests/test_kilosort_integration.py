"""Opt-in real CPU KS4 run and checkpoint replay; no lab recordings/downloads."""

from importlib import import_module
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(os.environ.get("ZLAB_RUN_KS4_INTEGRATION") == "1", "Opt-in real CPU Kilosort integration")
class KilosortIntegrationTests(unittest.TestCase):
    def test_real_sort_and_resume_are_identical(self):
        import torch
        import spikeinterface as si
        import spikeinterface.extractors as se
        from spikeinterface.sorters import run_sorter
        from sorting_scripts.kilosort_runtime import kilosort_runtime, resume_clustering

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        directory = tempfile.TemporaryDirectory(prefix="zlab-ks4-integration-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        recording, _ = si.generate_ground_truth_recording(
            durations=[20.], sampling_frequency=30000., num_channels=16, num_units=6,
            generate_sorting_kwargs={"firing_rates": 100., "refractory_period_ms": 4.}, seed=42,
        )
        recording = recording.save(folder=root / "recording", n_jobs=1)
        checkpoint = root / "sorter/precluster_checkpoint.pt"
        swarm = import_module("kilosort.swarmsplitter")
        # Count calls to ensure the actual graph/splitting path was reached.
        with patch.object(swarm, "check_CCG", wraps=swarm.check_CCG) as ccg:
            with kilosort_runtime(checkpoint):
                sorting = run_sorter(
                    "kilosort4", recording, folder=root / "sorter",
                    torch_device="cpu", do_correction=False, nearest_chans=8,
                    n_pcs=3, n_templates=3, templates_from_data=True, nskip=1,
                    remove_existing_folder=False, delete_recording_dat=False,
                )
            self.assertGreater(ccg.call_count, 0, "Synthetic fixture never reached refractory splitting")
        self.assertGreater(sorting.get_num_units(), 0)
        self.assertTrue(checkpoint.is_file())
        resumed = resume_clustering(checkpoint, root / "resumed")
        loaded = se.read_kilosort(resumed)
        self.assertEqual(sorting.get_num_units(), loaded.get_num_units())
        for name in ["spike_times.npy", "spike_clusters.npy", "templates.npy"]:
            np.testing.assert_array_equal(
                np.load(root / "sorter/sorter_output" / name), np.load(resumed / name),
                err_msg=f"Resume changed {name}",
            )
        self.assertTrue((resumed / "zlab_resume_complete.json").is_file())
