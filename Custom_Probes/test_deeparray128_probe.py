import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from probeinterface import read_probeinterface, write_probeinterface
from spikeinterface import NumpyRecording

from create_deeparray128_probe import OUTPUT_PATH, create_probe


class DeepArray128ProbeTests(unittest.TestCase):
    def test_dimensions_and_stagger(self):
        probe = create_probe()
        positions = probe.contact_positions
        self.assertEqual(probe.get_contact_count(), 128)
        self.assertEqual(probe.ndim, 2)
        self.assertEqual(probe.si_units, "um")
        self.assertEqual(len(np.unique(positions, axis=0)), 128)
        np.testing.assert_array_equal(positions[:64, 0], np.full(64, 43.0))
        np.testing.assert_array_equal(positions[64:, 0], np.zeros(64))
        np.testing.assert_array_equal(np.diff(positions[:64, 1]), np.full(63, -50.0))
        np.testing.assert_array_equal(np.diff(positions[64:, 1]), np.full(63, 50.0))
        np.testing.assert_array_equal(positions[:64, 1] - positions[64:][::-1, 1], np.full(64, 25.0))
        self.assertEqual(np.ptp(positions[:, 1]), 3175.0)
        self.assertTrue(all(shape == "circle" for shape in probe.contact_shapes))
        self.assertTrue(all(params == {"radius": 8.0} for params in probe.contact_shape_params))

    def test_numbered_endpoints_and_wiring(self):
        probe = create_probe()
        for channel, position in {0: [43, 3175], 63: [43, 25], 64: [0, 0], 127: [0, 3150]}.items():
            np.testing.assert_array_equal(probe.contact_positions[channel], position)
        np.testing.assert_array_equal(probe.device_channel_indices, np.arange(128))
        np.testing.assert_array_equal(probe.contact_ids, np.arange(128).astype(str))
        self.assertEqual(set(probe.shank_ids), {"0"})

    def test_manufacturer_quality_annotations(self):
        probe = create_probe()
        status = probe.contact_annotations["manufacturer_status"]
        expected_open = [50, 51, 52, 53, *range(56, 71), 72]
        np.testing.assert_array_equal(np.flatnonzero(status == "open"), expected_open)
        self.assertEqual(np.count_nonzero(status == "good"), 108)
        self.assertEqual(probe.get_contact_count(), 128)

    def test_json_round_trip_and_committed_artifact(self):
        expected = create_probe()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            write_probeinterface(path, expected)
            self.assertEqual(path.read_bytes(), OUTPUT_PATH.read_bytes())
            group = read_probeinterface(path)
        self.assertEqual(len(group.probes), 1)
        loaded = group.probes[0]
        np.testing.assert_array_equal(loaded.contact_positions, expected.contact_positions)
        np.testing.assert_array_equal(loaded.device_channel_indices, expected.device_channel_indices)
        np.testing.assert_array_equal(loaded.contact_ids, expected.contact_ids)
        np.testing.assert_array_equal(
            loaded.contact_annotations["manufacturer_status"],
            expected.contact_annotations["manufacturer_status"],
        )
        self.assertEqual(loaded.annotations, expected.annotations)

    def test_spikeinterface_channel_attachment(self):
        probe = read_probeinterface(OUTPUT_PATH).probes[0]
        recording = NumpyRecording(
            np.tile(np.arange(128, dtype=np.float32), (100, 1)),
            sampling_frequency=30000,
        ).rename_channels([f"A-{channel:03d}" for channel in range(128)])
        attached = recording.set_probe(probe)
        self.assertEqual(attached.get_num_channels(), 128)
        np.testing.assert_array_equal(attached.get_channel_locations(), probe.contact_positions)
        np.testing.assert_array_equal(attached.channel_ids, recording.channel_ids)
        np.testing.assert_array_equal(attached.get_traces(), recording.get_traces())

    def test_cli_independent_of_working_directory(self):
        script = Path(__file__).with_name("create_deeparray128_probe.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe.json"
            result = subprocess.run(
                [sys.executable, str(script), "--output", str(output)],
                cwd=directory, check=True, capture_output=True, text=True,
            )
            self.assertIn("128 contacts", result.stdout)
            self.assertEqual(output.read_bytes(), OUTPUT_PATH.read_bytes())


if __name__ == "__main__":
    unittest.main()
