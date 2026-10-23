import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import spikeinterface.full as si

from sorting_scripts import get_file, zsort


class IntanInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def make_files(self, names):
        files = []
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            files.append(path)
        return files

    def recording(self, value=0, samples=4, rate=30000, channels=2, dtype="uint16"):
        recording = si.NumpyRecording(
            np.full((samples, channels), value, dtype=dtype),
            sampling_frequency=rate,
        )
        recording.set_channel_gains(np.full(channels, 0.195))
        recording.set_channel_offsets(np.zeros(channels))
        return recording

    def test_discovery_natural_order_and_case(self):
        self.make_files(["part10.rhd", "part2.RHD", "part1.rhd", "settings.xml"])
        self.assertEqual(
            [p.name for p in get_file.get_rhd_files(self.root)],
            ["part1.rhd", "part2.RHD", "part10.rhd"],
        )

    def test_legacy_raw_and_single_file(self):
        path = self.make_files(["raw/chunk.rhd"])[0]
        self.assertEqual(get_file.get_rhd_files(self.root), [path])
        self.assertEqual(get_file.get_rhd_file(self.root), path)
        self.assertEqual(get_file.get_rhd_files(path), [path])

    def test_invalid_or_ambiguous_inputs_fail(self):
        with self.assertRaises(FileNotFoundError):
            get_file.get_rhd_files(self.root)
        with self.assertRaises(FileNotFoundError):
            get_file.get_rhd_files(self.root / "missing")
        other = self.make_files(["settings.xml"])[0]
        with self.assertRaises(ValueError):
            get_file.get_rhd_files(other)
        self.make_files(["one.rhd", "two.rhd"])
        with self.assertRaisesRegex(ValueError, "load_recording"):
            get_file.get_rhd_file(self.root)
        self.make_files(["raw/three.rhd"])
        with self.assertRaisesRegex(ValueError, "both"):
            get_file.get_rhd_files(self.root)

    def test_set_paths_infers_labels_without_data_root(self):
        self.make_files(["chunk1.rhd", "chunk2.rhd"])
        with patch.object(zsort, "_resolve_data_root", side_effect=AssertionError):
            paths = zsort.set_paths(self.root)
        self.assertEqual(paths["session"], self.root.name)
        self.assertEqual(paths["patient"], self.root.parent.name)
        self.assertEqual(paths["session_location"], self.root)
        self.assertEqual(len(paths["intan_files"]), 2)
        self.assertIsNone(paths["intan_file"])
        self.assertTrue(paths["sorted_data"].is_dir())

    def test_set_paths_legacy_and_explicit_file(self):
        path = self.make_files(["subject/session/raw/chunk.rhd"])[0]
        with patch.object(zsort, "_resolve_data_root", return_value=self.root):
            paths = zsort.set_paths("subject", "session")
        self.assertEqual(paths["intan_file"], path)
        explicit = zsort.set_paths("label", "run", recording_path=path)
        self.assertEqual(explicit["patient"], "label")
        self.assertEqual(explicit["session"], "run")
        self.assertEqual(explicit["recording_source"], path)

    def test_missing_input_creates_no_output(self):
        with self.assertRaises(FileNotFoundError):
            zsort.set_paths(recording_path=self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_lazy_concatenation_order_boundaries_and_provenance(self):
        self.make_files(["chunk2.rhd", "chunk1.rhd"])
        chunks = [self.recording(value=1), self.recording(value=2, samples=3)]
        with patch.object(zsort.si, "read_intan", side_effect=chunks) as read:
            recording = zsort.load_recording({"recording_source": self.root})
        self.assertEqual(recording.get_num_segments(), 1)
        self.assertEqual(recording.get_num_samples(), 7)
        np.testing.assert_array_equal(
            recording.get_traces(start_frame=3, end_frame=5),
            [[1, 1], [2, 2]],
        )
        self.assertEqual([call.args[0].name for call in read.call_args_list], ["chunk1.rhd", "chunk2.rhd"])
        self.assertEqual(recording.get_annotation("intan_source_sample_counts"), [4, 3])
        self.assertEqual(recording.get_annotation("intan_source_start_samples"), [0, 4])
        self.assertEqual(len(recording.get_annotation("intan_source_files")), 2)
        self.assertEqual(recording.get_dtype(), np.dtype("uint16"))
        self.assertEqual(len(list(self.root.iterdir())), 2)
        restored = si.load(recording.to_dict(recursive=True, include_annotations=True))
        self.assertEqual(restored.get_num_samples(), 7)
        self.assertEqual(
            restored.get_annotation("intan_source_files"),
            recording.get_annotation("intan_source_files"),
        )
        np.testing.assert_array_equal(restored.get_traces(), recording.get_traces())

    def test_single_chunk_and_stream_selection(self):
        path = self.make_files(["chunk.rhd"])[0]
        expected = self.recording()
        with patch.object(zsort.si, "read_intan", return_value=expected) as read:
            self.assertIs(zsort.load_recording(path, stream_id="1"), expected)
        read.assert_called_once_with(path, stream_id="1")

    def test_incompatible_chunks_fail_explicitly(self):
        self.make_files(["chunk1.rhd", "chunk2.rhd"])
        changes = [
            ("Sampling frequency", self.recording(rate=20000)),
            ("Channel IDs/order", self.recording(channels=3)),
            ("Recording dtype", self.recording(dtype="int16")),
        ]
        gain = self.recording()
        gain.set_channel_gains([0.2, 0.2])
        changes.append(("gain_to_uV", gain))
        offset = self.recording()
        offset.set_channel_offsets([1, 1])
        changes.append(("offset_to_uV", offset))
        reordered = self.recording().rename_channels([1, 0])
        changes.append(("Channel IDs/order", reordered))
        for message, other in changes:
            with self.subTest(message=message):
                with patch.object(zsort.si, "read_intan", side_effect=[self.recording(), other]):
                    with self.assertRaisesRegex(ValueError, message):
                        zsort.load_recording(self.root)

    def test_empty_multisegment_and_corrupt_inputs_fail(self):
        self.make_files(["chunk.rhd"])
        multisegment = si.NumpyRecording(
            [np.zeros((2, 2)), np.zeros((2, 2))], sampling_frequency=30000,
        )
        for recording, message in [
            (self.recording(samples=0), "no samples"),
            (multisegment, "one segment"),
        ]:
            with patch.object(zsort.si, "read_intan", return_value=recording):
                with self.assertRaisesRegex(ValueError, message):
                    zsort.load_recording(self.root)
        with patch.object(zsort.si, "read_intan", side_effect=ValueError("corrupt header")):
            with self.assertRaisesRegex(ValueError, "corrupt header"):
                zsort.load_recording(self.root)


if __name__ == "__main__":
    unittest.main()
