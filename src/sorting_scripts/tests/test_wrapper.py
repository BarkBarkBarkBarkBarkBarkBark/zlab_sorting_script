"""Check host integration wiring without recordings, GPU work, or cloud access."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from sorting_scripts import zsort


spec = importlib.util.spec_from_file_location(
    "zlab_wrapper", Path(__file__).resolve().parents[3] / "wrapper.py"
)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class WrapperTests(unittest.TestCase):
    def test_explicit_directory_uses_shared_loader_and_current_api(self):
        paths = {"recording_source": Path("/example/session")}
        with patch.object(zsort, "set_paths", return_value=paths) as set_paths, \
             patch.object(zsort, "load_recording", return_value="recording") as load, \
             patch.object(zsort, "set_probe", return_value="probed") as probe, \
             patch.object(zsort, "sort", return_value="sorting") as sort, \
             patch.object(zsort, "analyze", return_value="analyzer") as analyze, \
             patch.object(wrapper, "_write_output_json", return_value=None):
            result = wrapper.main({}, {
                "patient": "subject", "session": "run",
                "recording_path": "/example/session", "stream_id": "1",
                "probe_json": "probe.json",
            }, Mock())
        set_paths.assert_called_once_with(
            "subject", "run", recording_path="/example/session"
        )
        load.assert_called_once_with(paths, stream_id="1")
        probe.assert_called_once_with("recording", paths, "probe.json")
        sort.assert_called_once_with("probed", paths)
        analyze.assert_called_once_with("probed", "sorting", paths)
        self.assertEqual(result["summary"]["data"]["step"], "sort_analyze")

    def test_invalid_step_fails_before_creating_outputs(self):
        with patch.object(zsort, "set_paths") as set_paths:
            with self.assertRaisesRegex(ValueError, "Unknown processing step"):
                wrapper.main({}, {"step": "typo"}, Mock())
        set_paths.assert_not_called()

    def test_legacy_labels_and_manual_curation_sequence(self):
        paths = {"recording_source": Path("/example/session")}
        with patch.object(zsort, "set_paths", return_value=paths) as set_paths, \
             patch.object(zsort, "load_recording", return_value="recording"), \
             patch.object(zsort, "sort", return_value="sorting"), \
             patch.object(zsort, "analyze", return_value="analyzer"), \
             patch.object(zsort, "save_curated_data", return_value="curated") as curate, \
             patch.object(zsort, "generate_figures") as figures, \
             patch.object(wrapper, "_write_output_json", return_value=None):
            wrapper.main({}, {"patient": "subject", "session": "run", "step": "all"}, Mock())
        set_paths.assert_called_once_with("subject", "run", recording_path=None)
        curate.assert_called_once_with("subject", "run", "analyzer", paths)
        figures.assert_called_once_with("curated", paths)
