import subprocess
import sys
import unittest
from unittest.mock import patch

import param.parameterized as parameterized

from sorting_scripts import zsort


class CurationGuiTests(unittest.TestCase):
    def test_web_launcher_forwards_arguments_and_restores_setting(self):
        previous = parameterized.docstring_describe_params
        analyzer = object()
        with patch("spikeinterface_gui.run_mainwindow", return_value="window") as launch:
            result = zsort.launch_curation_gui(analyzer, start_app=False, port=5007)
        self.assertEqual(result, "window")
        launch.assert_called_once_with(
            analyzer=analyzer, mode="web", curation=True, start_app=False, port=5007,
        )
        self.assertEqual(parameterized.docstring_describe_params, previous)

    def test_failed_import_propagates_and_restores_setting(self):
        for original in (True, False):
            with self.subTest(original=original):
                with patch.object(parameterized, "docstring_describe_params", original):
                    with patch("importlib.import_module", side_effect=ImportError("test failure")):
                        with self.assertRaisesRegex(ImportError, "test failure"):
                            zsort.launch_curation_gui(object())
                    self.assertEqual(parameterized.docstring_describe_params, original)

    def test_desktop_launcher_does_not_apply_web_workaround(self):
        with patch("spikeinterface_gui.run_mainwindow") as launch:
            with patch("importlib.import_module", side_effect=AssertionError("unexpected import")):
                zsort.launch_curation_gui("analyzer", mode="desktop", curation=False)
        launch.assert_called_once_with(analyzer="analyzer", mode="desktop", curation=False)

    def test_ipython_gui_table_construction(self):
        code = """
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
result = shell.run_cell('''
import param.parameterized as parameterized
from unittest.mock import patch
from sorting_scripts import zsort
previous = parameterized.docstring_describe_params
with patch("spikeinterface_gui.run_mainwindow"):
    zsort.launch_curation_gui(object(), start_app=False)
assert parameterized.docstring_describe_params == previous
import pandas as pd
from spikeinterface_gui.utils_panel import SelectableTabulator
table = SelectableTabulator(pd.DataFrame({"unit_id": [1, 2]}))
assert len(table.value) == 2
assert "selection" in table.param.objects()
''')
if result.error_in_exec:
    raise result.error_in_exec
if result.error_before_exec:
    raise result.error_before_exec
print("IPYTHON_GUI_TABLE_OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("IPYTHON_GUI_TABLE_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
