# Z-Lab spike sorting

Turn Intan `.rhd` recordings into Kilosort4 sortings and curated SpikeInterface
SortingAnalyzers. Run locally or on a lab GPU workstation/AWS instance; cloud
accounts and Box are optional. Start with the manual workflow below.

## Start here

1. Follow [environment setup](docs/environment.md) to create a local `.venv`.
   Use Python **3.12**; the declared supported range is **3.10–3.12**.
2. Open this repository in VS Code and select the **Z-Lab sorting** notebook
   kernel, or run `python -m jupyterlab` in the activated environment.
3. Open [the manual notebook](notebooks/z-sort_notebook.ipynb). Replace its
   example patient/session labels and absolute recording path before running.
4. Verify channel order, attach the correct probe, sort, analyze, and curate.

The environment guide includes Linux/macOS and Windows activation, CPU-only
review, CUDA/PyTorch selection, optional UnitRefine models, and smoke checks.
An NVIDIA GPU with a working CUDA-enabled PyTorch installation is the lab's
sorting target. CPU machines can inspect results, run tests, and use the web GUI;
macOS/MPS is not a validated substitute for the CUDA sorting workflow.

## Minimal manual workflow

Run from the repository checkout with its environment active. Replace the example
path with a directory of chunks from **one continuous acquisition**, or one `.rhd`
file. Do not collect unrelated sessions in that directory.

```python
from sorting_scripts import zsort

paths = zsort.set_paths(recording_path="/absolute/path/to/subject/session")
print(paths["intan_files"])  # Check every file and its order before sorting.
recording = zsort.load_recording(paths, stream_id="0")
recording = zsort.set_probe(
    recording, paths, "deeparray128_P1925-127_H1925-116.json"
)
# Verify this probe's contact positions AND wiring match this recording first.
sorting = zsort.sort(recording, paths)
analyzer = zsort.analyze(recording, sorting, paths)
```

`stream_id="0"` is the default, not a guarantee that this is the correct Intan
stream. Inspect available streams with
`from spikeinterface.extractors import get_neo_streams`, followed by
`get_neo_streams("intan", file_path=paths["intan_files"][0])` when needed. All chunks must have compatible
channels, sample rates, dtype, and gains/offsets. Files are naturally sorted and
concatenated lazily; no merged `.rhd` is written. Inter-file gaps are **not**
inferred or filled; the resulting sample clock starts at zero. Discovery checks
only the supplied directory or its `raw/` subdirectory, not arbitrary descendants.

For a shared data layout, set `SORTING_DATA_ROOT` to an existing directory and use
`zsort.set_paths("subject", "session")`. Expected inputs are
`<root>/subject/session/*.rhd` or `<root>/subject/session/raw/*.rhd`. Explicit paths
avoid dependence on environment variables and legacy directory fallbacks. A file
input writes outputs beside that file; use a session directory when possible.

### Inspect and save manual curation

```python
window = zsort.launch_curation_gui(analyzer, port=5007)
```

Open the URL printed by the GUI. On a remote machine, forward port 5007 in
VS Code's Ports view or through SSH, then open it locally. Stop the server with
Ctrl+C (or interrupt the notebook cell) when finished. Mark removals/merges and
**click Save in the GUI** before running:

```python
curated = zsort.save_curated_data(
    paths["patient"], paths["session"], analyzer, paths
)
zsort.generate_figures(curated, paths)
```

Use the helper for notebook web launches: it works around the GUI 0.11.0 / Param
IPython `SelectableTabulator` docstring error and restores the Param setting.
`port`, `start_app=False`, and other GUI options are forwarded to upstream.
Saving GUI decisions alone does not apply them to the analyzer. The save helper
reloads `paths["analyzer_folder"]`, applies its
`spikeinterface_gui/curation_data.json`, and writes a new curated Zarr analyzer.
It is intended for this manual workflow, not for arbitrary automated analyzers.

### Outputs and reruns

For subject `patient1`, session `run1`, outputs are inside
`<session>/patient1-run1-sorted/`:

| Output | Purpose |
| --- | --- |
| `patient1-run1-sorter_folder/` | Kilosort4 sorting and sorter logs |
| `patient1-run1-analyzer_folder/` | Binary-folder analyzer and GUI decisions |
| `patient1-run1-curated_analyzer.zarr/` | Applied manual curation; later saves use `-v2`, `-v3`, … |
| `figs/` | Per-unit PNG summaries; matching filenames are replaced on rerun |
| `auto_curated_<timestamp>_<suffix>/` | Separate optional automated run and audit tables |

Keep original recordings accessible: saved analyzers can refer to their source
paths and are not guaranteed to be standalone raw-data archives. Reload with
`spikeinterface.load_sorting_analyzer("/path/to/analyzer")`.

**Current rerun behavior:** `sort()` and `analyze()` reuse readable results.
If loading an existing folder fails, they stop and preserve it. Readable cached
results are still not checked against input/probe/settings changes; choose new
output paths when those change. Sorting now guards the Kilosort 4.1.1 zero-span
CCG bug, saves a pre-clustering checkpoint by default, and retains its binary
recording. See [crash protection and resume instructions](docs/sorting_recovery.md)
for retry paths, disk use, and recovery of future runs.
For CUDA out-of-memory failures, `sort(..., batch_size=15000, clear_cache=True)`
provides an explicit smaller-batch retry; use new output folders as described
in that guide. Defaults remain 60000 samples and no forced cache clearing.

## Optional automated curation

Before expensive processing, check the configured classifiers with
`auto_curate.preflight(curation_config)`. This loads enabled models and catches
access/version/loading errors without computing recording metrics. Curation also
runs this check automatically before cleanup; see the guide below for its scope.

Use [the automated notebook](notebooks/z-sort_notebook_w_auto_curate.ipynb) and
[the curation guide](docs/auto_curation.md). Defaults remove duplicate spikes,
remove redundant units, and merge candidates. Quality and ML **filtering** are
disabled by default, but cleanup and merging still change units.

The default also enables UnitRefine labeling, which may download models and
requires their training-compatible environment. For an offline, model-free run:

```python
from sorting_scripts import auto_curate

config = auto_curate.AutoCurationConfig(enable_model_labeling=False)
result = auto_curate.auto_curate(analyzer, paths, config)
result.summary()
zsort.launch_curation_gui(result.analyzer, curation=False, **result.gui_kwargs())
```

`result.analyzer_folder` contains the saved automated analyzer. Inspect the
configuration and decision tables before accepting results. The models were
trained on mouse Neuropixels recordings and are unvalidated for human DeepArray
data. Bombcell requires newer SpikeInterface APIs and is unavailable in the
pinned baseline; do not upgrade the shared environment to enable it.

## Repository map and maintenance

| Location | Use |
| --- | --- |
| `src/sorting_scripts/zsort.py` | Paths, Intan loading, probe, sorting, analysis, manual curation |
| `src/sorting_scripts/auto_curate.py` | Automated policy and decision audit |
| `src/sorting_scripts/model_labeling.py` | Independent UnitRefine labeling and provenance |
| `Custom_Probes/` | Probe JSON, Deep Array generator, geometry tests |
| `wrapper.py` | SAW host integration (`main(inputs, params, context)`); not a standalone CLI |
| `notebooks/z-sort_notebook*.ipynb` | Current manual and automated workflows |
| Other notebooks | Exploratory/legacy examples; may contain machine-specific paths |
| `docs/` | Setup, curation, AWS notes, and review findings |
| [agent_reference.md](agent_reference.md) | Code map, versioned upstream APIs, maintenance rules |

From the repository root:

```bash
python -m pip check
python -m unittest discover -s src/sorting_scripts/tests -v
python -m unittest discover -s Custom_Probes -p 'test_*.py' -v
```

Tests use synthetic recordings and mocks. The real Hugging Face integration is
opt-in (`ZLAB_RUN_HF_INTEGRATION=1`); it downloads models and needs compatible
dependencies. Tests do not establish full GPU sorting or real-data validity.
Report issues with OS, Python/package versions, notebook step, traceback, and a
minimal synthetic reproduction when possible. Avoid committing recordings,
credentials, generated outputs, or notebook outputs containing lab data.

[Demo Intan data](https://data.mendeley.com/datasets/w767nnk5wh/1) is available
for a separate end-to-end trial. The optional Box helpers require a separately
installed/authenticated Box CLI and currently contain lab-specific folder IDs;
review them before transferring data. Existing [AWS SSH notes](docs/ssh_into_aws_machine.md)
and [instance notes](docs/start_ec2_instance.md) are infrastructure references,
not prerequisites for local use.

## Probe reference

### Deep Array 128L probe

[Generator](Custom_Probes/create_deeparray128_probe.py) and
[ProbeInterface JSON](Custom_Probes/deeparray128_P1925-127_H1925-116.json)
describe Diagnostic Biochips probe P1925-127 / holder H1925-116.
The generator requires NumPy and [ProbeInterface](https://probeinterface.readthedocs.io/en/main/api.html);
the repository requirements include ProbeInterface 0.3.2.

Dimensions come from page 16, detail B, of the local
`02-0008-01 Handling Guidance, DA128L_RevA.pdf`: 128 circular sites, 16 um
diameter, two 64-site columns separated by 43 um, 50 um pitch within each
column, and 25 um axial stagger. The combined center-to-center span is 3175 um.
This is a 2D drawing-plane representation of the wrapped probe, not a
reconstructed 3D cylindrical surface. No undocumented shank outline is invented.

The local `20260925_P1925-127_H1925-116.pdf` supplies the numbered layout:
left 127 to 64 and right 0 to 63, proximal to distal. Coordinates use x=0 for
left, x=43 um for right, and positive y toward the connector. Distal left ch64
is (0, 0); distal right ch63 is (43, 25); proximal right ch0 is (43, 3175).
Contacts, contact IDs, and device indices are ordered by printed channel 0-127.

**Before sorting, verify the recording's actual channel order against the
printed channel numbers.** Device indices are positional, not matched to
Intan channel names. If channels have been reordered, selected, or combined
across headstages, update the wiring accordingly before attaching the probe.
The report marks 108 channels good and 20 open; these are stored in the
`manufacturer_status` contact annotation. All 128 sites remain in the JSON.
Open channels are **not** automatically removed or disconnected; review the
recording and explicitly exclude appropriate channels before sorting.

From the repository root, using the existing sorting environment:

```bash
python Custom_Probes/create_deeparray128_probe.py
python -m unittest discover -s Custom_Probes -p 'test_deeparray128_probe.py'
```

The default JSON destination is beside the script, independent of the working
directory; `--output /path/to/probe.json` selects another destination.

```python
from probeinterface import read_probeinterface

probe = read_probeinterface(
    "Custom_Probes/deeparray128_P1925-127_H1925-116.json"
).probes[0]
# Only after verifying recording order corresponds to printed channels 0-127:
recording = recording.set_probe(probe)
```
