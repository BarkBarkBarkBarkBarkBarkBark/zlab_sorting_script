# Usability and maintainability review

Reviewed 2026-10-23. Scope: package/install metadata, Python modules, current
notebook entry points, probe generator, documentation, and synthetic tests.
Existing uncommitted notebook/curation changes were preserved. This is not a
validation of lab data, classifier accuracy, or GPU performance.

## Assessment

The core workflow is usable by someone familiar with SpikeInterface, and the
synthetic curation tests are a strong foundation. The main barriers to shared
lab use were implicit environment assumptions, stale onboarding text, and an
out-of-date integration wrapper. Setup and navigation are improved in this
change. A subsequent crash-recovery fix now preserves unreadable outputs and
adds pre-clustering checkpoints; cached-input identity validation remains open.

## Addressed in this change

| Finding | Change |
| --- | --- |
| Package metadata named `README.md`, but the actual file is `readme.md` | Corrected metadata to work on case-sensitive systems |
| `pip install -e .` installed no runtime dependencies | Declared dependencies and optional extras in `pyproject.toml` |
| Requirements mixed direct pins and implicit UI/model dependencies | Requirements select centralized extras; retained four workflow pins and documented Python range |
| Setup assumed an AWS `sorter` conda environment | Added local venv steps, Windows alternatives, CPU review, torch/CUDA selection, and kernel checks |
| Current model training requires an older sklearn | Added explicit optional model environment; preserved strict loader guard and documented uncertainty |
| README had obsolete paths and incorrect remove/merge/widget guidance | Replaced with current entry points, output names, GUI save sequence, automated policy, and links |
| Wrapper called nonexistent `sort_stuff`/`analyze_stuff` | Uses current functions; regression tests cover host wiring |
| Wrapper resolved legacy data before honoring an explicit input and did not load all chunks | Passes explicit source into `set_paths` and uses shared validated `load_recording` |
| Unknown wrapper steps could return a misleading success summary | Reject invalid steps before creating outputs |
| Agents had no entry point or version-aware API map | Added `AGENTS.md` and `agent_reference.md` |
| New environments/output directories could appear in Git | Added ignore patterns for alternate venvs, builds, notebook state, and named session outputs |

## Remaining findings, ordered by impact

### Addressed after review: unreadable outputs triggered destructive recreation

The original helpers treated any output-loading failure as permission to rerun
with destructive overwrite flags. The subsequent [crash-recovery change](sorting_recovery.md)
now preserves unreadable sorter/analyzer folders and raises instead. New runs
also save pre-clustering checkpoints and guard the Kilosort 4.1.1 zero-duration
CCG bug without modifying installed packages.

Remaining: readable cached outputs are still reused without input/probe/settings
identity validation. Record source identity and reject incompatible reuse in a
future change; choose a new destination when changing run inputs/settings.

### Medium: run configuration is spread across code and notebooks

`sort()` hardcodes preprocessing and KS4 invocation. `analyze()` hardcodes metric
extensions and `n_jobs=-1`, which can oversubscribe shared workstations. Current
notebooks contain a personal absolute data path and dataset-specific labels.
The auto-curation configuration is more structured than the sorting stage.

Recommended: a small explicit run configuration for source, probe, stream,
sorter parameters, job limits, and output root, saved alongside results. Add a
clean minimal example notebook with unexecuted, portable configuration. Verify
the interaction of wrapper preprocessing and KS4's internal preprocessing before
changing either; this review makes no scientific preprocessing change.

### Medium: probe assignment needs stronger identity checks

`set_probe()` takes the first probe from a JSON, returns early if a probe is
already attached, and checks contact count after attachment. Matching channel
count alone cannot validate wiring; the early return can hide a requested probe
change. Deep Array open-contact annotations are informational only.

Recommended: validate requested probe identity, channel mapping, recording
channel IDs, and intended exclusions explicitly; raise for ambiguous/multi-probe
files until supported. Preserve the current documented requirement for human
wiring verification.

### Medium: model defaults add external compatibility requirements

Automated curation enables model labeling by default. Its live model metadata
may require a different sklearn from the broad default install; model failure
raises and leaves a partial run. The optional model constraints improve setup,
but training SI 0.102.0 and runtime SI 0.103.0 still require an inference test.
Floating model revisions reduce reproducibility. See the
[existing model audit](model_labeling_audit.md).

Recommended: pin reviewed model commit IDs for lab runs and preflight versions,
cache/network availability, and required feature parameters before expensive
cleanup. Model-free operation is now documented. Do not equate labels with
permission to discard units, or disable failure checks to make a run appear to
succeed.

### Medium: saved analyzers are not guaranteed portable archives

Analyzers reference source recordings. A copied analyzer can still need the
original absolute path to recompute extensions or export raw traces. The manual
save helper also ignores the supplied analyzer in favor of reloading its fixed
path; it is unsuitable for saving GUI edits on an arbitrary automated result.

Recommended: document/test the lab's actual archive/move-and-reload procedure,
include provenance and stable relative paths where supported, and generalize
manual save to accept explicit analyzer/curation/output locations.

### Lower: optional Box helper remains lab-specific

[box_managment.py](../src/sorting_scripts/box_managment.py) has a fixed folder ID
and root, assumes a separately authenticated CLI, and splits CSV with `split(',')`.
Names containing commas or quoting can be parsed incorrectly. Its upload path
expects `session/sorted`, while current `set_paths` produces named `*-sorted`
directories, so current outputs may be skipped. Cloud transfer was not exercised.

Recommended: use `csv.DictReader`, parameterize roots/IDs, accept explicit result
paths, and add a list-only/dry-run preview before transfers.

### Lower: maintenance conventions need a shared enforcement point

No CI currently enforces fresh installation and synthetic tests. Python
3.10–3.12 and multiple OSs are declared targets, not a tested matrix. Older
notebooks and implementation briefs can be mistaken for current instructions.
The repository has no visible license file or formal contribution guide.

Recommended: start with a Linux Python 3.12 CI install/test job, then add other
supported interpreters/OSs; periodically run a separate small GPU acceptance
recording and opt-in pinned-model inference. Add contribution/issue guidance,
label or archive legacy notebooks, clear published outputs, and have the lab
choose a license before broader distribution.

## Validation completed

- Existing Linux Python 3.12.3 environment: 44 tests discovered, 43 passed and
  the opt-in real-model integration skipped; all 6 probe tests passed.
- Fresh isolated Python 3.12.3 venv: installed `.[gui,notebook,unitrefine]`, then
  repeated the same 44-test suite (43 passed, 1 opt-in skip) and 6 probe tests.
  This included the real web GUI/IPython table construction regression.
- Fresh resolved versions included NumPy 1.26.4, SciPy 1.17.1, sklearn 1.4.2,
  SI 0.103.0, ProbeInterface 0.3.2, GUI 0.11.0, Panel 1.10.0, Param 2.4.2,
  and skops 0.16.0. Dependency consistency checks passed in both environments.
- The full `requirements-unitrefine.txt`, including Kilosort4 and torch,
  resolved successfully in a dry run. GPU packages were not installed or run in
  the temporary environment. No real models were downloaded or evaluated.
- `git diff --check` passed. Existing user changes were retained; generated
  tracked egg-info changes from editable build verification were restored.

The isolated setup was exercised with uv's pip-compatible installer; the guide
uses standard `python -m pip` commands. There were upstream deprecation/resource
warnings during synthetic analysis, with no test failures. These checks do not
establish physical channel mapping, live-model accuracy, full KS4 GPU execution,
Python 3.10/3.11, or Windows/macOS compatibility. The original lab environment and
raw data were not changed.
