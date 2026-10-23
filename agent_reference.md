# Agent and maintainer reference

Read [readme.md](readme.md), [environment setup](docs/environment.md), and
[the usability review](docs/usability_review.md) before changing this project.
This document is a navigation/API reference; upstream docs are linked rather
than copied so their version and provenance remain clear.

## Supported baseline and API lookup

The workflow pins SpikeInterface 0.103.0, ProbeInterface 0.3.2, GUI 0.11.0, and
Kilosort 4.1.1. Read installed signatures before borrowing examples from latest
docs. Newer SI releases reorganize metrics and add curation APIs absent here.

| Task | Authoritative reference |
| --- | --- |
| SI API, recording/sorting/analyzer methods, extractors, metrics, widgets | [0.103.0 API](https://spikeinterface.readthedocs.io/en/0.103.0/api.html) |
| Analyzer creation, extensions, persistence | [0.103.0 postprocessing](https://spikeinterface.readthedocs.io/en/0.103.0/modules/postprocessing.html) |
| Remove/merge units, duplicates, automated curation | [0.103.0 curation](https://spikeinterface.readthedocs.io/en/0.103.0/modules/curation.html) |
| Sorter wrappers and parameters | [0.103.0 sorters](https://spikeinterface.readthedocs.io/en/0.103.0/modules/sorters.html) |
| Probe construction, device indices, JSON | [ProbeInterface API (rolling)](https://probeinterface.readthedocs.io/en/main/api.html), [0.3.2 source](https://github.com/SpikeInterface/probeinterface/tree/0.3.2) |
| Web/desktop curation UI and CLI | [GUI docs (rolling)](https://spikeinterface-gui.readthedocs.io/en/latest/), [0.11.0 source](https://github.com/SpikeInterface/spikeinterface-gui/tree/0.11.0) |
| KS4 and output files | [Kilosort guide (rolling)](https://kilosort.readthedocs.io/en/latest/README.html), [4.1.1 source](https://github.com/MouseLand/Kilosort/tree/v4.1.1) |
| Hardware-specific torch installation | [Official installer](https://pytorch.org/get-started/locally/) |
| Classifier artifacts/metadata | [Noise/neural model](https://huggingface.co/SpikeInterface/UnitRefine_noise_neural_classifier), [SUA/MUA model](https://huggingface.co/SpikeInterface/UnitRefine_sua_mua_classifier) |

If a tagged source link changes, use installed metadata/source rather than
assuming the default branch matches these versions. Inspect locally:

```python
import inspect
from importlib.metadata import version
import spikeinterface as si
import spikeinterface.curation as sc
import probeinterface as pi

print({p: version(p) for p in ["spikeinterface", "probeinterface"]})
print(inspect.signature(si.SortingAnalyzer.remove_units))
print(inspect.signature(si.SortingAnalyzer.merge_units))
print(inspect.signature(sc.apply_curation))
print(inspect.signature(pi.read_probeinterface))
print(inspect.signature(sc.load_model))
```

## Project entry points

- `zsort.set_paths(patient, session, *, recording_path=None)`: resolves inputs
  and creates an output directory. One positional path is also supported.
- `zsort.load_recording(paths_or_source, stream_id="0")`: validated, lazy chunk
  concatenation. Uses `get_file.get_rhd_files`; never select only the first chunk.
- `zsort.set_probe(recording, paths, filename_or_absolute_path)`: first probe in
  JSON. Positional device indices must match channel order. Multi-probe handling
  is not generalized; do not assume channel count alone proves correct wiring.
- `zsort.sort(recording, paths)` and `zsort.analyze(recording, sorting, paths)`:
  current function names. `sort_stuff`/`analyze_stuff` do not exist. Read the
  rerun caveat in the README before invoking these on real outputs. Unreadable
  existing outputs now fail closed. See [recovery](docs/sorting_recovery.md) for
  scoped KS4 guards, version restrictions, checkpointing, and resume.
- `kilosort_runtime.kilosort_runtime` and `resume_clustering`: scoped guard and
  trusted local checkpoints; never mutate installed packages or remove failed
  run outputs. Hooks are process-global: one Kilosort run per process.
- `zsort.launch_curation_gui`: use this wrapper for web/IPython compatibility.
- `zsort.save_curated_data`: manual analyzer-folder GUI decisions only; reloads
  from disk even when passed an analyzer. New versioned Zarr destination.
- `auto_curate.AutoCurationConfig`, `auto_curate.auto_curate`, `AutoCurator.run`:
  [curation guide](docs/auto_curation.md) describes outputs and cleanup policy.
- `model_labeling`: independent classifier passes, strict sklearn guard,
  metrics, model provenance, namespaced properties, and GUI extra columns.
- `wrapper.main`: SAW host contract; takes parameters/context, no argparse CLI.
  `curate` needs a previously saved manual GUI JSON; `all` does not open a GUI.
- `box_managment.py` (historical spelling): optional Box CLI, fixed lab folder
  ID/root. Never infer permission to upload/download from a request to review code.

## Scientific invariants

Preserve raw recordings, source ordering, sample counts, channel identities,
gains/offsets, and source annotations. Intan concatenation does not recover
wall-clock gaps. Verify probe wiring separately from geometry; manufacturer
open-contact annotations do not automatically remove channels.

`SortingAnalyzer.remove_units(remove_unit_ids=[...])` takes IDs, not a curation
dictionary. `merge_units(merge_unit_groups=[[...]], merging_mode="hard")`
recomputes extensions; retain the returned analyzer. GUI curation dictionaries
are applied with `sc.apply_curation(...)`. `plot_unit_summary` is in
`spikeinterface.widgets`; compute required analyzer extensions before plotting.

Auto-curation defaults do alter spike trains/unit identities through cleanup
and merging even when filtering is off. Thresholds are examples, not validated
biological truth. Preserve decision tables and original-to-final IDs. Keep human
labels and each classifier's properties separate. Model labeling happens after
merging; do not inherit a source unit's prediction as the merged unit's result.

Do not silently upgrade SI for Bombcell, mutate installed packages to fix GUI
issues, bypass model compatibility guards, invent unavailable model labels, or
report successful curation when its success marker was never written. New model
revisions need metadata/feature compatibility checks and an explicit test.

## Change and validation workflow

1. Inspect `git status`/diff first and preserve unrelated user changes, especially
   notebooks and ongoing curation work. Keep dependency declarations centralized.
2. Use synthetic recordings for tests. Do not run costly GPU sorting or contact
   cloud/Box/model services merely to inspect code. Real HF inference is opt-in.
3. Run `python -m pip check`, then:

   ```bash
   python -m unittest discover -s src/sorting_scripts/tests -v
   python -m unittest discover -s Custom_Probes -p 'test_*.py' -v
   ```

4. Update human instructions when changing paths, parameters, output names,
   defaults, or APIs. Clear notebook outputs containing data before publishing;
   do not rewrite unrelated notebooks just to format them.
5. Report the environment and exact validation scope. Synthetic tests do not
   prove GPU execution, physical wiring, live model accuracy, or archive portability.

`docs/auto_curation_instructions.md` and `docs/hf_noise_prediction_model.md`
are historical implementation briefs. The current code, tests,
`docs/auto_curation.md`, and `docs/model_labeling_audit.md` describe behavior.
No project-wide CI or fully locked cross-platform environment exists yet.
