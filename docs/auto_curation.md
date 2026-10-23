# Automated curation

The [automatic notebook](../notebooks/z-sort_notebook_w_auto_curate.ipynb)
keeps recording, sorting, and analyzer creation in `zsort`, followed by a small
configuration and one call to [auto_curate](../src/sorting_scripts/auto_curate.py).
No GUI interaction or saved GUI curation file is required.

```python
from sorting_scripts import auto_curate

config = auto_curate.AutoCurationConfig()
result = auto_curate.auto_curate(analyzer, path_dict, config)
curated_data = result.analyzer
result.summary()
```

## What runs

1. Validate configuration and optional labeler availability.
2. Remove within-unit duplicate spikes with stock `remove_duplicated_spikes`
   (`keep_first`, 0.3 ms), then remove empty units.
3. Create a new disk-backed analyzer using the source analyzer's recording,
   scaling, and sparsity. Raw recordings are not copied or rewritten. Compute
   waveforms, templates, noise, amplitudes, locations, histograms, correlograms,
   similarity, and configured quality/template metrics.
4. Remove redundant units with stock `remove_redundant_units`, keeping the
   higher-amplitude unit by default. Alignment is used for comparison, not to
   shift the retained spike trains.
5. Find merge groups using stock `compute_merge_unit_groups`, once, with the
   `similarity_correlograms` preset. Apply with `SortingAnalyzer.merge_units`
   in **hard** mode; recompute rather than approximate waveforms. The default
   merge censor period is 0.3 ms. `merge_steps_params` forwards SI step settings.
6. Recompute metrics and quality labels on the resulting candidates, then run
   independent noise/neural and SUA/MUA inference **after merging**. Both quality
   filtering and ML filtering are disabled by default. There is no iterative merge loop.
7. Save the curated analyzer and audit outputs, optionally exporting a stock
   report and/or Phy directory. Empty final selections are valid and saved;
   visual exports are skipped when no units survive.

`CurationSorting` is not needed: analyzer-level stock operations keep extensions
consistent with the edited sorting. The helper does not implement a sorter,
classifier, duplicate detector, or merge scoring algorithm.

## Policy and metrics

The example labels units using SNR >= 5, firing rate >= 0.1 Hz, and
`isi_violations_ratio` <= 0.5. These are **illustrative starting points**, not
validated criteria for this lab, probe, species, or recording duration. Low-rate
units can be meaningful. An ISI violation ratio is not a percentage.

Threshold keys are **metric output column names**, whereas
`quality_metric_names` takes SI calculation names: for example, `isi_violation`
produces `isi_violations_ratio` and `isi_violations_count`. Unknown/missing columns
raise an error. Bounds use `greater` (>=), `less` (<=), and optional `abs=True`.
NaN and infinite thresholded values fail; no criteria are silently ignored.
An empty threshold dictionary explicitly labels every candidate `good`.
Set `apply_quality_filtering=True` to restore threshold/Bombcell-based selection
using `keep_labels`. This switch is independent of `apply_model_filtering`.

Default metrics also include spike counts, presence ratio, and amplitude cutoff
for inspection, but the latter two are not default rejection criteria. Presence
and amplitude completeness estimates can be unreliable on short recordings or
units with few spikes. SI warnings should be reviewed rather than suppressed.

Use `quality_metric_params` for SI's metric parameter dictionary, and
`template_metric_names` / `template_metric_params` for waveform metrics.
`extension_params` overrides SI extension parameters, e.g. waveform windows,
random spike sampling, location methods, or correlogram resolution.
`random_seed=0` controls default random sampling/noise/quality metrics; explicit
extension overrides can change that seed. `compute_principal_components` and
`compute_spike_locations` enable optional expensive extensions. Drift metrics
automatically request spike locations.

The source analyzer is never changed, so the helper intentionally computes a
fresh analysis rather than trusting cached metrics with potentially different
parameters. Hard merging and optional full-metric labeling may be expensive on
long recordings. Start with `n_jobs=1`, then tune for available memory.

## Version-specific labeling

The repository pins **SpikeInterface 0.103.0**. The instruction document lists
several newer APIs unavailable there. This change does not silently upgrade the
environment or replace installed packages.

- `label_method="threshold"` works on 0.103.0 using SI-computed metrics and the
  documented DataFrame-selection approach. When available, SI's public
  `threshold_metrics_label_units` performs labeling (the current public name,
  rather than the instruction document's `threshold_metrics`).
- `label_method="bombcell"` calls stock `bombcell_label_units`. It requires a
  compatible newer SI installation and computes full quality/template metrics,
  principal components, and spike locations. Pass `bombcell_thresholds` to set
  noise/MUA/non-somatic criteria, or use stock `bombcell_get_default_thresholds`.
  All enabled threshold columns must exist; unavailable metrics raise instead
  of allowing Bombcell to skip tests. Set `keep_labels=("good", "mua")` only if
  that matches the study's policy. Calibrated microvolt scaling matters.
- UnitRefine classification is a separate stage, not `label_method="model"`.
  The old single-model adapter is superseded rather than duplicated. It uses
  stock `load_model` and `ModelBasedClassification.predict_labels`, available in
  the pinned SI. Each classifier gets an isolated in-memory analyzer selection,
  because this SI version unconditionally writes shared classifier properties.
  Only separately named ML properties are copied to the saved result.

## UnitRefine configuration and compatibility

`enable_model_labeling=True` runs the noise/neural classifier on all post-cleanup,
post-merge candidates, then SUA/MUA only on noise-predicted neural units. If the
noise classifier is explicitly disabled, SUA/MUA evaluates all candidates. Empty
populations do not load a model. Non-evaluated predictions are blank, with NaN
confidence and `not_evaluated` status. Neither model overwrites human labels.

Defaults use `SpikeInterface/UnitRefine_noise_neural_classifier` and
`SpikeInterface/UnitRefine_sua_mua_classifier`. `models` overrides repository IDs;
`model_revisions={"noise": "<commit>", "sua_mua": "<commit>"}` pins revisions.
HF snapshots are used from cache first, downloaded when absent, then passed to
SI's supported loader. Actual snapshot commits and artifact SHA-256 are recorded.
`model_trust=True` permits model deserialization: only use repositories you trust.
No arbitrary classifier or custom sklearn inference replaces the SI implementation.

**Current environment blocker (verified October 23, 2026):** both published
artifacts require scikit-learn **1.4.2**, while this repository's environment has
**1.9.0**. Actual inference was attempted and failed in `SimpleImputer` with a
missing `_fill_dtype` attribute. The implementation now detects the version
mismatch **before deserializing or computing expensive model metrics**, raises
an actionable error, and writes failure metadata. It does not patch serialized
estimators or change installed packages. Use a separate environment with the
model's declared sklearn version and compatible NumPy/SciPy; verify its complete
dependency resolution before running the notebook. Do not blindly downgrade the
working sorter environment. To run the existing cleanup/quality workflow without
ML while resolving that environment, set `enable_model_labeling=False`.

`enforce_metric_params=True` reproduces and validates supplied training settings;
explicit conflicts with configured metric parameters raise. Missing supported
features trigger stock metric computation (including PCA, spike locations, and
multi-channel template metrics as required). Unsupported features fail. The
current published models have **empty training metric parameters**: equivalence
cannot be certified. That limitation is recorded as
`training_parameters_available=False`, not represented as successful validation.
Short-recording NaNs remain subject to the model's stock imputation pipeline.

`apply_model_filtering=False` always keeps units regardless of their ML labels.
Predictions are advisory and **unvalidated for human DeepArray**. To opt into a
specific filtering policy, supply independently configurable rules, for example:

```python
config.model_filtering = {
    "noise": {"exclude_labels": ["noise"], "min_probability": 0.95},
    "sua_mua": {"exclude_labels": ["mua"], "min_probability": 0.99},
}
config.apply_model_filtering = True
```

These confidence levels are illustrative, not recommendations. Without the
switch, rules only flag `filter_candidate`; with it, they also set `filter_applied`.
Enabling filtering without a policy raises. Only evaluated units can be filtered.
`persist_labels_to_sorting=False` disables property persistence but retains the CSV;
`preserve_model_probabilities=False` omits numeric confidence properties, not CSV data.
`model_failure_policy="raise"` is the only supported failure policy: no silent success.

## Outputs and review

Each call creates a new `sorted_data/auto_curated_<UTC timestamp>_<suffix>/`:

- `analyzer/`: final binary-folder SortingAnalyzer, directly loadable with
  `spikeinterface.load_sorting_analyzer(result.analyzer_folder)`.
- `analyzer_precuration/`: cleanup-stage analyzer for audit and recovery.
- `config.json`: exact requested settings, written before expensive work.
- `metrics_before.csv`: metrics before redundant-unit removal and merging.
- `metrics_after.csv`, `labels.csv`: every final candidate, including rejected
  units, so rejection decisions remain inspectable.
- `decisions.csv`: one row per original ID, with empty/redundant/kept/merged-kept/
  rejected disposition and post-merge candidate ID/label when applicable.
- `curation.json`: successful curation marker, SI version, config, counts,
  redundant pairs, and source-to-new-ID merge groups.
- `unit_ml_classifications.csv`: independent labels, confidence, status, model
  IDs/revisions, filtering flags, review requirement, and retained status per
  post-merge candidate. Accessible as `result.ml_classifications` in the notebook.
- `model_curation_metadata.json`: model provenance, required metrics, actual
  metric settings, inference timestamps, evaluated/class counts, limitations,
  and failures. Written even when model loading/inference fails.
- `unit_ml_features_noise.csv`, `unit_ml_features_sua_mua.csv`: exact feature
  inputs by evaluated unit ID, written before inference for reproducibility.
- `report/`, `phy/`: optional stock exports. Export errors propagate; the saved
  analyzer and curation audit remain available if an optional export fails.

Original IDs appear in `decisions.csv`; merged IDs may be new SI-generated IDs.
An ID in `final_unit_id` on a rejected row identifies the rejected **candidate**,
not a unit present in the saved analyzer. Redundant and empty units have no
final ID. `input_spikes` and `duplicate_spikes_removed` describe cleanup, not
the number of spikes discarded by subsequent merging/quality selection.

Existing destinations are never overwritten. For an explicit fresh destination:

```python
result = auto_curate.AutoCurator(config).run(analyzer, "/new/output/directory")
```

Partial runs remain for diagnosis; absence of `curation.json` means curation did
not complete. The saved analyzer references its original recording, which must
remain accessible to recompute waveforms or export raw data to Phy. GUI review
is optional via
`zsort.launch_curation_gui(result.analyzer, curation=False, **result.gui_kwargs())`.
The pinned GUI supports `extra_unit_properties`; this explicitly exposes
`ml_noise_label`, `ml_noise_probability`, `ml_sua_label`, `ml_sua_probability`,
status and provenance columns in its Unit List. The helper's arrays come from
the saved sorting in unit-ID order and remain available after analyzer reload
through `model_labeling.model_gui_kwargs(reloaded)`. The CSV is a companion review
table with matching IDs, not an assumption that all custom properties are visible.
Opening a GUI does not automatically apply manual edits to this run.

## Tests

```bash
.venv/bin/python -m unittest discover -s src/sorting_scripts/tests -v
```

Synthetic tests exercise the pinned SI's actual extensions, duplicate removal,
redundancy removal, merge detection, hard merges, export, reload, sparse/string
IDs, empty selections, immutable source analyzer, and decision tables. No lab
recordings are sorted or curated by the test suite.
Dual-model tests use deterministic outputs, real analyzer save/reload and the
stock GUI table builder. They cover filtering, failures, feature computation,
version mismatch, post-merge classification, integer/string IDs, zero units,
all-noise populations, disabled classifiers, and repeat runs. A real-HF integration
test is opt-in; it requires compatible dependencies and may download models.
