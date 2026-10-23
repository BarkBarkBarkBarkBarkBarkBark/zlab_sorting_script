# Task: Audit and Implement Default ML-Based Unit Labeling

## Objective

Inspect the existing SpikeInterface automated curation pipeline and determine whether it already supports Hugging Face model-based unit classification.

If implemented, verify that it works correctly and that classifications are preserved for visual review.

If missing or incomplete, extend the existing implementation.

**Default behavior: Classify and label units, but do not automatically exclude units based on ML predictions.**

Preserve all existing automated merging, duplicate removal, quality-metric calculation, and GUI functionality.

## 1. Audit the existing implementation first

Search the repository for:

- `model_based_label_units`
- `unitrefine_label_units`
- `auto_label_units`
- `load_model`
- `get_required_metrics_from_model`
- `classifier_label`
- `classifier_probability`
- `SortingAnalyzer`
- `compute_quality_metrics`
- `compute_template_metrics`

Determine:

1. Whether Hugging Face classification is implemented.
2. Which models are used.
3. Whether inference actually runs or is merely configured.
4. Whether model loading failures are detected.
5. Whether required metrics are computed.
6. Whether predictions are persisted.
7. Whether predictions are accessible through the GUI.
8. Whether either model overwrites the other's predictions.
9. Whether classifications are currently used to remove units.

Do not duplicate existing functionality.

Document any deficiencies before implementing changes.

## 2. Add configuration parameters

Implement the following configuration, adapting it to the project's existing configuration architecture.

```python
CURATION_CONFIG = {
    "enable_model_labeling": True,
    "apply_model_filtering": False,
    "enable_noise_classifier": True,
    "enable_sua_classifier": True,
    "models": {
        "noise": "SpikeInterface/UnitRefine_noise_neural_classifier",
        "sua_mua": "SpikeInterface/UnitRefine_sua_mua_classifier",
    },
    "model_trust": True,
    "enforce_metric_params": True,
    "preserve_model_probabilities": True,
    "persist_labels_to_sorting": True,
    "model_failure_policy": "raise",
}
```

The default behavior must be non-destructive.

Changing `enable_model_labeling` to `False` should skip model loading and inference without affecting the rest of the curation pipeline.

Changing `apply_model_filtering` to `True` should enable a separate, explicitly configured filtering policy.

Do not equate the model's predicted label with automatic permission to remove a unit.

## 3. Verify compatibility with installed SpikeInterface

Inspect the installed SpikeInterface version.

The API has changed across recent releases, so use the appropriate supported function rather than relying on deprecated imports.

Check the available implementations of:

```python
from spikeinterface.curation import (
    model_based_label_units,
    load_model,
    get_required_metrics_from_model,
)
```

If the installed version uses an alternative API, adapt accordingly.

Inspect signatures using `inspect.signature()` and check the installed documentation.

Do not upgrade SpikeInterface automatically if doing so risks breaking the existing processing pipeline.

## 4. Implement model initialization and validation

At the beginning of the model-labeling stage:

1. Resolve the configured model repository IDs.
2. Load the models using SpikeInterface's supported model loader.
3. Confirm that the returned model objects and metadata are valid.
4. Inspect required quality and template metrics.
5. Check whether the analyzer contains those metrics.
6. Compute missing supported metrics using the model-required parameters when feasible.
7. Reject incompatible metric configurations rather than silently running inference with invalid features.
8. Record the model version and revision.

Models should be downloaded when unavailable locally, provided network access is available.

Use cached models when available.

If the model cannot be loaded and `model_failure_policy="raise"`, raise an informative exception.

Never silently report successful ML curation when inference has not actually run.

## 5. Implement two independent classifiers

### Classifier A: Noise versus neural activity

Model:

`SpikeInterface/UnitRefine_noise_neural_classifier`

Run against all eligible units.

Persist:

- Predicted class
- Prediction confidence
- Model identifier
- Model revision
- Classification status

### Classifier B: SUA versus MUA

Model:

`SpikeInterface/UnitRefine_sua_mua_classifier`

By default, apply the classifier to the population predicted to be neural by Classifier A.

Do not permanently remove the noise-classified units when creating this subset.

Units excluded from the second classification should receive a distinguishable status such as `not_evaluated`, not an invented SUA/MUA prediction.

Preserve both models' predictions.

### Important: Avoid property collisions

SpikeInterface may store model outputs under shared property names:

- `classifier_label`
- `classifier_probability`

Running two classifiers consecutively can overwrite these properties.

Use `set_predictions_as_properties=False` when supported, then persist predictions under unique names.

Suggested sorting properties:

```python
MODEL_PROPERTIES = {
    "noise_label": "ml_noise_label",
    "noise_probability": "ml_noise_probability",
    "sua_label": "ml_sua_label",
    "sua_probability": "ml_sua_probability",
}
```

Use the actual column names returned by the installed API, rather than assuming they are fixed.

Store numeric confidence values and labels separately.

Preserve these properties through supported analyzer-saving operations.

## 6. Make labels available for visual review

This is a primary requirement.

The researcher should be able to examine automatically generated classifications during visual inspection.

Preferred implementation:

Expose the following information through the existing SpikeInterface GUI workflow:

- Noise/neural prediction
- Noise/neural confidence
- SUA/MUA prediction
- SUA/MUA confidence

Investigate whether custom sorting properties can be displayed in the existing GUI's Unit List.

If supported, configure the viewer to expose these columns.

If the current web GUI does not support arbitrary properties, do not assume they are visible merely because they exist in the sorting.

Instead, generate a companion review table or another supported inspection interface that associates the predictions with the GUI's unit IDs.

Do not replace existing user-assigned curation labels.

Keep ML predictions and human decisions as separate fields.

## 7. Preserve labeling through automated merges

Investigate when model classification currently occurs relative to duplicate removal and automated unit merging.

Prefer running classification against the final curated analyzer so predictions correspond to the final unit identities.

If classifications are also useful for determining merge eligibility, implement pre-merge classification as a separate optional stage.

After a merge, do not simply inherit one source unit's classifier prediction.

Recompute model-derived features and rerun classification for the merged unit when possible.

Maintain a source-to-final unit mapping so all decisions remain traceable.

## 8. Default filtering behavior

When:

`apply_model_filtering=False`

The pipeline must:

- Retain all units regardless of model prediction.
- Save both sets of ML labels.
- Expose confidence information.
- Generate a review report.
- Avoid modifying existing human curation labels.

When:

`apply_model_filtering=True`

Require configurable filtering criteria.

Do not introduce hard-coded assumptions about acceptable confidence.

Noise filtering and SUA/MUA filtering should be independently configurable.

For now, default to excluding no units based solely on model inference.

These classifiers were trained on mouse Neuropixels recordings and have not been validated for this human DeepArray dataset.

## 9. Outputs

Generate a machine-readable classification report.

Suggested filename:

`unit_ml_classifications.csv`

Suggested columns:

```text
unit_id
noise_label
noise_probability
sua_mua_label
sua_mua_probability
model_labeling_status
filter_candidate
filter_applied
review_required
```

Also generate:

`model_curation_metadata.json`

Include:

- SpikeInterface version
- Hugging Face repository IDs
- Model revisions
- Required metric names
- Metric parameter configuration
- Model inference timestamps
- Number of units evaluated
- Number classified as noise
- Number classified as neural
- Number classified as SUA
- Number classified as MUA
- Number not evaluated
- Model-loading or inference failures

Do not conflate ML classifications with unit removal counts.

## 10. Tests and acceptance criteria

Add or update tests demonstrating the following:

1. Model labeling runs by default.
2. Disabling model labeling skips both classifiers.
3. Neither classifier overwrites the other's results.
4. Model predictions are mapped to the correct unit IDs.
5. The SUA/MUA model is run on the intended population.
6. Predictions persist after saving and reloading the analyzer.
7. Model-based filtering is disabled by default.
8. A model-loading failure is clearly reported.
9. An analyzer with missing required metrics is handled correctly.
10. The existing automated merging workflow remains functional.
11. The visual review interface can access the predictions.
12. Rerunning the pipeline does not duplicate or corrupt labels.
13. Zero-unit and all-noise edge cases are handled.
14. Merged units have valid post-merge classifications rather than blindly inherited predictions.

Use mock model outputs for deterministic unit tests.

If feasible, add a separate integration test using the actual Hugging Face models.

## Deliverables

1. A brief audit of the existing implementation.
2. The smallest maintainable set of code changes required.
3. Updated configuration defaults.
4. Automated tests.
5. Documentation explaining how to enable or disable classification and filtering.
6. A demonstration that labels are saved and accessible for visual review.

**Do not consider the task complete merely because the models load. The expected outcome is persistent, separately named, reviewable predictions for every eligible final unit.**