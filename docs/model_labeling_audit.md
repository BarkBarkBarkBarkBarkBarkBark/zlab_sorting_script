# Model-labeling audit (2026-10-23)

Before this extension, `auto_curate.py` provided an opt-in, single-model adapter
to `model_based_label_units`. No other repository source implemented HF inference.
The notebook selected threshold labeling, not either UnitRefine model. The adapter
required APIs absent in the pinned SpikeInterface 0.103.0 environment, although
that environment provides the supported `load_model`, `auto_label_units`, and
`ModelBasedClassification` APIs. No automatic upgrade is needed.

Deficiencies identified before implementation:

- Neither named UnitRefine model was selected or run by default.
- The single-model adapter could reject missing metric columns, but did not
  initialize both models, compute their required features, or record revisions.
- It persisted only a generic label/confidence and used that label for removal.
  Independent ML labeling and explicit filtering policies were missing.
- There was no neural-only SUA/MUA pass, not-evaluated status, dual-model report,
  or verified visual-review path. The existing web GUI launch does not configure
  custom ML columns; merely storing sorting properties is insufficient. The
  installed GUI does support `extra_unit_properties`, which the new workflow uses.
- Legacy SI classification writes shared `classifier_label` and
  `classifier_probability` properties unconditionally. Running both classifiers
  directly on the working analyzer would overwrite predictions.
- Cleanup and hard merges already precede labeling and have lineage reporting;
  these mechanisms should be reused, not replaced.

The two HF `model_info.json` files currently declare SI 0.102.0 and scikit-learn
1.4.2 training requirements, label conversions, and **empty** `metric_params`.
Consequently, training-parameter equivalence cannot be certified from these
artifacts. Record that limitation explicitly; enforce all supplied parameters,
and never describe unspecified training settings as verified.

Implementation target: default dual inference on final post-merge identities,
separate persisted properties and CSV review table, no ML-based removal unless
an independent policy is explicitly supplied. Existing threshold-based quality
filtering remains separate and configurable. Models trained on mouse Neuropixels
are unvalidated for human DeepArray recordings; all predictions require review.

## Verification findings

Actual HF model downloads succeeded. Both revisions were inspected:

- Noise: `280cd07ff8310286ded5c0cfdb382e92e2a43a98`
- SUA/MUA: `9ee216e7e0e7dbf0778cb07a8d1753700dbbad3b`

Both require 37 feature columns. Actual SI feature computation succeeded on a
synthetic recording. Inference then failed because scikit-learn 1.9.0 cannot run
the 1.4.2-trained `SimpleImputer` (`_fill_dtype` missing). Loading is not proof of
working inference. A pre-load dependency check now prevents that unsupported
combination. No sorter packages were upgraded/downgraded. Deterministic tests
verify dual inference orchestration, persistence/reload, and actual stock Unit
List table columns independently of this external model compatibility blocker.
