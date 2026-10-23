"""Lab orchestration of stock SpikeInterface curation, without changing raw results.

The default threshold workflow supports the repository's SpikeInterface 0.103.0.
Bombcell labeling requires its public API in a newer SI. UnitRefine labeling
uses the stock loaded-model classifier available in the pinned environment.
Thresholds are starting points, not validated biological ground truth.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.curation as curation
import spikeinterface.postprocessing

try:
    import spikeinterface.metrics.quality
    import spikeinterface.metrics.template
except ModuleNotFoundError as error:
    if error.name not in {"spikeinterface.metrics", "spikeinterface.metrics.quality"}:
        raise
    import spikeinterface.qualitymetrics


@dataclass
class AutoCurationConfig:
    """Notebook-facing policy; SI parameters remain ordinary dictionaries.

    Threshold bounds are inclusive, matching SI's threshold labeler. Missing or
    non-finite thresholded values fail. Empty thresholds explicitly keep all units.
    UnitRefine predictions are independent of quality labels and require review.
    """

    label_method: str = "threshold"
    thresholds: dict = field(default_factory=lambda: {
        "snr": {"greater": 5.0},
        "firing_rate": {"greater": 0.1},
        "isi_violations_ratio": {"less": 0.5},
    })
    keep_labels: tuple[str, ...] = ("good",)
    quality_metric_names: tuple[str, ...] = (
        "num_spikes", "firing_rate", "snr", "isi_violation",
        "presence_ratio", "amplitude_cutoff",
    )
    quality_metric_params: dict = field(default_factory=dict)
    template_metric_names: tuple[str, ...] = ()
    template_metric_params: dict = field(default_factory=dict)
    extension_params: dict = field(default_factory=dict)
    compute_principal_components: bool = False
    compute_spike_locations: bool = False
    random_seed: int = 0
    remove_duplicate_spikes: bool = True
    censored_period_ms: float = 0.3
    remove_redundant_units: bool = True
    redundant_kwargs: dict = field(default_factory=lambda: {
        "align": True, "remove_strategy": "highest_amplitude",
        "agreement_threshold": 0.2, "duplicate_threshold": 0.8,
    })
    auto_merge: bool = True
    merge_preset: str = "similarity_correlograms"
    merge_steps_params: dict = field(default_factory=dict)
    merge_censor_ms: float | None = 0.3
    bombcell_thresholds: dict | None = None
    enable_model_labeling: bool = True
    apply_model_filtering: bool = False
    enable_noise_classifier: bool = True
    enable_sua_classifier: bool = True
    models: dict = field(default_factory=lambda: {
        "noise": "SpikeInterface/UnitRefine_noise_neural_classifier",
        "sua_mua": "SpikeInterface/UnitRefine_sua_mua_classifier",
    })
    model_revisions: dict = field(default_factory=dict)
    model_trust: bool = True
    enforce_metric_params: bool = True
    preserve_model_probabilities: bool = True
    persist_labels_to_sorting: bool = True
    model_failure_policy: str = "raise"
    model_filtering: dict = field(default_factory=dict)
    apply_quality_filtering: bool = False
    export_report: bool = False
    export_phy: bool = False
    job_kwargs: dict = field(default_factory=lambda: {
        "n_jobs": 1, "chunk_duration": "1s", "progress_bar": True,
    })


@dataclass
class AutoCurationResult:
    """Saved curated analyzer plus decisions on every original unit.

    ``labels`` and ``metrics_after`` include rejected candidates as well as kept
    units. ``decisions`` maps original IDs through removals and merges. Reload
    the saved result with ``si.load_sorting_analyzer(result.analyzer_folder)``.
    """

    analyzer: object
    output_folder: Path
    metrics_before: pd.DataFrame
    metrics_after: pd.DataFrame
    labels: pd.DataFrame
    decisions: pd.DataFrame
    merge_groups: list
    ml_classifications: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_metadata: dict = field(default_factory=dict)

    def gui_kwargs(self) -> dict:
        """Expose namespaced ML columns in the stock GUI's Unit List."""
        from .model_labeling import model_gui_kwargs

        return model_gui_kwargs(self.analyzer)

    @property
    def analyzer_folder(self) -> Path:
        return self.output_folder / "analyzer"

    def summary(self) -> dict:
        return {
            "input_units": len(self.decisions),
            "kept_units": self.analyzer.get_num_units(),
            "merged_groups": len(self.merge_groups),
            "original_unit_decisions": self.decisions["decision"].value_counts().to_dict(),
            "output_folder": str(self.output_folder),
        }


def _require_api(name):
    function = getattr(curation, name, None)
    if function is None:
        raise RuntimeError(
            f"SpikeInterface {si.__version__} does not provide curation.{name}. "
            "Use label_method='threshold' with the pinned environment, or install "
            "a compatible newer SpikeInterface environment for this labeler."
        )
    return function


def _validate_config(config):
    from .model_labeling import validate_model_config

    validate_model_config(config)
    if config.label_method not in {"threshold", "bombcell"}:
        raise ValueError("label_method must be 'threshold' or 'bombcell'; use enable_model_labeling for ML.")
    if not config.keep_labels:
        raise ValueError("keep_labels must contain at least one label.")
    if config.censored_period_ms < 0 or (
        config.merge_censor_ms is not None and config.merge_censor_ms < 0
    ):
        raise ValueError("Censor periods must be non-negative milliseconds.")
    for metric, bounds in config.thresholds.items():
        if not isinstance(bounds, dict) or set(bounds) - {"greater", "less", "abs"}:
            raise ValueError(f"Invalid threshold bounds for {metric}.")
        lower, upper = bounds.get("greater"), bounds.get("less")
        if lower is None and upper is None:
            raise ValueError(f"Specify greater and/or less for {metric}.")
        if any(value is not None and not np.isfinite(value) for value in (lower, upper)):
            raise ValueError(f"Threshold bounds for {metric} must be finite.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Reversed threshold bounds for {metric}.")
    if config.label_method == "bombcell":
        _require_api("bombcell_label_units")
        if config.bombcell_thresholds is None:
            _require_api("bombcell_get_default_thresholds")
    if {"sorting_or_sorting_analyzer", "extra_outputs"}.intersection(config.redundant_kwargs):
        raise ValueError("redundant_kwargs cannot override the analyzer or extra_outputs.")
    if {"quality_metrics", "template_metrics"}.intersection(config.extension_params):
        raise ValueError("Configure metrics with the dedicated metric fields, not extension_params.")


def _compute_metrics(analyzer, config):
    if analyzer.get_num_units() == 0:
        return pd.DataFrame(index=analyzer.unit_ids)
    extensions = {
        "random_spikes": {"max_spikes_per_unit": 500, "seed": config.random_seed},
        "waveforms": {},
        "templates": {},
        "noise_levels": {"seed": config.random_seed},
        "spike_amplitudes": {},
        "unit_locations": {},
        "isi_histograms": {},
        "correlograms": {"window_ms": 100, "bin_ms": 1},
        "template_similarity": {},
    }
    full_metrics = config.label_method == "bombcell"
    if full_metrics or config.compute_principal_components:
        extensions["principal_components"] = {"n_components": 3, "mode": "by_channel_global"}
    if full_metrics or config.compute_spike_locations or "drift" in config.quality_metric_names:
        extensions["spike_locations"] = {}
    extensions.update(config.extension_params)
    analyzer.compute(extensions, **config.job_kwargs)
    metric_params = {
        "metric_names": None if full_metrics else list(config.quality_metric_names),
        "skip_pc_metrics": not (full_metrics or config.compute_principal_components),
        "seed": config.random_seed,
    }
    if config.quality_metric_params:
        metric_params["metric_params"] = config.quality_metric_params
    analyzer.compute("quality_metrics", **metric_params, **config.job_kwargs)
    metrics = analyzer.get_extension("quality_metrics").get_data().copy()
    if full_metrics or config.template_metric_names:
        analyzer.compute(
            "template_metrics",
            metric_names=None if full_metrics else list(config.template_metric_names),
            **config.template_metric_params,
        )
        metrics = metrics.join(analyzer.get_extension("template_metrics").get_data())
    return metrics


def _check_metric_columns(metrics, names):
    missing = sorted(set(names) - set(metrics.columns))
    if missing:
        raise ValueError(
            f"Required metric columns were not computed: {missing}. "
            "Configure quality_metric_names/template_metric_names and their parameters; "
            "do not silently skip curation criteria."
        )


def _threshold_labels(metrics, thresholds):
    _check_metric_columns(metrics, thresholds)
    passed = pd.Series(True, index=metrics.index)
    reasons = pd.Series("", index=metrics.index)
    for metric, bounds in thresholds.items():
        values = metrics[metric].abs() if bounds.get("abs", False) else metrics[metric]
        accepted = pd.Series(np.isfinite(values), index=metrics.index)
        if bounds.get("greater") is not None:
            accepted &= values >= bounds["greater"]
        if bounds.get("less") is not None:
            accepted &= values <= bounds["less"]
        passed &= accepted
        reasons.loc[~accepted] += metric + ";"
    stock_labeler = getattr(curation, "threshold_metrics_label_units", None)
    if stock_labeler is not None and thresholds:
        labels = stock_labeler(metrics, thresholds=thresholds, nan_policy="fail")
        labels.loc[~passed, "label"] = "noise"
    else:
        labels = pd.DataFrame({"label": np.where(passed, "good", "noise")}, index=metrics.index)
    labels["reason"] = reasons.str.rstrip(";")
    return labels


def _label_units(analyzer, metrics, config):
    if config.label_method == "threshold":
        return _threshold_labels(metrics, config.thresholds)
    if config.label_method == "bombcell":
        thresholds = config.bombcell_thresholds
        if thresholds is None:
            thresholds = _require_api("bombcell_get_default_thresholds")()
        required = [
            metric for section in thresholds.values() for metric, bounds in section.items()
            if bounds.get("greater") is not None or bounds.get("less") is not None
        ]
        _check_metric_columns(metrics, required)
        labels = _require_api("bombcell_label_units")(analyzer, thresholds=thresholds)
    if "label" not in labels or not labels.index.is_unique:
        raise ValueError("The stock labeler must return one 'label' per unit ID.")
    labels = labels.reindex(metrics.index).copy()
    if labels["label"].isna().any():
        raise ValueError("The stock labeler did not label every candidate unit.")
    return labels


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class AutoCurator:
    """Run cleanup, redundancy removal, hard merges, then final unit labeling.

    Every run gets a new directory. The source sorting/analyzer and any manual
    GUI curation are never modified. Hard merges recompute extensions against
    the recording rather than approximating waveforms. This can be expensive.
    """

    def __init__(self, config: AutoCurationConfig | None = None):
        self.config = config if config is not None else AutoCurationConfig()
        _validate_config(self.config)

    def run(self, analyzer, output_folder: str | Path) -> AutoCurationResult:
        config = self.config
        _validate_config(config)
        if not analyzer.has_recording():
            raise ValueError("Automatic curation requires the analyzer's original recording.")
        output_folder = Path(output_folder).expanduser().resolve()
        config_json = json.dumps(asdict(config), default=_json_default, allow_nan=False, indent=2)
        output_folder.mkdir(parents=True, exist_ok=False)
        (output_folder / "config.json").write_text(config_json + "\n")
        original_ids = list(analyzer.unit_ids)
        original_counts = analyzer.sorting.count_num_spikes_per_unit()
        decisions = pd.DataFrame(index=pd.Index(original_ids, name="unit_id"))
        decisions["decision"] = "pending"
        decisions["final_unit_id"] = pd.Series(index=decisions.index, dtype=object)
        decisions["label"] = ""
        sorting = analyzer.sorting
        if config.remove_duplicate_spikes:
            sorting = curation.remove_duplicated_spikes(
                sorting, censored_period_ms=config.censored_period_ms, method="keep_first",
            )
        cleaned_counts = sorting.count_num_spikes_per_unit()
        empty_ids = [unit_id for unit_id in original_ids if cleaned_counts[unit_id] == 0]
        decisions.loc[empty_ids, "decision"] = "empty"
        sorting = sorting.remove_empty_units()
        working = si.create_sorting_analyzer(
            sorting, analyzer.recording, format="binary_folder",
            folder=output_folder / "analyzer_precuration",
            sparse=analyzer.is_sparse(),
            sparsity=si.ChannelSparsity(
                analyzer.sparsity.mask[analyzer.sorting.ids_to_indices(sorting.unit_ids)],
                sorting.unit_ids, analyzer.channel_ids,
            ) if analyzer.is_sparse() else None,
            return_in_uV=analyzer.return_in_uV,
        )
        metrics_before = _compute_metrics(working, config)
        metrics_before.to_csv(output_folder / "metrics_before.csv", index_label="unit_id")
        redundant_pairs = []
        if config.remove_redundant_units and working.get_num_units() > 1:
            retained, redundant_pairs = curation.remove_redundant_units(
                working, extra_outputs=True, **config.redundant_kwargs,
            )
            removed_ids = [unit_id for unit_id in working.unit_ids if unit_id not in retained.unit_ids]
            decisions.loc[removed_ids, "decision"] = "redundant"
            working = working.select_units(retained.unit_ids)
        lineage = {unit_id: [unit_id] for unit_id in working.unit_ids}
        merge_groups = []
        if config.auto_merge and working.get_num_units() > 1:
            groups = curation.compute_merge_unit_groups(
                working, preset=config.merge_preset, resolve_graph=True,
                steps_params=config.merge_steps_params, **config.job_kwargs,
            )
            if groups:
                working, merged_ids = working.merge_units(
                    groups, merging_mode="hard", censor_ms=config.merge_censor_ms,
                    return_new_unit_ids=True, **config.job_kwargs,
                )
                for group, new_id in zip(groups, merged_ids):
                    sources = [source for unit_id in group for source in lineage.pop(unit_id)]
                    lineage[new_id] = sources
                    merge_groups.append({"source_unit_ids": sources, "new_unit_id": new_id})
        metrics_after = _compute_metrics(working, config)
        labels = _label_units(working, metrics_after, config) if working.get_num_units() else pd.DataFrame(
            {"label": pd.Series(dtype=str)}, index=working.unit_ids,
        )
        from .model_labeling import label_final_units, persist_model_labels

        ml_classifications, model_metadata = label_final_units(working, config, output_folder)
        keep_mask = labels["label"].isin(config.keep_labels) if config.apply_quality_filtering else pd.Series(
            True, index=working.unit_ids,
        )
        keep_mask &= ~ml_classifications["filter_applied"]
        ml_classifications["retained"] = keep_mask
        kept_ids = working.unit_ids[keep_mask.reindex(working.unit_ids).to_numpy(dtype=bool)]
        for unit_id, sources in lineage.items():
            keep = bool(keep_mask.loc[unit_id])
            for source in sources:
                decisions.at[source, "decision"] = (
                    "merged_kept" if keep and len(sources) > 1 else "kept" if keep else "rejected"
                )
                decisions.at[source, "final_unit_id"] = unit_id
                decisions.at[source, "label"] = labels.at[unit_id, "label"]
        saved = working.select_units(
            kept_ids, format="binary_folder", folder=output_folder / "analyzer",
        )
        saved.set_sorting_property(
            "auto_curation_label", labels.loc[kept_ids, "label"].to_numpy(dtype=str),
        )
        persist_model_labels(saved, ml_classifications, config)
        if "probability" in labels:
            saved.set_sorting_property(
                "auto_curation_probability", labels.loc[kept_ids, "probability"].to_numpy(),
            )
        result = AutoCurationResult(
            saved, output_folder, metrics_before, metrics_after, labels, decisions, merge_groups,
            ml_classifications, model_metadata,
        )
        metrics_after.to_csv(output_folder / "metrics_after.csv", index_label="unit_id")
        labels.to_csv(output_folder / "labels.csv", index_label="unit_id")
        ml_classifications.to_csv(output_folder / "unit_ml_classifications.csv", index_label="unit_id")
        decisions.to_csv(output_folder / "decisions.csv")
        manifest = {
            "status": "curated", "created_utc": datetime.now(timezone.utc).isoformat(),
            "spikeinterface_version": si.__version__, "config": asdict(config),
            "source_analyzer_folder": str(analyzer.folder) if analyzer.folder else None,
            "original_unit_ids": original_ids, "retained_unit_ids": list(saved.unit_ids),
            "input_spikes": sum(original_counts.values()),
            "duplicate_spikes_removed": sum(original_counts.values()) - sum(cleaned_counts.values()),
            "redundant_pairs": redundant_pairs, "merge_groups": merge_groups,
            "summary": result.summary(),
        }
        (output_folder / "curation.json").write_text(
            json.dumps(manifest, default=_json_default, allow_nan=False, indent=2) + "\n"
        )
        if saved.get_num_units() and config.export_report:
            from spikeinterface.exporters import export_report

            export_report(saved, output_folder / "report", **config.job_kwargs)
        if saved.get_num_units() and config.export_phy:
            from spikeinterface.exporters import export_to_phy

            export_to_phy(
                saved, output_folder / "phy", compute_pc_features=config.compute_principal_components
                or config.label_method == "bombcell", **config.job_kwargs,
            )
        return result


def auto_curate(analyzer, path_dict: dict, config: AutoCurationConfig | None = None):
    """Notebook entry point; allocate a unique run under ``sorted_data``.

    Existing runs are never silently reused after settings change or overwritten.
    For a specific destination use ``AutoCurator(config).run(analyzer, folder)``.
    """
    from uuid import uuid4

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    folder = Path(path_dict["sorted_data"]) / f"auto_curated_{timestamp}_{uuid4().hex[:8]}"
    return AutoCurator(config).run(analyzer, folder)
