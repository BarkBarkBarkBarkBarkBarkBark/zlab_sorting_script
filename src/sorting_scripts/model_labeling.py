"""Independent UnitRefine labeling using SpikeInterface's loaded-model API."""

from datetime import datetime, timezone
import hashlib
import inspect
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.curation as curation
import spikeinterface.postprocessing as postprocessing


MODEL_PROPERTIES = {
    "noise_label": "ml_noise_label",
    "noise_probability": "ml_noise_probability",
    "sua_mua_label": "ml_sua_label",
    "sua_mua_probability": "ml_sua_probability",
    "noise_status": "ml_noise_status",
    "sua_mua_status": "ml_sua_status",
    "noise_model_id": "ml_noise_model_id",
    "noise_model_revision": "ml_noise_model_revision",
    "sua_mua_model_id": "ml_sua_model_id",
    "sua_mua_model_revision": "ml_sua_model_revision",
}


def validate_model_config(config):
    if config.model_failure_policy != "raise":
        raise ValueError("model_failure_policy currently supports only 'raise'.")
    if config.apply_model_filtering and not config.enable_model_labeling:
        raise ValueError("Model filtering requires enable_model_labeling=True.")
    if config.apply_model_filtering and not config.model_filtering:
        raise ValueError("apply_model_filtering requires an explicit model_filtering policy.")
    if config.enable_model_labeling:
        for name, enabled in (("noise", config.enable_noise_classifier), ("sua_mua", config.enable_sua_classifier)):
            if enabled and (not isinstance(config.models.get(name), str) or not config.models[name].strip()):
                raise ValueError(f"Configure a Hugging Face repository ID for {name}.")
    for name, rule in config.model_filtering.items():
        if name not in {"noise", "sua_mua"} or set(rule) != {"exclude_labels", "min_probability"}:
            raise ValueError("Each model_filtering rule needs exclude_labels and min_probability.")
        if not rule["exclude_labels"] or not 0 <= rule["min_probability"] <= 1:
            raise ValueError("Filtering needs excluded labels and a probability between zero and one.")
        if config.apply_model_filtering and not (
            config.enable_noise_classifier if name == "noise" else config.enable_sua_classifier
        ):
            raise ValueError(f"Filtering requires the {name} classifier to be enabled.")


def _load_classifier(repo_id, revision, trust_model):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from sklearn.pipeline import Pipeline

    options = {"repo_id": repo_id, "revision": revision or "main", "allow_patterns": ["*.json", "*.skops"]}
    try:
        folder = Path(snapshot_download(**options, local_files_only=True))
    except LocalEntryNotFoundError:
        folder = Path(snapshot_download(**options))
    if not (folder / "model_info.json").is_file() or not list(folder.glob("*.skops")):
        folder = Path(snapshot_download(**options))
    raw_info = json.loads((folder / "model_info.json").read_text())
    trained_sklearn = raw_info.get("requirements", {}).get("scikit-learn")
    installed_sklearn = version("scikit-learn")
    if trained_sklearn and trained_sklearn != installed_sklearn:
        raise RuntimeError(
            f"{repo_id} was trained with scikit-learn {trained_sklearn}, but this environment has "
            f"{installed_sklearn}. Use a separate model-compatible environment; no packages were changed. "
            "Cross-version sklearn model loading is unsupported."
        )
    model, info = curation.load_model(model_folder=folder, trust_model=trust_model)
    if not isinstance(model, Pipeline):
        raise ValueError("SpikeInterface model loading did not return an sklearn Pipeline.")
    if not isinstance(info, dict) or not isinstance(info.get("metric_params"), dict):
        raise ValueError("Model metadata must include a metric_params dictionary.")
    required = list(getattr(model, "feature_names_in_", []))
    if not required or not all(isinstance(name, str) for name in required):
        raise ValueError("Loaded model has no valid feature_names_in_.")
    files = sorted(folder.glob("*.skops"))
    if len(files) != 1:
        raise ValueError("Expected exactly one model artifact per classifier repository.")
    provenance = {
        "repo_id": repo_id, "revision": folder.name,
        "sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "requirements": info.get("requirements", {}),
        "required_metrics": required, "training_metric_params": info["metric_params"],
        "training_parameters_available": bool(info["metric_params"]),
    }
    return model, info, provenance


def _preload_classifiers(config):
    """Check every enabled model before cleanup, including conditional SUA use."""
    validate_model_config(config)
    loaded = {}
    if not config.enable_model_labeling:
        return loaded
    for name, enabled in (("noise", config.enable_noise_classifier), ("sua_mua", config.enable_sua_classifier)):
        if not enabled:
            continue
        try:
            loaded[name] = _load_classifier(config.models[name], config.model_revisions.get(name), config.model_trust)
        except Exception as error:
            raise RuntimeError(f"UnitRefine {name} preflight failed: {error}") from error
    return loaded


def _metric_frame(analyzer):
    frames = [analyzer.get_extension(name).get_data() for name in ("quality_metrics", "template_metrics")
              if analyzer.get_extension(name) is not None]
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=analyzer.unit_ids)


def _prepare_metrics(analyzer, required, info, config):
    try:
        from spikeinterface.metrics import quality as qualitymetrics
        from spikeinterface.metrics.quality.quality_metric_list import qm_compute_name_to_column_names
    except ModuleNotFoundError:
        import spikeinterface.qualitymetrics as qualitymetrics
        from spikeinterface.qualitymetrics.quality_metric_list import qm_compute_name_to_column_names

    missing = set(required) - set(_metric_frame(analyzer).columns)
    template_names = set(postprocessing.get_template_metric_names())
    quality_by_column = {column: name for name, columns in qm_compute_name_to_column_names.items()
                         for column in columns}
    unsupported = missing - template_names - set(quality_by_column)
    if unsupported:
        raise ValueError(f"Model requires unsupported metric columns: {sorted(unsupported)}")
    training = info["metric_params"]
    for extension_name, training_key, configured in (
        ("quality_metrics", "quality_metric_params", config.quality_metric_params),
        ("template_metrics", "template_metric_params", config.template_metric_params.get("metric_params", {})),
    ):
        expected = training.get(training_key, {})
        if not isinstance(expected, dict):
            raise ValueError(f"Invalid training parameters for {extension_name}.")
        expected_metric_params = expected.get("metric_params", {})
        if config.enforce_metric_params:
            for name, params in expected_metric_params.items():
                if name in configured and configured[name] != params:
                    raise ValueError(f"Configured {name} parameters conflict with the model training parameters.")
        needed = missing & (template_names if extension_name == "template_metrics" else set(quality_by_column))
        if not needed and not expected:
            continue
        current = analyzer.get_extension(extension_name)
        params = dict(current.params) if current is not None else {}
        names = set(params.get("metric_names") or [])
        names.update(needed if extension_name == "template_metrics" else (quality_by_column[name] for name in needed))
        names.update(expected.get("metric_names") or [])
        params.update(expected)
        params["metric_names"] = sorted(names)
        params["metric_params"] = {**params.get("metric_params", {}), **expected_metric_params}
        if extension_name == "quality_metrics":
            pc_names = set(qualitymetrics.get_quality_pca_metric_list())
            if names & pc_names:
                if analyzer.get_extension("principal_components") is None:
                    analyzer.compute("principal_components", n_components=3, mode="by_channel_global", **config.job_kwargs)
                params["skip_pc_metrics"] = False
            if "drift" in names and analyzer.get_extension("spike_locations") is None:
                analyzer.compute("spike_locations", **config.job_kwargs)
            analyzer.compute(extension_name, **params, **config.job_kwargs)
        else:
            analyzer.compute(extension_name, **params)
        if config.enforce_metric_params and expected:
            actual = analyzer.get_extension(extension_name).params
            for key, value in expected.items():
                if key == "metric_names":
                    continue
                if key == "metric_params":
                    if any(actual[key].get(name) != setting for name, setting in value.items()):
                        raise ValueError(f"Cannot reproduce training parameters for {extension_name}.")
                elif actual.get(key) != value:
                    raise ValueError(f"Cannot reproduce model parameter {key} for {extension_name}.")
    frame = _metric_frame(analyzer)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Required model metrics were not computed: {sorted(missing)}")
    return frame, {name: analyzer.get_extension(name).params for name in ("quality_metrics", "template_metrics")
                   if analyzer.get_extension(name) is not None}


def _predict(analyzer, model, info, config):
    from spikeinterface.curation.model_based_curation import ModelBasedClassification

    classifier = ModelBasedClassification(analyzer, model)
    options = {"model_info": info, "enforce_metric_params": config.enforce_metric_params}
    if "set_predictions_as_properties" in inspect.signature(classifier.predict_labels).parameters:
        options["set_predictions_as_properties"] = False
    predictions = classifier.predict_labels(**options)
    label_column = next((name for name in ("prediction", "label", "classifier_label") if name in predictions), None)
    probability_column = next((name for name in ("probability", "confidence", "classifier_probability") if name in predictions), None)
    if label_column is None or probability_column is None or not predictions.index.is_unique:
        raise ValueError("Stock classifier did not return uniquely indexed labels and probabilities.")
    if set(predictions.index) != set(analyzer.unit_ids):
        raise ValueError("Stock classifier predictions do not match the evaluated unit IDs.")
    predictions = predictions.loc[analyzer.unit_ids, [label_column, probability_column]].copy()
    predictions.columns = ["label", "probability"]
    if predictions["label"].isna().any() or not predictions["probability"].between(0, 1).all():
        raise ValueError("Stock classifier returned missing labels or invalid probabilities.")
    return predictions


def label_final_units(analyzer, config, output_folder, *, loaded_models=None):
    """Label post-merge units without changing the analyzer or human properties.

    Legacy SI writes shared properties; each pass therefore runs on its own
    in-memory unit selection. Only namespaced results leave that selection.
    """
    table = pd.DataFrame(index=pd.Index(analyzer.unit_ids, name="unit_id"))
    for name in ("noise", "sua_mua"):
        table[f"{name}_label"] = ""
        table[f"{name}_probability"] = np.nan
        table[f"{name}_status"] = "disabled" if not config.enable_model_labeling else "not_evaluated"
        table[f"{name}_model_id"] = ""
        table[f"{name}_model_revision"] = ""
    table["model_labeling_status"] = "disabled" if not config.enable_model_labeling else "not_evaluated"
    table["filter_candidate"] = False
    table["filter_applied"] = False
    table["review_required"] = True
    metadata = {"spikeinterface_version": si.__version__, "models": {}, "failures": [],
                "configured_models": config.models, "requested_revisions": config.model_revisions,
                "enforce_metric_params": config.enforce_metric_params,
                "apply_model_filtering": config.apply_model_filtering,
                "enabled": config.enable_model_labeling,
                "warning": "Mouse Neuropixels models are unvalidated for human DeepArray data."}
    try:
        for name, enabled in (("noise", config.enable_noise_classifier), ("sua_mua", config.enable_sua_classifier)):
            if not config.enable_model_labeling or not enabled:
                table[f"{name}_status"] = "disabled"
                continue
            table[f"{name}_model_id"] = config.models[name]
            table[f"{name}_model_revision"] = config.model_revisions.get(name, "")
            ids = analyzer.unit_ids
            if name == "sua_mua" and config.enable_noise_classifier:
                ids = ids[table["noise_label"].str.lower().eq("neural").to_numpy()]
            if len(ids) == 0:
                metadata["models"][name] = {"repo_id": config.models.get(name), "evaluated_units": 0,
                                            "not_evaluated_units": len(table)}
                continue
            repo_id = config.models[name]
            model, info, provenance = (
                loaded_models[name] if loaded_models is not None else
                _load_classifier(repo_id, config.model_revisions.get(name), config.model_trust)
            )
            provenance = dict(provenance)
            metadata["models"][name] = provenance
            selected = analyzer.select_units(analyzer.unit_ids)
            required = list(model.feature_names_in_)
            features, params = _prepare_metrics(selected, required, info, config)
            features_file = f"unit_ml_features_{name}.csv"
            features.loc[ids, required].to_csv(Path(output_folder, features_file), index_label="unit_id")
            provenance["features_file"] = features_file
            if len(ids) != selected.get_num_units():
                selected = selected.select_units(ids)
            predictions = _predict(selected, model, info, config)
            allowed = {"neural", "noise"} if name == "noise" else {"sua", "mua"}
            if not set(predictions["label"].str.lower()).issubset(allowed):
                raise ValueError(f"Unexpected {name} classes; expected {sorted(allowed)}.")
            table.loc[ids, f"{name}_label"] = predictions["label"].to_numpy(dtype=str)
            table.loc[ids, f"{name}_probability"] = predictions["probability"].to_numpy(dtype=float)
            table.loc[ids, f"{name}_status"] = "evaluated"
            table[f"{name}_model_id"] = repo_id
            table[f"{name}_model_revision"] = provenance["revision"]
            provenance.update({"inference_utc": datetime.now(timezone.utc).isoformat(),
                               "evaluated_units": len(ids), "not_evaluated_units": len(table) - len(ids),
                               "class_counts": predictions["label"].value_counts().to_dict(),
                               "computed_metric_params": params})
        table["model_labeling_status"] = np.where(
            table[["noise_status", "sua_mua_status"]].eq("evaluated").any(axis=1), "evaluated",
            "disabled" if not config.enable_model_labeling else "not_evaluated",
        )
        for name, rule in config.model_filtering.items():
            candidate = table[f"{name}_label"].str.lower().isin([label.lower() for label in rule["exclude_labels"]])
            candidate &= table[f"{name}_probability"].ge(rule["min_probability"])
            candidate &= table[f"{name}_status"].eq("evaluated")
            table["filter_candidate"] |= candidate
        table["filter_applied"] = table["filter_candidate"] & config.apply_model_filtering
    except Exception as error:
        metadata["failures"].append({"classifier": name, "error": str(error),
                                     "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        table["model_labeling_status"] = "failed"
        table.loc[ids, f"{name}_status"] = "failed"
        raise RuntimeError(f"UnitRefine {name} labeling failed: {error}. Check model access, dependencies, and metric settings.") from error
    finally:
        from .auto_curate import _json_default

        metadata["status_counts"] = table["model_labeling_status"].value_counts().to_dict()
        Path(output_folder, "model_curation_metadata.json").write_text(
            json.dumps(metadata, default=_json_default, indent=2, allow_nan=False) + "\n"
        )
        table.to_csv(Path(output_folder, "unit_ml_classifications.csv"))
    return table, metadata


def persist_model_labels(analyzer, table, config):
    if not config.persist_labels_to_sorting:
        return
    selected = table.loc[analyzer.unit_ids]
    for column, property_name in MODEL_PROPERTIES.items():
        if "probability" in column and not config.preserve_model_probabilities:
            continue
        dtype = float if "probability" in column else str
        analyzer.set_sorting_property(property_name, selected[column].to_numpy(dtype=dtype))


def model_gui_kwargs(analyzer):
    """Options accepted by the pinned GUI's web and desktop Unit List."""
    names = [name for name in MODEL_PROPERTIES.values() if name in analyzer.sorting.get_property_keys()]
    return {"extra_unit_properties": {name: analyzer.sorting.get_property(name) for name in names}}
