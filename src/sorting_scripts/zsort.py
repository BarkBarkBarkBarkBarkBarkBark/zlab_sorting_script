import json
from pathlib import Path

from sorting_scripts import get_file

import spikeinterface.full as si
import probeinterface as pi
import spikeinterface.sorters as ss
from spikeinterface.sorters import run_sorter
from spikeinterface.curation import apply_curation

import numpy as np

import os
from pathlib import Path
import time


def _ensure_signed_recording(recording):
    """
    SpikeInterface preprocessing (e.g., bandpass_filter) does not support unsigned dtypes
    (common for Intan: uint16). Convert to signed if needed.
    """
    try:
        if hasattr(recording, "get_dtype"):
            dtype = recording.get_dtype()
            if np.issubdtype(dtype, np.unsignedinteger):
                print(f"[zsort] recording dtype is unsigned ({dtype}); converting to signed")
                return si.unsigned_to_signed(recording)
    except Exception as e:
        # If dtype introspection fails for any reason, keep going with original recording.
        print(f"[zsort] warning: could not inspect/convert dtype: {e}")
    return recording



def _find_repo_root(start: Path) -> Path:
    """
    Find the sorting_script repo root by walking upward until we find pyproject.toml
    (or .git as a fallback).
    """
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
    raise RuntimeError("Could not locate repo root (no pyproject.toml or .git found).")


def _resolve_data_root(repo_root: Path) -> Path:
    """
    Resolve the data root in a way that matches your REALTIME_SORTING layout.

    Priority:
      1) SORTING_DATA_ROOT / DATA_ROOT env var (portable across machines)
      2) sibling data/ folder (REALTIME_SORTING/data)  <-- your screenshot layout
      3) repo-local data/ folder (sorting_script/data)
      4) legacy fallback ~/codespace/data (optional)
    """
    env = os.environ.get("SORTING_DATA_ROOT") or os.environ.get("DATA_ROOT")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Data root from env var does not exist: {p}")

    sibling = repo_root.parent / "data"
    if sibling.exists():
        return sibling.resolve()

    inside = repo_root / "data"
    if inside.exists():
        return inside.resolve()

    legacy = Path.home() / "codespace" / "data"
    if legacy.exists():
        return legacy.resolve()

    raise FileNotFoundError(
        "Could not find a data root. Create a 'data/' folder next to the repo, "
        "or set SORTING_DATA_ROOT (or DATA_ROOT)."
    )


def set_paths(patient: str, session: str) -> dict:
    try:
        here = Path(__file__)
    except NameError:
        here = Path.cwd()

    repo_root = _find_repo_root(here)
    data_root = _resolve_data_root(repo_root)

    session_location = data_root / patient / session

    sorted_data = session_location / f"{patient}-{session}-sorted"
    sorter_output_folder = sorted_data / f"{patient}-{session}-sorter_folder"
    analyzer_folder = sorted_data / f"{patient}-{session}-analyzer_folder"

    sorted_data.mkdir(parents=True, exist_ok=True)

    intan_file = None
    try:
        intan_file = get_file.get_rhd_file(session_location)
    except FileNotFoundError:
        print(f"[set_paths] No .rhd found under: {session_location}")
    except Exception as e:
        print(f"[set_paths] Could not resolve .rhd under {session_location}: {e}")

    return {
        "patient": patient,
        "session": session,
        "repo_root": repo_root,
        "data_root": data_root,
        "session_location": session_location,
        "sorted_data": sorted_data,
        "sorter_output_folder": sorter_output_folder,
        "analyzer_folder": analyzer_folder,
        "intan_file": intan_file,  # will be None if not found
    }



def set_probe(recording, path_dict: dict, custom_probe_name):
    """
    Load a Probe from JSON and attach it to the recording.

    recording: spikeinterface recording object
    path_dict: 
    custom_probe_name: name of the .json file in the custom probe directory
    for the example its neuronexus-A16x1_2mm_50_177_A16.json
    """
    try:
        probe = recording.get_probe()
        if probe is not None:
            print("Probe already attached!")
            return recording
    except:
        print("no probe, attaching one")
    

    repo_root = path_dict["repo_root"]
    # Support both historical and current folder naming.
    probe_dir = repo_root / "Custom_Probes"
    if not probe_dir.exists():
        probe_dir = repo_root / "custom_probes"
    probe_path = probe_dir / custom_probe_name

    probegroup = pi.read_probeinterface(probe_path)
    if len(probegroup.probes) < 1:
        raise ValueError(f"No probes found in: {probe_path}")

    probe = probegroup.probes[0]
    recording = recording.set_probe(probe, in_place=False)

    n_rec = recording.get_num_channels()
    n_probe = probe.get_contact_count()
    if n_probe != n_rec:
        raise ValueError(
            f"Probe contacts ({n_probe}) != recording channels ({n_rec}). "
            "Pick the correct probe variant or subset/remap accordingly."
        )

    return recording



def sort(recording, path_dict: dict):
    """
    Load an existing sorting from sorter folder if present; otherwise run KS4.
    Returns a spikeinterface Sorting object.
    """
    sorter_output_folder = path_dict["sorter_output_folder"]

    try:
        sorting = ss.read_sorter_folder(sorter_output_folder)
        return sorting
    except Exception:
        rec = _ensure_signed_recording(recording)
        rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
        rec = si.center(rec)
        rec = si.whiten(rec)

        sorting = run_sorter(
            sorter_name="kilosort4",
            recording=rec,
            folder=sorter_output_folder,
            remove_existing_folder=True,
            verbose=True,
        )
        return sorting


def analyze(recording, sorting, path_dict: dict):
    """
    Load a SortingAnalyzer if it exists; otherwise create + compute extensions.
    Returns a SortingAnalyzer.
    """
    analyzer_folder = path_dict["analyzer_folder"]

    try:
        sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)
        return sorting_analyzer
    except Exception as e:
        print("No valid analyzer found, creating a new one")
        print(f"Reason: {e}")

    recording = _ensure_signed_recording(recording)
    recording_filtered = si.bandpass_filter(recording, freq_min=300, freq_max=6000)

    job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="1s")

    sorting_analyzer = si.create_sorting_analyzer(
        sorting=sorting,  # IMPORTANT: pass the Sorting object, not the folder path
        recording=recording_filtered,
        folder=analyzer_folder,
        overwrite=True,
        format="binary_folder",
        **job_kwargs,
    )

    sorting_analyzer.compute(
        {
            "random_spikes": dict(method="uniform", max_spikes_per_unit=500),
            "waveforms": {},
            "templates": {},
            "noise_levels": {},
            "unit_locations": dict(method="monopolar_triangulation"),
            "isi_histograms": {},
            "correlograms": dict(window_ms=100, bin_ms=5),
            "principal_components": dict(
                n_components=3,
                mode="by_channel_global",
                whiten=True,
            ),
            "quality_metrics": dict(metric_names=["snr", "firing_rate"]),
            "spike_amplitudes": {},
            "template_similarity": dict(method="l1"),
        },
        **job_kwargs,
    )

    return sorting_analyzer


def save_curated_data(patient: str, session: str, sorting_analyzer, path_dict: dict):
    """
    Apply GUI curation_data.json to the analyzer and save a new curated analyzer as zarr.
    Returns the saved curated analyzer.
    """
    curation_filepath = (
        path_dict["analyzer_folder"] / "spikeinterface_gui" / "curation_data.json"
    )
    
    base_out = path_dict["sorted_data"] / f"{patient}-{session}-curated_analyzer.zarr"

    if not curation_filepath.exists():
        raise FileNotFoundError(
            f"Missing curation file: {curation_filepath}\n"
            "Run the GUI curation and click Save first."
        )

    out = base_out
    if out.exists():
        stem = out.name.replace(".zarr", "")
        parent = out.parent
        k = 2
        while True:
            candidate = parent / f"{stem}-v{k}.zarr"
            if not candidate.exists():
                out = candidate
                break
            k += 1

    with open(curation_filepath, "r") as f:
        curation_dict = json.load(f)

    sorting_analyzer = si.load_sorting_analyzer(path_dict["analyzer_folder"])
    clean_analyzer = apply_curation(sorting_analyzer, curation_dict_or_model=curation_dict, merging_mode = "hard")
    saved = clean_analyzer.save_as(format="zarr", folder=out)

    print(f"Wrote: {out}")
    return saved



def generate_figures(analyzer_obj, path_dict, close_figures: bool = True):
    """
    Generate per-unit summary figures and save them to disk.

    Note: In Jupyter, matplotlib will auto-render any open figures created in a cell.
    Setting close_figures=True prevents notebook rendering while still saving files.
    """
    import matplotlib.pyplot as plt
    import spikeinterface.widgets as sw

    fig_dir = path_dict["sorted_data"] / "figs"
    fig_dir.mkdir(parents = True, exist_ok = True)

    sorting_obj = analyzer_obj.sorting
    unit_ids = sorting_obj.unit_ids

    for id in unit_ids:
        w = sw.plot_unit_summary(
            analyzer_obj,
            unit_id=id,
            backend="matplotlib",
            ncols=5,
            figsize=(16, 9),
            figtitle=f"Unit summary: {id}"
            )

        w.figure.savefig(
            Path(fig_dir, f"{id}-figure.png")
        )

        if close_figures:
            plt.close(w.figure)
