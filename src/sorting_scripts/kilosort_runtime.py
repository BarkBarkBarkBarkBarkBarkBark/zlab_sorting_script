"""Scoped compatibility and recoverable clustering for the pinned Kilosort 4.1.1.

No installed files are modified. Hooks are process-global while active: run one
sort at a time per Python process. Checkpoints use torch serialization and must
only be loaded from trusted local runs.
"""

from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import random
import threading
import time
from unittest.mock import patch

import numpy as np


_RUNTIME_LOCK = threading.Lock()
SUPPORTED_VERSIONS = {"kilosort": "4.1.1", "spikeinterface": "0.103.0"}


def _check_versions():
    for package, expected in SUPPORTED_VERSIONS.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(
                f"Kilosort runtime hooks require {package}=={expected}, found {actual}. "
                "Validate the adapter before changing versions."
            )


def _guarded_check(original, result_size):
    """Only bypass undefined CCGs; ordinary inputs use unchanged upstream code."""
    def check(st1, st2=None, *args, **kwargs):
        second = st1 if st2 is None else st2
        if len(st1) == 0 or len(second) == 0:
            return (False, False) if result_size == 2 else (False, False, np.nan)
        if not np.isfinite(st1).all() or not np.isfinite(second).all():
            raise ValueError("Kilosort CCG spike times must be finite.")
        duration = max(np.max(st1), np.max(second)) - min(np.min(st1), np.min(second))
        if duration == 0:
            return (False, False) if result_size == 2 else (False, False, np.nan)
        if not np.isfinite(duration):
            raise ValueError("Kilosort CCG spike times must be finite.")
        return original(st1, st2, *args, **kwargs)
    return check


def _rng_state(torch):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state, torch):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Resume needs the checkpoint's CUDA device count.")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


@contextmanager
def kilosort_runtime(checkpoint_path=None):
    """Guard degenerate CCGs and save inputs immediately before final clustering.

    The checkpoint is committed by atomic rename before upstream clustering
    starts. An existing checkpoint is never replaced. Failures propagate and
    all hooks are restored. The binary recording is retained by zsort.sort().
    """
    import torch

    _check_versions()
    if not _RUNTIME_LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent or nested Kilosort runs in one process are unsupported.")
    try:
        swarmsplitter = import_module("kilosort.swarmsplitter")
        ccg = import_module("kilosort.CCG")
        runtime = import_module("kilosort.run_kilosort")
        original_cluster = runtime.cluster_spikes
        signature = inspect.signature(original_cluster)

        def cluster(*args, **kwargs):
            if checkpoint_path is not None:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                data = bound.arguments
                destination = Path(checkpoint_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive reservation prevents accidental checkpoint replacement.
                reservation = destination.with_suffix(destination.suffix + ".writing")
                with reservation.open("xb"):
                    pass
                try:
                    if destination.exists():
                        raise FileExistsError(f"Checkpoint already exists: {destination}")
                    torch.save({
                        "schema": 1, "versions": dict(SUPPORTED_VERSIONS),
                        "st": data["st"], "tF": data["tF"], "ops": data["ops"],
                        "imin": data["bfile"].imin, "device": str(data["device"]),
                        "rng": _rng_state(torch),
                    }, reservation)
                    os.replace(reservation, destination)
                finally:
                    reservation.unlink(missing_ok=True)
            return original_cluster(*args, **kwargs)

        with patch.object(swarmsplitter, "check_CCG", _guarded_check(swarmsplitter.check_CCG, 2)), \
             patch.object(ccg, "check_CCG", _guarded_check(ccg.check_CCG, 3)), \
             patch.object(runtime, "cluster_spikes", cluster):
            yield
    finally:
        _RUNTIME_LOCK.release()


def resume_clustering(checkpoint_path, output_folder):
    """Resume a trusted checkpoint into a NEW raw Kilosort output directory.

    Returns its path; load with spikeinterface.extractors.read_kilosort().
    This does not rewrite the failed SpikeInterface log or pretend that its
    sorter folder completed. Uses the original compute device and RNG state.
    In pinned KS4 4.1.1, final clustering/merging do not read the binary file.
    Keep that recording for subsequent Phy/trace inspection.
    """
    import torch

    _check_versions()
    output = Path(output_folder).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Resume requires a new output directory: {output}")
    # weights_only=False is necessary for NumPy arrays/ops: trusted files only.
    # Retain each tensor's original CPU/CUDA placement. ops mixes both, so
    # moving every tensor to the compute device would change upstream behavior.
    data = torch.load(checkpoint_path, weights_only=False)
    if data.get("schema") != 1 or data.get("versions") != SUPPORTED_VERSIONS:
        raise ValueError("Unsupported Kilosort checkpoint schema or package versions.")
    device = torch.device(data["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This checkpoint needs its original CUDA device.")

    runtime = import_module("kilosort.run_kilosort")
    ops = data["ops"]
    ops["settings"]["results_dir"] = str(output)
    tic = time.time()
    with kilosort_runtime():
        previous_rng = _rng_state(torch)
        try:
            _restore_rng(data["rng"], torch)
            output.mkdir(parents=True, exist_ok=False)
            clu, wall, st, features = runtime.cluster_spikes(
                st=data["st"], tF=data["tF"], ops=ops, device=device,
                bfile=None, tic0=tic,
            )
            runtime.save_sorting(
                ops=ops, results_dir=output, st=st, clu=clu, tF=features,
                Wall=wall, imin=data["imin"], tic0=tic,
            )
            (output / "zlab_resume_complete.json").write_text(
                json.dumps({"checkpoint": str(Path(checkpoint_path).resolve()),
                            "versions": SUPPORTED_VERSIONS}, indent=2) + "\n"
            )
        finally:
            _restore_rng(previous_rng, torch)
    return output
