# Local environment setup

Run commands from the **repository root** (the directory containing
`pyproject.toml`). Install a regular CPython 3.12 interpreter first. Python
3.10–3.12 is the declared compatibility range; 3.12 is the recommended starting
point. Newer Python releases are intentionally excluded until tested with the
pinned scientific stack. A venv isolates Python packages; it does not install an
NVIDIA driver or provision a GPU.

## 1. Create and activate

If a conda environment is active, deactivate it before creating a venv. Do not
replace a working lab environment; create a new directory for evaluation.

Linux/macOS, bash or zsh:

```bash
cd /path/to/zlab_sorting_script
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

If Python reports that `ensurepip`/venv is unavailable on Debian/Ubuntu, ask your
machine administrator to install the matching `python3.12-venv` OS package.

Windows PowerShell:

```powershell
cd C:\path\to\zlab_sorting_script
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If execution policy blocks activation, use `.\.venv\Scripts\python.exe` wherever
these instructions say `python`; activation is optional and no policy change is
required. Use absolute Windows paths or Python raw strings for recording paths.

Confirm `python -c "import sys; print(sys.executable)"` points into this checkout's
`.venv`. Reactivate the environment in each new terminal; `deactivate` exits it.

## 2. Choose the workload

### Full sorting workstation

Use an NVIDIA/CUDA machine for the lab sorting workflow. Check `nvidia-smi` first.
Install the appropriate **torch** command from the official
[PyTorch installer](https://pytorch.org/get-started/locally/) using its OS, pip,
and CUDA choices. Select a driver-compatible build with torch >=2.1 (Kilosort4's
minimum). Torchvision/torchaudio are not used by this project. Run that command
inside this venv, then:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

Expect `CUDA available: True` before starting a GPU run. Installing torch first
lets pip retain that compatible CUDA build rather than choosing a generic build
through Kilosort's dependency. No CUDA wheel index or driver version is hardcoded
in requirements because those depend on the machine. See the official
[Kilosort guide](https://kilosort.readthedocs.io/en/latest/README.html) for upstream
hardware/setup guidance. This repository does not need MATLAB or Docker for KS4.

### CPU-only analysis, tests, and web curation

```bash
python -m pip install -e ".[gui,notebook]"
python -m pip check
```

This omits Kilosort and its torch dependency. `spikeinterface[full]` still installs
scientific analysis and model tooling (and may install CUDA Python bindings on
non-macOS systems); those bindings alone do not establish GPU support. The two
workflow notebooks import torch in diagnostic cells: skip/adapt those cells for
this environment, and work with an existing analyzer instead of calling sort().
No Qt installation is needed for the web GUI. The optional desktop GUI and legacy
sortingview notebook backends are not included; install those separately only
when required. A base code-only install is `python -m pip install -e .`.

### Optional UnitRefine model labeling

The lab's prior model audit found scikit-learn **1.4.2** in both models' training
metadata. Cross-version estimator loading is unsupported; newer sklearn is not
interchangeable. Create a **separate** venv (for example `.venv-unitrefine`) and,
after the torch step if sorting is needed, install:

```bash
python -m pip install -r requirements-unitrefine.txt
python -m pip check
```

This selects sklearn 1.4.2 and NumPy <2, while retaining the workflow package
pins. It is a compatibility candidate, not proof that every live model revision
works: SI was 0.102.0 at model training, whereas this repo uses 0.103.0. Model
metadata and APIs can change. See [the prior audit](model_labeling_audit.md) and
[configuration/provenance guide](auto_curation.md). Pin `config.model_revisions`
to reviewed Hugging Face commit IDs for reproducible inference. The loader
rejects sklearn mismatches and records failures; do not bypass that check.

For CPU-only model review use
`python -m pip install -e ".[gui,notebook,unitrefine]"` instead. To avoid downloads
and model compatibility requirements, explicitly set
`AutoCurationConfig(enable_model_labeling=False)`; this does not disable cleanup
or automatic merging. Real model tests are opt-in and require network/cache.

## 3. Select the notebook kernel and check imports

```bash
python -m ipykernel install --user --name zlab-sorting --display-name "Z-Lab sorting"
python -c "import sys; from importlib.metadata import version; from sorting_scripts import zsort, auto_curate; print(sys.executable); print({p: version(p) for p in ['spikeinterface', 'probeinterface', 'numpy', 'scikit-learn', 'spikeinterface-gui']})"
python -m unittest discover -s src/sorting_scripts/tests -v
python -m unittest discover -s Custom_Probes -p 'test_*.py' -v
```

Kernel registration writes a local Jupyter kernelspec, not project files. In
VS Code select **Z-Lab sorting** in the notebook kernel picker and verify
`sys.executable`. Alternatively start `python -m jupyterlab` from this terminal.
If you recreate/move the environment, register the kernel again. To remove its
registration later: `jupyter kernelspec uninstall zlab-sorting`.

A missing import usually means the wrong kernel or an uninstalled optional extra.
Restart the kernel after dependency changes. A CUDA-false result requires checking
the torch build and NVIDIA driver, not rerunning sorting. On a remote server,
forward the GUI port and follow the URL the launcher prints. Use
`zsort.launch_curation_gui` for notebook launches to apply the existing Param
workaround.

## Compatibility and reproducibility

`pyproject.toml` is the dependency source of truth. `requirements.txt` selects its
GUI/notebook/sorter extras and an editable install. Core workflow versions remain
SI 0.103.0, ProbeInterface 0.3.2, GUI 0.11.0, and Kilosort 4.1.1; scientific and
notebook dependencies have broader bounds. This is an install specification,
**not a fully locked environment or a claim of all-platform validation**. Linux
Python 3.12 has fresh-environment installation and synthetic test coverage; see
[the review validation](usability_review.md#validation-completed) for exact versions.
Windows, macOS, other declared Python versions, and GPU sorting need their own
acceptance runs. Wheel availability can differ by OS and architecture.

After validating a machine, record its resolved packages with the run's outputs:

```bash
python -m pip freeze > environment-freeze.txt
python -m pip check
```

Also record OS, Python, GPU/driver, torch/CUDA versions, code revision, probe,
sorter settings, curation config, and model commits. A freeze from another OS is
not a universal replacement for requirements. Upgrade one environment copy at a
time, run synthetic tests, then validate a small recording before sharing it.
