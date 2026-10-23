# Installing from source

Clone the repository using its Git remote, then open the `zlab_sorting_script`
checkout. Follow [the local environment guide](environment.md) to create a venv
and install the appropriate extras. All install commands run from the repository
root, not `src/` or `notebooks/`.

For the full workflow, after creating/activating the venv and selecting PyTorch:

```bash
python -m pip install -r requirements.txt
```

The editable install makes changes under `src/` available without reinstalling.
Restart notebook kernels after editing imported modules or dependencies.
See [the README](../readme.md) for data paths and the manual workflow.
