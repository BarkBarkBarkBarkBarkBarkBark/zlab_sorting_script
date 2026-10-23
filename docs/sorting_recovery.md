# Kilosort crash protection and clustering recovery

The lab's 2026-10-23 Kilosort 4.1.1 run reached final clustering after ~68 minutes
and failed in `swarmsplitter.check_CCG`: candidate spike trains with zero combined
time span caused `CCG_metrics` to divide by zero. A small synthetic pair of
coincident spike trains reproduces the original exception.

## What the adapter changes

`zsort.sort()` now runs inside the scoped adapter in
[`kilosort_runtime.py`](../src/sorting_scripts/kilosort_runtime.py):

- Empty or zero-duration CCG inputs return “not refractory,” matching the
  [upstream swarmsplitter guard](https://github.com/MouseLand/Kilosort/blob/main/kilosort/swarmsplitter.py).
  This does not force a merge or discard spikes. Normal CCG calculations use the
  original upstream function unchanged. The same degenerate-input protection is
  applied to the CCG entry point used by subsequent template merging. Its third
  return value is NaN when the statistic is undefined. Non-finite inputs raise.
- Immediately before final clustering, it writes `precluster_checkpoint.pt`
  into the SpikeInterface sorter folder. This contains detected spike times,
  feature tensors, Kilosort ops, the time offset, compute device, package pins,
  and Python/NumPy/PyTorch random-generator states. A temporary `.writing` file
  is atomically renamed only after serialization succeeds. Disk/write failure
  stops the run before clustering; it does not silently disable checkpointing.
- Existing checkpoints are not replaced. The binary recording is retained
  rather than deleted after success, so a checkpoint/output still has its
  source available for trace inspection. Expect additional disk use: features
  are proportional to spike count × nearest channels × PCs, plus ops and binary.
- Installed files are not edited. Hooks are restored on success or failure.
  They are process-global while active; do not start other Kilosort work in the
  same Python process. Concurrent/nested adapter calls are rejected.
- The adapter requires Kilosort **4.1.1** and SI **0.103.0**. It fails closed on
  different versions rather than assuming private integration points still work.

Pass `checkpoint=False` to `zsort.sort` only when checkpointing is intentionally
unwanted; the CCG guard and output protections still apply, but a late failure
will have no saved features and the binary may be deleted on success.

## Existing run protection

Both `sort()` and `analyze()` reuse readable results. If a destination already
exists but loading fails, they raise and preserve it. They never silently replace
an unreadable folder. A readable result is still not compared against the input
recording/probe/settings: choose a new destination when those change.

For a deliberate fresh attempt, choose **new, nonexistent** folders. For example,
after loading the updated module in a fresh kernel or with `importlib.reload`:

```python
from pathlib import Path
from sorting_scripts import zsort

retry_paths = dict(path_dict)
retry_paths["sorter_output_folder"] = Path(path_dict["sorted_data"]) / "sorter_attempt_2"
retry_paths["analyzer_folder"] = Path(path_dict["sorted_data"]) / "analyzer_attempt_2"
# Verify both destinations do not already exist before this expensive step.
sorting = zsort.sort(rec, retry_paths)
analyzer = zsort.analyze(rec, sorting, retry_paths)
```

Keep `retry_paths` for later GUI/manual-save calls so they use the new analyzer.
No retry is launched automatically. Existing logs and raw recordings remain intact.
The old failed run has no pre-clustering checkpoint; adding this feature cannot
recover feature tensors that were never saved.

## CUDA out-of-memory during drift correction

The DB007 attempt at 14:57 on 2026-10-23 failed during initial spike detection
for drift correction, at batch 215 with 128 channels and 60000 samples per batch.
Its 3.62 GiB GPU could not allocate a 584 MiB temporary tensor. This is before
the pre-clustering checkpoint, so clustering resume cannot recover this attempt.

Restart the notebook kernel to release the failed process's tensors, reload the
recording/probe, and select new destinations using `retry_paths` above. Then try:

```python
sort = zsort.sort(rec, retry_paths, batch_size=15000, clear_cache=True)
analyzer = zsort.analyze(rec, sort, retry_paths)
```

These options are supported by the pinned SI/KS4 versions. Defaults remain
60000 samples and `clear_cache=False`. At 30 kHz, 15000 samples is 0.5 seconds
per batch; it reduces batch-dependent temporary tensors but does not guarantee
the entire sort will fit. Smaller batches can change drift estimation and sorting
results; cache clearing may slow execution. Review results before accepting them.
If memory still runs out, use a GPU with more available VRAM.

PyTorch also suggests `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for
fragmentation. If using it, set it in the first cell after restarting the kernel,
before importing torch or the sorting modules. It does not increase GPU capacity.
See [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
and [Kilosort batch size](https://kilosort.readthedocs.io/en/latest/parameters.html#batch-size).
No real-data retry is launched by this change.

## Resume a future checkpoint

Only load a checkpoint generated by your own trusted run. Torch's general
serialization includes Python objects; `weights_only=False` is required here.
Use the same environment and original CPU/CUDA device; tensors retain their
original placement, and CUDA random-state restoration requires the same device
count. Do not move/delete the checkpoint's source recording.

```python
import spikeinterface.extractors as se
from sorting_scripts.kilosort_runtime import resume_clustering

output = resume_clustering(
    retry_paths["sorter_output_folder"] / "precluster_checkpoint.pt",
    "/absolute/path/to/new_resumed_kilosort_output",
)
sorting = se.read_kilosort(output)
analyzer = zsort.analyze(rec, sorting, retry_paths)
```

Resume skips preprocessing, drift correction, and spike extraction. In the
pinned Kilosort version final clustering/merging operate on saved features and
do not read the binary. It restores the recorded RNG states and writes into a
new **raw Kilosort** output directory; it does not alter the old SpikeInterface
failure log. `zlab_resume_complete.json` is written only after clustering and
export succeed. On failure the checkpoint survives and the partial destination
is retained; retry into another new destination. Resume restores the caller's
RNG state afterward. Its recorded runtime measures the resumed stage.

## Validation commands

```bash
python -m unittest discover -s src/sorting_scripts/tests -p 'test_sorting_recovery.py' -v
python -m unittest discover -s src/sorting_scripts/tests -v
python -m unittest discover -s Custom_Probes -p 'test_*.py' -v
```

Sorter-specific tests require the optional sorter extra; CPU-only review
installations skip them. Tests reproduce the unpatched CCG failure, compare
ordinary guarded/unguarded results, exercise actual checkpoint serialization,
simulate a late crash and resumed RNG/input replay, reject checkpoint replacement,
and verify disk-write failures, partial resumes, output preservation, hook
restoration, and version gating. Full GPU validation remains a separate check.

## Validation completed (2026-10-23)

- Combined suite with `ZLAB_RUN_KS4_INTEGRATION=1`: 59 tests discovered,
  58 passed, and the separate real-Hugging-Face model test skipped.
- All 6 probe tests passed. `pip check` and `git diff --check` passed.
- The 14 focused regression tests include the original zero-duration crash and
  ordinary-result equivalence, failure/preservation checks, and checkpoint replay.
- The real CPU integration sorted a generated 20-second, 16-channel recording,
  confirmed calls into refractory graph splitting, then resumed the saved
  checkpoint. Spike times, cluster IDs, and template arrays matched exactly.
  An earlier 8-second smoke test also passed.
- GPU validation could not run: `nvidia-smi` outside the sandbox reported
  `Failed to initialize NVML: Driver/library version mismatch` (library 595.99).
  Resolve GPU availability and verify `torch.cuda.is_available()` before a full
  GPU retry. CPU tests do not establish CUDA correctness or real-data validity.
- The failed DB007 run, raw recordings, installed Kilosort source, and package
  versions were left unchanged. Its failure log retains its original timestamp.
