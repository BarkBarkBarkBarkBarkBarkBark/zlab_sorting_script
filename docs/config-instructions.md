# Configuring a sorting run

Local use does not require AWS or a pre-existing conda environment. Start with
[environment setup](environment.md), then [the README workflow](../readme.md).

- Prefer an explicit session directory in `zsort.set_paths(recording_path=...)`.
- For label-based lookup, set `SORTING_DATA_ROOT` to an existing directory
  containing `<patient>/<session>/*.rhd` (or `<session>/raw/*.rhd`).
- Select the Intan stream and verify file order/continuity before concatenating.
- Choose a probe from `Custom_Probes/` and verify its positional channel wiring.
- Review output locations before invoking sorting/analysis; unreadable existing
  results are preserved and raise. Choose new destinations for retries; see
  [crash protection and recovery](sorting_recovery.md).
- Configure automated policy with `AutoCurationConfig`; see
  [automated curation](auto_curation.md) for defaults and model compatibility.

[AWS SSH](ssh_into_aws_machine.md) and [instance setup](start_ec2_instance.md)
remain optional infrastructure notes. Demo data:
[Intan example dataset](https://data.mendeley.com/datasets/w767nnk5wh/1).
