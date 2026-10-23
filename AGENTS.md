# Working in this repository

Read [agent_reference.md](agent_reference.md) for the code map, upstream API
references, scientific invariants, and validation commands. Read
[readme.md](readme.md) for the human workflow and output/rerun caveats.

Preserve existing user changes and raw recordings. Use synthetic tests for
development. Keep dependencies in `pyproject.toml`, with requirements files
selecting extras. Do not silently upgrade the pinned scientific APIs or invoke
real sorting, model downloads, or cloud transfers as part of a code review.
