"""Discover Intan inputs without silently discarding recording chunks."""

import re
from pathlib import Path


def get_rhd_files(recording_path: str | Path) -> list[Path]:
    """Find an RHD file, directory of chunks, or legacy session/raw directory.

    Files are naturally sorted by filename (including Intan's timestamp suffix).
    Discovery is not recursive, so generated/output subdirectories are excluded.
    """
    source = Path(recording_path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".rhd":
            raise ValueError(f"Expected an .rhd file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Recording path does not exist: {source}")

    def find_files(directory: Path) -> list[Path]:
        return [
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".rhd"
        ]

    files = find_files(source)
    raw_files = find_files(source / "raw") if (source / "raw").is_dir() else []
    if files and raw_files:
        raise ValueError(
            f"RHD files exist in both {source} and its raw/ directory; "
            "pass an unambiguous recording directory."
        )
    files = files or raw_files
    if not files:
        raise FileNotFoundError(f"No .rhd files found in {source} or its raw/ directory")

    def natural_key(path: Path):
        return [
            (1, int(part)) if part.isdigit() else (0, part)
            for part in re.split(r"(\d+)", path.name.casefold())
        ], path.name

    return sorted(files, key=natural_key)


def get_rhd_file(session_location: str | Path) -> Path:
    """Return a single RHD input; require explicit concatenation for chunked data."""
    files = get_rhd_files(session_location)
    if len(files) != 1:
        raise ValueError(
            f"Found {len(files)} RHD files in {session_location}. "
            "Use zsort.load_recording() to concatenate all chunks."
        )
    return files[0]