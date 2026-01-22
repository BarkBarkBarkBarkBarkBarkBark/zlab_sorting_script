def get_rhd_file(session_location):
    """Return the first .rhd file for this session.

    Back-compat note:
    - Older layouts passed a session folder that contained a `raw/` subfolder.
    - SAW layout passes the raw session folder directly:
        .../data/raw/<patient>/<session>/
    """
    candidates = [session_location, session_location / "raw"]
    for folder in candidates:
        try:
            rhd_files = sorted(folder.glob("*.rhd"))
        except Exception:
            rhd_files = []
        if not rhd_files:
            continue
        if len(rhd_files) > 1:
            print(
                f"Warning: Multiple .rhd files found in {folder}, using the first one: {rhd_files[0].name}"
            )
        intan_file = rhd_files[0]
        print(f"Found Intan file: {intan_file}")
        return intan_file

    raise FileNotFoundError(
        f"No .rhd file found in {session_location} (or {session_location / 'raw'})"
    )