def get_rhd_file(session_location):
    "test doc string"
    # Dynamically retrieve the .rhd file in the session folder
    raw_folder = session_location / "raw"
    rhd_files = list(raw_folder.glob("*.rhd"))
    print(len(rhd_files))
    if len(rhd_files) == 0:
        print(f"No .rhd file found in {raw_folder}")
    elif len(rhd_files) > 1:
        print(f"Warning: Multiple .rhd files found in {raw_folder}, using the first one: {rhd_files[0].name}")
    intan_file = rhd_files[0]
    
    print(f"Found Intan file: {intan_file}")
    return intan_file