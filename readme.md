
**Goal:** Produce a curated SpikeInterface sorting output (clean_analyzer.zarr) from raw Intan .rhd data, using AWS compute, Kilosort4, and the SpikeInterface GUI.

  

**Final Deliverable:** A curated analyzer saved to disk (typically clean_analyzer.zarr/) and uploaded back to storage.

**Demo Data:** [can be found here](https://data.mendeley.com/datasets/w767nnk5wh/1)

### Deep Array 128L probe

[Generator](Custom_Probes/create_deeparray128_probe.py) and
[ProbeInterface JSON](Custom_Probes/deeparray128_P1925-127_H1925-116.json)
describe Diagnostic Biochips probe P1925-127 / holder H1925-116.
The generator requires NumPy and [ProbeInterface](https://probeinterface.readthedocs.io/en/main/api.html);
the repository requirements include ProbeInterface 0.3.2.

Dimensions come from page 16, detail B, of the local
`02-0008-01 Handling Guidance, DA128L_RevA.pdf`: 128 circular sites, 16 um
diameter, two 64-site columns separated by 43 um, 50 um pitch within each
column, and 25 um axial stagger. The combined center-to-center span is 3175 um.
This is a 2D drawing-plane representation of the wrapped probe, not a
reconstructed 3D cylindrical surface. No undocumented shank outline is invented.

The local `20260925_P1925-127_H1925-116.pdf` supplies the numbered layout:
left 127 to 64 and right 0 to 63, proximal to distal. Coordinates use x=0 for
left, x=43 um for right, and positive y toward the connector. Distal left ch64
is (0, 0); distal right ch63 is (43, 25); proximal right ch0 is (43, 3175).
Contacts, contact IDs, and device indices are ordered by printed channel 0-127.

**Before sorting, verify the recording's actual channel order against the
printed channel numbers.** Device indices are positional, not matched to
Intan channel names. If channels have been reordered, selected, or combined
across headstages, update the wiring accordingly before attaching the probe.
The report marks 108 channels good and 20 open; these are stored in the
`manufacturer_status` contact annotation. All 128 sites remain in the JSON.
Open channels are **not** automatically removed or disconnected; review the
recording and explicitly exclude appropriate channels before sorting.

From the repository root, using the existing sorting environment:

```bash
python Custom_Probes/create_deeparray128_probe.py
python -m unittest discover -s Custom_Probes -p 'test_deeparray128_probe.py'
```

The default JSON destination is beside the script, independent of the working
directory; `--output /path/to/probe.json` selects another destination.

```python
from probeinterface import read_probeinterface

probe = read_probeinterface(
    "Custom_Probes/deeparray128_P1925-127_H1925-116.json"
).probes[0]
# Only after verifying recording order corresponds to printed channels 0-127:
recording = recording.set_probe(probe)
```

  

### 1. Log In & Environment Setup
- Connect to the AWS instance via VS Code → Remote SSH.
- Open a terminal on the remote machine.
- Activate the sorting environment:

```
conda deactivate

conda activate sorter
```

  

### 2. Open & Run the Notebook

Open the Sorting Notebook in VS Code.
Set the dataset at the top of the notebook:
  
```
patient = "raw_intan"

session = "Session1"
```

The notebook will:
- Load the Intan .rhd file
- Attach the probe definition
- Run Kilosort4 if needed
- Build a SortingAnalyzer with waveforms, PCA, metrics, etc.

Run cells top to bottom until both of the following exists:
1. A Kilosort output folder (sorted/sorter_folder/)
2. A SortingAnalyzer folder (sorted/analyzer_folder/)

  
### 3. Launch the Spike Curation GUI

Once the analyzer exists, do not use Jupyter for curation.
From the terminal in vscode, run:

  
```
sigui --mode=web --curation "/path/to/your/codespace/data/raw_intan/Session1/sorted/analyzer_folder"
```
  
The GUI opens in your browser.

### 4. Perform Curation

Curation can be performed manually, just like osort. First a directory of figures could be generated, and then sifted through. 

Alternatively, merge candidates can be detected based on waveform similarity and ISI, and accepted or rejected manually. 

Obvious noise can be deleted, either using the GUI or in the notebook. 

Units areremoved with the `analyers.remove_units()` method, and merged with the `analyzers.merge_units()` method. In both cases, the main arguments are the sorting_analyzer and a dictionary containing the units to delete or merge. 

Quality metrics are generated using the method `spikeinterface.widget.plot_unit_summary()`, after calculating the needed extensions, example is in the notebook. 
