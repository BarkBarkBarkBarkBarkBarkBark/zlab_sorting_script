"""Generate the DA128L geometry for probe P1925-127 / holder H1925-116."""

import argparse
from pathlib import Path

import numpy as np
from probeinterface import Probe, write_probeinterface


OUTPUT_PATH = Path(__file__).with_name("deeparray128_P1925-127_H1925-116.json")
OPEN_CHANNELS = (
    50, 51, 52, 53, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 66, 67, 68, 69, 70, 72,
)


def create_probe() -> Probe:
    """Use printed channel numbers as contact IDs and device channel indices."""
    channels = np.arange(128)
    right = channels < 64
    positions = np.empty((128, 2), dtype=float)
    positions[:, 0] = np.where(right, 43.0, 0.0)
    # Origin: distal left contact (ch64). Positive y points toward the connector.
    positions[:, 1] = np.where(right, 3175.0 - channels * 50.0, (channels - 64) * 50.0)

    probe = Probe(
        ndim=2,
        si_units="um",
        name="Deep Array 128L P1925-127 H1925-116",
        model_name="DA128L",
        manufacturer="Diagnostic Biochips",
        serial_number="P1925-127_H1925-116",
    )
    probe.set_contacts(
        positions=positions,
        shapes="circle",
        shape_params={"radius": 8.0},
        contact_ids=channels.astype(str),
        shank_ids=np.full(128, "0"),
    )
    probe.set_device_channel_indices(channels)
    probe.annotate(
        geometry_source="02-0008-01 Handling Guidance, DA128L_RevA.pdf, page 16, detail B",
        channel_map_source="20260925_P1925-127_H1925-116.pdf, pages 1-2",
        geometry_representation="2D drawing-plane layout of the wrapped probe; not a 3D surface model",
        coordinate_convention="x: left=0, right=43 um; y: proximal positive; ch64=(0, 0)",
        channel_mapping="device index equals printed channel number; verify Intan recording order before use",
        column_pitch_um=50.0,
        column_separation_um=43.0,
        axial_stagger_um=25.0,
        contact_diameter_um=16.0,
        recording_span_um=3175.0,
        quality_note="Manufacturer report: 108 good, 20 open, 0 short; no contacts removed",
    )
    probe.annotate_contacts(
        column=np.where(right, "right", "left"),
        manufacturer_status=np.where(np.isin(channels, OPEN_CHANNELS), "open", "good"),
    )
    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="ProbeInterface JSON destination")
    args = parser.parse_args()
    probe = create_probe()
    write_probeinterface(args.output, probe)
    print(f"Saved {probe.get_contact_count()} contacts to {args.output.resolve()}")
    print("20 manufacturer-reported open channels are annotated, not removed.")
    print("Verify that Intan channel order matches printed channels 0-127 before sorting.")


if __name__ == "__main__":
    main()
