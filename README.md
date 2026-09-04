# jsats3d

`jsats3d` is a Python package developed by Kleinschmidt to clean, synchronize, process, and position 3D acoustic telemetry data (JSATS) for small-to-medium scale fish tracking studies (<1,000 tagged individuals, <20 receivers).

The package provides a standardized workflow covering raw receiver data import, metronome beacon alignment, linear clock drift correction, machine-learning multipath classification, Daniel Deng's exact 3D TDOA positioning solver, and 3D kernel utilization density estimation.

---

## Installation

Install `jsats3d` in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

Or install core dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Quickstart

```python
import jsats3d as js

# Create project database
js.create_project_db("./data", "study.db")

# Set study parameters
js.set_study_parameters(
    utc_conv=-5,
    bm_elev=100.0,
    bm_elev_units="feet",
    output_units="meters",
    masterReceiver="REC_01",
    synch_time_start="2026-05-01 00:00:00",
    synch_time_end="2026-06-01 00:00:00",
    dbName="./data/study.db",
)

# Import cabled receiver raw data
js.acoustic_data_import(
    site="REC_01",
    recType="Teknologic",
    rawDataFiles="./data/raw",
    projectDB="./data/study.db",
)
```

For complete usage instructions, see [RUNNING.md](RUNNING.md) and the step-by-step workflow guide in [docs/workflow.md](docs/workflow.md).

---

## Project Database Schema Overview

`jsats3d` uses a standardized SQLite database schema to manage telemetry processing steps reproducibly:

- **tblStudyParameters**: Global study configuration including master receiver ID, UTC offset, benchmark elevation, and synchronization window start/end timestamps.
- **tblTag**: Tag registry detailing tag IDs, tag type (`study` or `beacon`), and pulse repetition intervals.
- **tblReceiver**: Receiver deployment metadata, coordinates (`X`, `Y`, `Z`, `X_t`, `Y_t`, `Z_t`), associated beacon tag IDs, and elevation reference mode (`BM` or `WSEL`).
- **tblWSEL**: Water surface elevation time-series data for dynamic Z-coordinate interpolation.
- **tblInterpolatedTemp**: Timestamped temperature observations used for speed of sound calculations.
- **tblDetectionRaw**: Ingested raw receiver detection records (timestamps, microsecond counters, sequence numbers, SNR, Amplitude, NBW).
- **tblMetronomeUnfiltered / tblMetronomeFiltered**: Metronome beacon transmission enumerations and initial lag rankings.
- **tblMetronomeSecondFiltered**: Multipath-filtered beacon transmissions used for master clock reference.
- **tblDetectionFilterPrimary / tblDetectionFilterSecondary**: Multipath classification training and prediction outputs.
- **tblDetectionClockFixed**: Clock-drift corrected detection timestamps and temperature-interpolated speed of sound values.
- **tblPositions_Deng**: 3D TDOA positioning solutions calculated using Daniel Deng's exact solution method (storing centroids, candidate solutions A/B, time of arrival, and convex hull bounds checks).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
