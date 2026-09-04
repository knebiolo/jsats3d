# Running jsats3d

This guide provides quickstart instructions for running `jsats3d` workflows.

## Environment Setup

Install `jsats3d` in development mode:

```bash
pip install -e .[dev]
```

---

## Standard Processing Pipeline

```python
import jsats3d as js

# Step 1: Create a standardized SQLite project database
js.create_project_db("./data", "project.db")

# Step 2: Configure study parameters
js.set_study_parameters(
    utc_conv=-5,
    bm_elev=100.0,
    bm_elev_units="feet",
    output_units="meters",
    masterReceiver="REC_01",
    synch_time_start="2026-05-01 00:00:00",
    synch_time_end="2026-06-01 00:00:00",
    dbName="./data/project.db"
)

# Step 3: Ingest raw Teknologic receiver detections
js.acoustic_data_import(
    site="REC_01",
    recType="Teknologic",
    rawDataFiles="./data/raw_rec01",
    projectDB="./data/project.db"
)

# Step 4: Perform clock drift correction
clock_obj = js.clock_fix_object(
    curr_receiver="REC_02",
    receiver_list=["REC_01", "REC_02", "REC_03", "REC_04"],
    projectDB="./data/project.db",
    scratchWS="./scratch",
    figureWS="./figures"
)
js.clock_fix(clock_obj)

# Step 5: Solve 3D TDOA positions using Daniel Deng's exact method
pos_solver = js.position(
    tag="TAG_1001",
    resolved_clocks=["REC_01", "REC_02", "REC_03", "REC_04"],
    projectDB="./data/project.db",
    outputWS="./output",
    figureWS="./figures"
)
pos_solver.Deng()

# Step 6: 3D Kernel Density Utilization Estimation & Volume Plot
kde = js.kernels("Deng", "./data/project.db", "./output", tag_ID="TAG_1001")
kde.plot()
```

---

## Testing & Quality Assurance

Run the complete test suite:

```bash
pytest tests
```

Run ruff linter check:

```bash
ruff check jsats3d tests
```
