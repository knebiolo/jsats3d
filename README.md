# jsats3d

jsats3d was developed by Kleinschmidt to assist researchers with cleaning and managing small scale acoustic telemetry studies (<1000 tagged individuals, <20 receivers) The resulting dataset represents a standardized, repeatable and defensible product. This notebook carries out every task one may have when analyzing acoustic telemetry data, including: creating a standardized technology agnostic project database, importing raw data directly from receiver downloads, identifying and removing multipath, synchronizing clocks, positioning fish in 3D space, and visualizing resultant point clouds with kernel density functions. In future iterations of the software, we will replace the functional sample scripts with a Jupyter Notebook to guide users through training, classification, and data preparation.

Just open up the Project Notebook and get started

For a local 2025 clock-filter experiment, set `segmented_clock_dbscan = true` in
`[ats_2025_pipeline]`. This splits the observed beacon residual curve at changes
of at least 50 ms (a step-free beacon gap does not split it; the correction is
linearly interpolated across the gap), then applies one
DBSCAN pass per segment with the actual final timestamp included. Segments with
one or two knots are reported and retained without density screening. Within
multi-point segments, fish clock corrections retain linear interpolation. At a
jump, endpoint residuals are held constant up to the midpoint between the last
pre-jump and first post-jump beacon observations; a single-knot segment uses a
constant residual only over its estimated segment. These midpoint jump times
are assumptions, not measured events. The option is experimental and defaults to
false, preserving the existing five-pass 2019/2025 behavior. Use a separate
database and regenerate clock-fixed and downstream results when changing it.
