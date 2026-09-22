"""Produce exploratory DBSCAN diagnostics without filtering detections."""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


FEATURE_COLUMNS = ["lag_seconds", "relative_sigstr"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-k-distance", required=True)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-output")
    parser.add_argument(
        "--eps-values", nargs="+", type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    return parser.parse_args()


def make_diagnostics(features_csv, min_samples):
    if min_samples < 2:
        raise ValueError("min-samples must be at least 2")
    data = pd.read_csv(features_csv)
    missing = [column for column in FEATURE_COLUMNS if column not in data]
    if missing:
        raise ValueError("Missing diagnostic features: %s" % missing)
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=FEATURE_COLUMNS).copy()
    if data.empty:
        raise ValueError("No complete feature rows found")

    # Exploratory only: standardization makes unlike feature units comparable,
    # but its study-derived scale must not become a production transform.
    values = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0] = 1.0
    scaled = (values - means) / scales
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(scaled)
    distances, _ = neighbors.kneighbors(scaled)
    data["k_distance_exploratory"] = np.sort(distances[:, -1])
    summary = data.groupby("Rec_ID").agg(
        rows=("Rec_ID", "size"),
        epochs=("epoch_number", "nunique"),
        lag_median=("lag_seconds", "median"),
        lag_p95=("lag_seconds", lambda values: values.quantile(0.95)),
        relative_sigstr_median=("relative_sigstr", "median"),
        relative_sigstr_p05=("relative_sigstr", lambda values: values.quantile(0.05)),
    ).reset_index()
    k_distance = data[["source_rowid", "Tag_ID", "Rec_ID", "epoch_number", "k_distance_exploratory"]]
    return summary, k_distance, {"rows": len(data), "means": means, "scales": scales}


def sweep_diagnostics(features_csv, eps_values, min_samples_values=(2, 3, 4)):
    data = pd.read_csv(features_csv)
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=FEATURE_COLUMNS)
    values = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    scales = values.std(axis=0)
    scales[scales == 0] = 1.0
    scaled = (values - values.mean(axis=0)) / scales
    results = []
    for min_samples in min_samples_values:
        for eps in eps_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
            noise = labels == -1
            results.append({
                "eps_exploratory": eps,
                "min_samples_exploratory": min_samples,
                "input_rows": len(labels),
                "noise_rows": int(noise.sum()),
                "retained_rows": int((~noise).sum()),
                "noise_fraction": float(noise.mean()),
                "cluster_count": int(len(set(labels)) - (-1 in labels)),
            })
    return pd.DataFrame(results)


def main():
    args = parse_args()
    summary, k_distance, metadata = make_diagnostics(args.features_csv, args.min_samples)
    for path in (args.output_summary, args.output_k_distance):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    k_distance.to_csv(args.output_k_distance, index=False)
    print("Created: %s" % args.output_summary)
    print("Created: %s" % args.output_k_distance)
    print("Complete feature rows: %s" % metadata["rows"])
    print("Exploratory scaling means: %s" % metadata["means"].tolist())
    print("Exploratory scaling std: %s" % metadata["scales"].tolist())
    print("No detections filtered")
    if args.sweep:
        if not args.sweep_output:
            raise ValueError("--sweep-output is required with --sweep")
        sweep = sweep_diagnostics(args.features_csv, args.eps_values)
        os.makedirs(os.path.dirname(os.path.abspath(args.sweep_output)), exist_ok=True)
        sweep.to_csv(args.sweep_output, index=False)
        print("Created: %s" % args.sweep_output)


if __name__ == "__main__":
    main()