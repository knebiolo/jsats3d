"""Preflight checks for receiver-clock residual analysis."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SyncReadiness:
    ready: bool
    reasons: tuple
    checked_rows: dict


def _missing_columns(frame, required):
    return tuple(column for column in required if column not in frame)


def assess_sync_readiness(
    detections,
    receivers,
    temperature,
    beacon_epochs,
    minimum_receivers=4,
):
    """Assess whether clock residuals can be computed without assumptions.

    This function does not estimate offsets, drift, sound speed, or beacon
    periods. It only checks required data coverage and structure.
    """
    reasons = []
    required_detection = ("Tag_ID", "Rec_ID", "seconds")
    required_receiver = ("Rec_ID", "X", "Y", "Z")
    required_temperature = ("timeStamp", "C")
    required_epoch = ("Tag_ID", "Rec_ID", "seconds", "transNo")
    for label, frame, required in (
        ("detections", detections, required_detection),
        ("receivers", receivers, required_receiver),
        ("temperature", temperature, required_temperature),
        ("beacon_epochs", beacon_epochs, required_epoch),
    ):
        missing = _missing_columns(frame, required)
        if missing:
            reasons.append("%s missing columns: %s" % (label, sorted(missing)))

    if not reasons:
        geometry = receivers.dropna(subset=["X", "Y", "Z"])
        if len(geometry) < minimum_receivers:
            reasons.append(
                "fewer than %s complete receiver geometries" % minimum_receivers
            )

        epoch_counts = beacon_epochs.groupby("transNo")["Rec_ID"].nunique()
        if epoch_counts.empty:
            reasons.append("no beacon epochs available")
        elif (epoch_counts < minimum_receivers).all():
            reasons.append(
                "no beacon epoch has at least %s receivers" % minimum_receivers
            )

        detection_times = pd.to_numeric(detections["seconds"], errors="coerce").dropna()
        temperature_times = pd.to_datetime(temperature["timeStamp"], errors="coerce")
        temperature_times = temperature_times.dropna().astype("datetime64[ns]").astype("int64") / 1e9
        if detection_times.empty or temperature_times.empty:
            reasons.append("missing detection or temperature time coverage")
        elif detection_times.min() < temperature_times.min() or detection_times.max() > temperature_times.max():
            reasons.append("temperature coverage does not span detection times")

    return SyncReadiness(
        ready=not reasons,
        reasons=tuple(reasons),
        checked_rows={
            "detections": len(detections),
            "receivers": len(receivers),
            "temperature": len(temperature),
            "beacon_epochs": len(beacon_epochs),
        },
    )