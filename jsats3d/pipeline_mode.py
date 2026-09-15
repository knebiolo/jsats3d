"""Select legacy or ATS-2025 processing at the detection boundary."""
from dataclasses import dataclass

import pandas as pd


LEGACY_SIGNAL_FIELDS = ("SNR", "NBW", "FreqOff")
ATS_REQUIRED_FIELDS = ("dateTime", "tagCode", "amp", "receiverName")
LEGACY_OUTPUT_FIELDS = (
    "timeStamp",
    "seconds",
    "Tag_ID",
    "Rec_ID",
    "FreqOff",
    "Amplitude",
    "NBW",
    "SNR",
    "Valid",
    "Pascals",
    "Celsius",
)


@dataclass(frozen=True)
class PipelineMode:
    name: str
    reason: str
    missing_fields: tuple


def _non_null_fields(data, fields):
    return [field for field in fields if field in data and data[field].notna().any()]


def detect_pipeline_mode(data):
    """Detect mode from schema and populated signal fields.

    Legacy mode requires all legacy classifier fields to exist and contain data.
    ATS-2025 mode accepts the reduced processed-deliverable schema.
    """
    has_ats_schema = all(field in data for field in ATS_REQUIRED_FIELDS)
    has_legacy_schema = all(field in data for field in ("timeStamp", "Tag_ID", "Rec_ID"))
    if not has_ats_schema and not has_legacy_schema:
        raise ValueError("Cannot identify detection schema")

    missing_legacy = tuple(
        field for field in LEGACY_SIGNAL_FIELDS
        if field not in data or not data[field].notna().any()
    )
    if not missing_legacy:
        return PipelineMode("legacy", "legacy signal fields are populated", ())
    return PipelineMode(
        "ats_2025",
        "legacy signal fields are unavailable or entirely NULL",
        missing_legacy,
    )


def select_pipeline_mode(data, requested="auto"):
    """Return mode or fail if an explicit request conflicts with data."""
    if requested not in ("auto", "legacy", "ats_2025"):
        raise ValueError("Unknown pipeline mode: %s" % requested)
    detected = detect_pipeline_mode(data)
    if requested == "auto":
        return detected
    if requested != detected.name:
        raise ValueError(
            "Requested %s mode, but data supports %s mode: %s"
            % (requested, detected.name, detected.reason)
        )
    return detected


def to_legacy_detection_shape(data):
    """Map ATS detection columns to legacy names without fabricating metrics."""
    result = data.rename(columns={
        "dateTime": "timeStamp",
        "tagCode": "Tag_ID",
        "amp": "Amplitude",
        "receiverName": "Rec_ID",
    }).copy()
    if "timeStamp" not in result:
        raise ValueError("Detection data lacks dateTime/timeStamp")
    result["timeStamp"] = pd.to_datetime(result["timeStamp"], errors="coerce")
    result["seconds"] = result["timeStamp"].astype("int64") / 1e9
    result["Tag_ID"] = result["Tag_ID"].astype(str).str.strip()
    result["Rec_ID"] = result["Rec_ID"].astype(str).str.strip()
    for field in ("FreqOff", "NBW", "SNR", "Pascals", "Celsius"):
        if field not in result:
            result[field] = pd.NA
    if "Valid" not in result:
        result["Valid"] = True
    return result[[field for field in LEGACY_OUTPUT_FIELDS if field in result]]