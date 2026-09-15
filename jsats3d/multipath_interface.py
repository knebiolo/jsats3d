"""Pluggable multipath filtering contracts for legacy and ATS-2025 data."""
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd


@dataclass(frozen=True)
class FilterResult:
    """Filtered rows plus auditable stage accounting."""

    data: pd.DataFrame
    method: str
    input_count: int
    retained_count: int
    rejected_count: int
    rejection_reason: str


class MultipathFilter(Protocol):
    method: str

    def filter(self, detections: pd.DataFrame) -> FilterResult:
        ...


def require_features(detections, features):
    """Fail before filtering when approved features are unavailable."""
    missing = [feature for feature in features if feature not in detections]
    if missing:
        raise ValueError("Missing multipath features: %s" % sorted(missing))


class CallableMultipathFilter:
    """Adapt an existing legacy or external filter without hiding its method."""

    def __init__(self, function: Callable[[pd.DataFrame], pd.DataFrame], method: str):
        self.function = function
        self.method = method

    def filter(self, detections):
        input_count = len(detections)
        filtered = self.function(detections.copy())
        if not isinstance(filtered, pd.DataFrame):
            raise TypeError("Multipath filter must return a pandas DataFrame")
        if len(filtered) > input_count:
            raise ValueError("Multipath filter returned more rows than it received")
        return FilterResult(
            data=filtered,
            method=self.method,
            input_count=input_count,
            retained_count=len(filtered),
            rejected_count=input_count - len(filtered),
            rejection_reason="filter_rejected" if len(filtered) < input_count else "none",
        )


class UnfilteredBaseline:
    """Explicit baseline for diagnostics before a filter is approved."""

    method = "unfiltered_baseline"

    def filter(self, detections):
        return FilterResult(
            data=detections.copy(),
            method=self.method,
            input_count=len(detections),
            retained_count=len(detections),
            rejected_count=0,
            rejection_reason="none",
        )