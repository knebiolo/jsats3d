# -*- coding: utf-8 -*-
"""Backward compatibility module for jsats3d.

Submodules are now split into db, ingest, sync, multipath, positioning, and density.
"""

from .db import avg_temp, create_project_db, set_study_parameters, temp_interpolator
from .density import kernels
from .ingest import acoustic_data_import, study_data_import, teknologic_import
from .multipath import multipath_2, multipath_classifier, multipath_data_management, multipath_data_object
from .positioning import position, positions_data_management, sos, sos_apply
from .sync import beacon_epoch, clock_fix, clock_fix_object, epoch_fix_data_management

__all__ = [
    "create_project_db",
    "set_study_parameters",
    "temp_interpolator",
    "avg_temp",
    "study_data_import",
    "teknologic_import",
    "acoustic_data_import",
    "beacon_epoch",
    "clock_fix_object",
    "clock_fix",
    "epoch_fix_data_management",
    "multipath_data_object",
    "multipath_2",
    "multipath_data_management",
    "multipath_classifier",
    "sos",
    "sos_apply",
    "position",
    "positions_data_management",
    "kernels",
]
