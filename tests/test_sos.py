# -*- coding: utf-8 -*-
"""Unit tests for speed of sound calculations in jsats3d."""

import numpy as np
import pandas as pd
from jsats3d.positioning import sos, sos_apply


def test_sos_known_temperatures():
    """Verify speed of sound interpolation values across standard water temperatures."""
    sos_10 = sos(10.0)
    sos_20 = sos(20.0)
    sos_25 = sos(25.0)

    # Speed of sound in fresh water (~1400-1500 m/s converted to ft/s is ~4700-4950 ft/s)
    assert 4700.0 < sos_10 < 4800.0
    assert 4800.0 < sos_20 < 4950.0
    assert sos_10 < sos_20 < sos_25


def test_sos_apply():
    """Verify DataFrame row-wise application of speed of sound interpolation."""
    df = pd.DataFrame({"Celsius": [15.0, 22.5]})
    result_0 = sos_apply(df.iloc[0])
    result_1 = sos_apply(df.iloc[1])

    assert np.isclose(result_0, sos(15.0))
    assert np.isclose(result_1, sos(22.5))
