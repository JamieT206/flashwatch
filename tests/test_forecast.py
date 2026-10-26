import numpy as np
import pytest

eccodes = pytest.importorskip("eccodes")

from flashwatch.forecast import N_DAYS, weekly  # noqa: E402


def write_grib(path, fields):
    """fields: list of (short_name, typeOfStatisticalProcessing or None, hdate, member, s0, s1, value)."""
    with open(path, "wb") as f:
        for name, stat, hdate, member, s0, s1, value in fields:
            h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            if stat is not None:
                eccodes.codes_set(h, "productDefinitionTemplateNumber", 11)
            for k, v in [("Ni", 3), ("Nj", 2), ("latitudeOfFirstGridPointInDegrees", 49.5),
                         ("longitudeOfFirstGridPointInDegrees", 256.5), ("latitudeOfLastGridPointInDegrees", 48.0),
                         ("longitudeOfLastGridPointInDegrees", 259.5), ("iDirectionIncrementInDegrees", 1.5),
                         ("jDirectionIncrementInDegrees", 1.5), ("shortName", name), ("dataDate", hdate),
                         ("number", member)]:
                eccodes.codes_set(h, k, v)
            if stat is not None:
                eccodes.codes_set(h, "typeOfStatisticalProcessing", stat)
                eccodes.codes_set(h, "startStep", s0)
            eccodes.codes_set(h, "endStep", s1)
            eccodes.codes_set_values(h, np.full(6, float(value)))
            eccodes.codes_write(h, f)
            eccodes.codes_release(h)


def test_weekly_from_daily_means_and_accumulations(tmp_path):
    fields = []
    for hdate in (20050613, 20060613):
        for member in (0, 1):
            for day in range(N_DAYS + 3):
                fields.append(("2t", 0, hdate, member, 24 * day, 24 * day + 24, 280 + day))
            for day in range(N_DAYS + 4):
                fields.append(("tp", 1, hdate, member, 0, 24 * day, 0.002 * day))      # accumulated: 2 mm/day
    write_grib(tmp_path / "x.grib", fields)
    wk, years, lats, lons = weekly(tmp_path / "x.grib")
    assert years == [2005, 2006]
    assert np.allclose(lons, [-103.5, -102.0, -100.5]) and np.allclose(lats, [49.5, 48.0])
    assert wk["t2m"].shape == (2, 2, 4, 2, 3)
    assert np.allclose(wk["t2m"][0, 0, :, 0, 0], [283, 290, 297, 304])     # mean of 280+day per week
    assert np.allclose(wk["tp"], 0.002)
