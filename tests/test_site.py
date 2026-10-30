import numpy as np
import pandas as pd

from flashwatch.site.build import with_fill


def test_fill_uses_the_forecast_model_where_it_has_a_value_and_the_land_model_elsewhere():
    issue = pd.to_datetime(["2012-07-01", "2012-07-03", "2012-07-05"])
    fill = pd.DataFrame({"year": 2012, "issue": issue, "iy": 4, "ix": 6, "E1_week1": [0.1, 0.2, 0.3],
                         "E1_week2": [0.01, 0.02, 0.03], "B0_week1": 0.05, "rz_pct": [50, 45, 40]})
    m2 = pd.DataFrame({"year": 2012, "issue": issue[[1]], "iy": 4, "ix": 6, "M2_week1": [0.9], "M2_week2": [0.8]})
    out = with_fill(m2, "M2", fill, "E1")
    assert len(out) == 3 and out.fc.tolist() == [False, True, False]
    np.testing.assert_allclose(out.FW_week1, [0.1, 0.9, 0.3])
    np.testing.assert_allclose(out.FW_week2, [0.01, 0.8, 0.03])
    assert out.rz_pct.tolist() == [50, 45, 40]
