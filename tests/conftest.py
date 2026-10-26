import numpy as np
import pandas as pd
import pytest

from flashwatch.cube import Cube


def make_cube(years=range(2001, 2007), ny=4, nx=5, seed=0):
    time = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    rng = np.random.default_rng(seed)
    T = len(time)

    def ar1(scale, mean):
        e = rng.normal(size=(T, ny, nx)).astype(np.float32)
        x = np.empty_like(e)
        x[0] = e[0]
        for i in range(1, T):
            x[i] = 0.95 * x[i - 1] + 0.3 * e[i]
        return mean + scale * x

    return Cube(time=time, rzsm=ar1(0.03, 0.3), sfsm=ar1(0.05, 0.25), t2m=ar1(3, 290), vpd=np.abs(ar1(0.5, 1.5)),
                precip=np.abs(ar1(0.002, 0.002)), evap=-np.abs(ar1(0.001, 0.002)), pet=-np.abs(ar1(0.001, 0.004)),
                ssrd=ar1(2e6, 2e7), masked=np.zeros((T, ny, nx), bool), land=np.ones((ny, nx), bool))


@pytest.fixture(scope="session")
def cube():
    return make_cube()
