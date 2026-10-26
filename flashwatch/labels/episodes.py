"""Group onset cell-days into episodes: space-time clusters of 8-connected cells whose onset
days are within +/-7 days. Episodes are the unit for event-level scoring and say how many
independent events the data holds.
"""

from __future__ import annotations

import numpy as np


def episodes(onset: np.ndarray, max_gap: int = 7) -> tuple[np.ndarray, int]:
    """Label onset events (time, lat, lon) with episode ids (0 = no onset). Returns (ids, count).

    Two onsets join when their cells are equal or 8-neighbours and their days differ by at
    most max_gap; episodes are the connected components of that relation (union-find).
    """
    t, y, x = np.nonzero(onset)
    n = len(t)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_cell: dict[tuple[int, int], list[int]] = {}
    for i in np.argsort(t, kind="stable"):
        by_cell.setdefault((y[i], x[i]), []).append(i)
    for i in range(n):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for j in by_cell.get((y[i] + dy, x[i] + dx), ()):
                    if j != i and abs(int(t[j]) - int(t[i])) <= max_gap:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
    roots = np.array([find(i) for i in range(n)], dtype=int)
    _, ids = np.unique(roots, return_inverse=True)
    out = np.zeros(onset.shape, dtype=np.int32)
    out[t, y, x] = ids + 1
    return out, int(ids.max() + 1) if n else 0
