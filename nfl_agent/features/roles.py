"""Evidence-based role-change flags for the usage-history view.

A flag is raised at game i when the mean of the player's last 3 games (i-2..i)
differs from the mean of the 8 games before them by at least `min_diff` AND by
more than 2 standard errors (Welch-style). This is a statistical flag, not a
confirmed depth-chart change: injuries during a game, blowouts or scheme can
produce the same pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def role_change_flags(history: pd.DataFrame, col: str, recent: int = 3, base: int = 8,
                      min_diff: float = 0.10) -> pd.DataFrame:
    h = history.sort_values("kickoff_utc").reset_index(drop=True)
    x = h[col].to_numpy(float)
    rows = []
    for i in range(recent + base - 1, len(h)):
        r = x[i - recent + 1:i + 1]
        b = x[i - recent - base + 1:i - recent + 1]
        r, b = r[~np.isnan(r)], b[~np.isnan(b)]
        if len(r) < recent or len(b) < base // 2:
            continue
        diff = r.mean() - b.mean()
        se = np.sqrt(r.var(ddof=1) / len(r) + b.var(ddof=1) / len(b)) if len(r) > 1 and len(b) > 1 else np.inf
        if abs(diff) >= min_diff and abs(diff) > 2 * se:
            rows.append(dict(game_id=h.loc[i, "game_id"], kickoff_utc=h.loc[i, "kickoff_utc"], metric=col,
                             recent_mean=r.mean(), prior_mean=b.mean(), diff=diff,
                             direction="up" if diff > 0 else "down"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # keep the first flag of each consecutive run so one change is not marked 3 times
    keep, last_idx = [], -10
    idx_map = {g: i for i, g in enumerate(h["game_id"])}
    for _, r in out.iterrows():
        j = idx_map[r["game_id"]]
        if j - last_idx > recent:
            keep.append(True)
        else:
            keep.append(False)
        last_idx = j
    return out[keep].reset_index(drop=True)
