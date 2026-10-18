"""The single gateway to history. Feature code reads past rows only through AsofIndex.

For each entity (player, team) rows are sorted by `available_at` and an exponentially weighted running
sum is kept (decay per entity game, extra discount per season boundary; the recursion is ported from
nfl_agent/features/pregame.py::_ew_state @90f63d6). A query at time `asof` sees the state after the
last row with available_at < asof (strict), so a game's own row, which becomes available hours after
its kickoff, can never be read for that game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIDE = "left"   # searchsorted side: 'left' + strict '<' semantics. Tests plant 'right' and must fail.


def to_us(s: pd.Series) -> np.ndarray:
    """Microseconds since the epoch (UTC) for a tz-aware datetime series."""
    return (s.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[us]").to_numpy().astype(np.int64))


def decay_for(half_life_games: float | None) -> float:
    return 1.0 if not half_life_games else 0.5 ** (1.0 / half_life_games)


class AsofIndex:
    def __init__(self, df: pd.DataFrame, entity: str | list[str], values: list[str], *,
                 half_life: float | None = None, season_discount: float = 1.0, weights: str | None = None,
                 time_col: str = "available_at"):
        self.entity = [entity] if isinstance(entity, str) else list(entity)
        self.values = list(values)
        self.decay = decay_for(half_life)
        self.season_discount = float(season_discount)
        d = df.dropna(subset=[time_col]).sort_values(self.entity + [time_col, "game_id"], kind="mergesort")
        x = d[self.values].astype(float).fillna(0.0).to_numpy()
        if weights is not None:
            x = x * d[weights].astype(float).fillna(0.0).to_numpy()[:, None]
        x = np.column_stack([x, np.ones(len(d))])            # last column: effective games
        times = to_us(d[time_col])
        seasons = d["season"].to_numpy(dtype=np.int64)
        keys = d[self.entity].astype(str).agg("|".join, axis=1).to_numpy() if len(self.entity) > 1 else \
            d[self.entity[0]].astype(str).to_numpy()
        states = np.zeros_like(x)
        self._slices: dict[str, tuple[int, int]] = {}
        start = 0
        n = len(d)
        while start < n:
            end = start
            while end < n and keys[end] == keys[start]:
                end += 1
            s = np.zeros(x.shape[1])
            prev = None
            for i in range(start, end):
                if prev is not None and seasons[i] != prev:
                    s = s * self.season_discount ** (seasons[i] - prev)
                s = s * self.decay + x[i]
                states[i] = s
                prev = seasons[i]
            self._slices[keys[start]] = (start, end)
            start = end
        self._times, self._seasons, self._states = times, seasons, states
        self._game_ids = d["game_id"].to_numpy()

    def _key(self, ent) -> str:
        return "|".join(map(str, ent)) if isinstance(ent, (tuple, list)) else str(ent)

    def state(self, ent, asof: pd.Timestamp, query_season: int) -> np.ndarray:
        """Weighted sums (values..., effective games) from rows with available_at < asof."""
        sl = self._slices.get(self._key(ent))
        if sl is None:
            return np.zeros(len(self.values) + 1)
        a, b = sl
        t = np.int64(pd.Timestamp(asof).tz_convert("UTC").value // 1000)
        idx = a + int(np.searchsorted(self._times[a:b], t, side=SIDE)) - 1
        if idx < a:
            return np.zeros(len(self.values) + 1)
        gap = max(0, int(query_season) - int(self._seasons[idx]))
        return self._states[idx] * self.season_discount ** gap

    def last_game(self, ent, asof: pd.Timestamp) -> str | None:
        sl = self._slices.get(self._key(ent))
        if sl is None:
            return None
        a, b = sl
        t = np.int64(pd.Timestamp(asof).tz_convert("UTC").value // 1000)
        idx = a + int(np.searchsorted(self._times[a:b], t, side=SIDE)) - 1
        return None if idx < a else str(self._game_ids[idx])

    def frame(self, ents, asof: pd.Timestamp, query_season: int) -> pd.DataFrame:
        rows = [self.state(e, asof, query_season) for e in ents]
        out = pd.DataFrame(rows, columns=self.values + ["eff_games"])
        out.insert(0, "entity", [self._key(e) for e in ents])
        return out


def visible(df: pd.DataFrame, asof: pd.Timestamp, time_col: str = "available_at") -> pd.DataFrame:
    """Rows of a history table that were available strictly before `asof`."""
    return df[df[time_col] < pd.Timestamp(asof)]
