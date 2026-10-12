"""Team codes. nflverse codes (LA for the Rams) are used everywhere inside the code;
display codes (LAR) only on pages. Aliases accept the common historical and app spellings."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import CONFIG_DIR

ALIASES = {"LAR": "LA", "STL": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV", "OAK": "LV",
           "SD": "LAC", "SDG": "LAC", "KAN": "KC", "GNB": "GB", "NWE": "NE", "NOR": "NO",
           "SFO": "SF", "TAM": "TB", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}


@lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "teams.csv").set_index("team")


def code(team: str) -> str:
    """Normalize any accepted spelling to the nflverse code. Raises on unknown teams."""
    t = str(team).strip().upper()
    t = ALIASES.get(t, t)
    if t not in table().index:
        raise ValueError(f"unknown team code: {team!r}")
    return t


def display(team: str) -> str:
    return str(table().loc[code(team), "display"])


def name(team: str) -> str:
    return str(table().loc[code(team), "name"])


def normalize(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Map historical franchise codes in the given columns to current nflverse codes."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(ALIASES)
    return df
