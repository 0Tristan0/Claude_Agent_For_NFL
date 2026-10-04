"""Export model probability curves for upcoming games (for the Receiving Line Checker page).

For every WR/TE expected to play in a game kicking off within the next few days,
this writes P(receiving yards >= k) and P(receptions >= k) for every whole k,
so the page can price ANY line (64.5 yards, 4.5 receptions) without running
Python. Each curve comes with:

* a bootstrap band (10th-90th percentile when the player's own history is
  resampled), i.e. uncertainty about the probability itself;
* the workload baseline's curve, for comparison;
* evidence level, flags, and whether recent games are missing from the data.

The page never changes these numbers; it only reads them.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..evaluation import metrics as M
from ..features.pregame import player_features
from ..models.baselines import WorkloadBaseline
from ..models.simulate import stable_seed
from .service import ForecastService, utcnow

log = logging.getLogger(__name__)

KMAX_Y = 200        # yards curve covers k = 0..201 (P(Y >= 201) is the open tail)
KMAX_C = 16         # receptions curve covers k = 0..17


def ge_curve(x: np.ndarray, kmax: int) -> np.ndarray:
    """P(X >= k) for k = 0..kmax+1 from integer draws (values above kmax+1 pooled)."""
    x = np.clip(np.rint(np.asarray(x, float)), 0, kmax + 1).astype(int)
    counts = np.bincount(x, minlength=kmax + 2)
    return (counts[::-1].cumsum()[::-1] / len(x))[: kmax + 2]


def workload_curve(b: WorkloadBaseline, row: pd.DataFrame, kmax: int) -> np.ndarray:
    """The workload baseline's P(X >= k), k = 0..kmax+1."""
    mu = max(float(b.mean_of(row)[0] * b.scale), 1e-9)
    r = b.ratios[int(np.digitize([mu], b.edges)[0])]
    ks = np.arange(kmax + 2)
    return 1.0 - np.searchsorted(r, ks / mu, side="left") / len(r)


def bootstrap_bands(svc: ForecastService, row: pd.DataFrame, model, B: int, n_sims: int):
    """10th/90th percentile curves when the player's prior games are resampled (Poisson weights)."""
    pid, gid = row.iloc[0]["player_id"], row.iloc[0]["game_id"]
    sub = svc.features[svc.features["player_id"] == pid].sort_values("kickoff_utc").reset_index(drop=True)
    rng = np.random.default_rng(stable_seed("slate-boot", pid, gid))
    ys, cs = [], []
    for b in range(B):
        w = rng.poisson(1.0, size=len(sub)).astype(float)
        pf = player_features(sub, svc.params, svc.priors, mult={pid: w})
        pf = pf[pf["game_id"] == gid]
        new = row.copy()
        for c in pf.columns:
            if c not in ("player_id", "game_id"):
                new[c] = pf[c].to_numpy()
        new["ypt_ew"] = new["catch_rate_ew"] * new["ypr_ew"]
        d, _ = model.simulate_draws(new, n_sims, stable_seed("slate-bootsim", pid, gid, b))
        ys.append(ge_curve(d["yards"][0], KMAX_Y))
        cs.append(ge_curve(d["receptions"][0], KMAX_C))
    qy = np.quantile(np.stack(ys), [0.1, 0.9], axis=0)
    qc = np.quantile(np.stack(cs), [0.1, 0.9], axis=0)
    return qy[0], qy[1], qc[0], qc[1]


def _r(a, nd=3):
    """Round a decreasing tail curve and drop its trailing zeros (the page treats missing k as 0)."""
    v = [round(float(x), nd) for x in a]
    while len(v) > 1 and v[-1] == 0:
        v.pop()
    return v


def stale_games(games: pd.DataFrame, team_games: pd.DataFrame, team: str, kickoff) -> int:
    """The team's games before `kickoff` whose stats are not in the data (played already or still to come).

    A forecast cannot include these games, so a week-5 forecast made before the team's week-4 game
    is flagged even though that game has not kicked off yet.
    """
    with_stats = set(team_games.loc[team_games["has_stats"], "game_id"])
    g = games
    mine = g[((g["home_team"] == team) | (g["away_team"] == team)) & (g["kickoff_utc"] < kickoff)
             & ~g["game_id"].isin(with_stats)]
    return int(len(mine))


def calibration_summary(data_dir: Path) -> dict:
    """Pooled held-out calibration of the model's milestone probabilities (test season)."""
    p = data_dir / "eval" / "test_preds.parquet"
    info_p = data_dir / "eval" / "info.json"
    if not p.exists():
        return {}
    preds = pd.read_parquet(p)
    s = preds[preds["model"] == "structured"]
    bins = tuple(np.round(np.arange(0, 1.0001, 0.1), 2)[:-1]) + (1.0001,)
    out = {}
    for key, col, thr in (("yards", "receiving_yards", config.RECEIVING_YARD_MILESTONES),
                          ("rec", "receptions", config.RECEPTION_MILESTONES)):
        ps = np.concatenate([s[f"{key}_p_ge_{t}"].to_numpy(float) for t in thr])
        ys = np.concatenate([(s[col].to_numpy(float) >= t).astype(float) for t in thr])
        tab = M.calibration_table(ps, ys, bins=bins)
        out[key] = [{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v))
                     for k, v in r.items()} for r in tab.to_dict("records")]
    info = json.loads(info_p.read_text()) if info_p.exists() else {}
    out["test_season"] = info.get("test_season")
    out["n_player_games"] = info.get("n_test")
    return out


def build_slate(svc: ForecastService, days: float = 9.0, bootstrap: int = 20, n_sims: int = config.N_SIMS_APP,
                boot_sims: int = 4000, now=None) -> dict:
    now = now or pd.Timestamp(utcnow())
    q = svc.features[svc.features["is_query"]]
    q = q[(q["kickoff_utc"] > now) & (q["kickoff_utc"] <= now + timedelta(days=days))]
    q = q.sort_values(["kickoff_utc", "team", "tpg_ew"], ascending=[True, True, False])
    players, games = [], {}
    for i, (_, r) in enumerate(q.iterrows()):
        pid, gid = r["player_id"], r["game_id"]
        row = svc.row(pid, gid)
        bundle = svc.models_for_season(int(r["season"]))
        model = bundle["structured"]
        fc = svc.forecast(pid, gid, n_sims=n_sims, bootstrap=0)
        draws, _ = model.simulate_draws(row, n_sims, stable_seed(config.GLOBAL_SEED, pid, gid, 1.0))
        y_ge = ge_curve(draws["yards"][0], KMAX_Y)
        c_ge = ge_curve(draws["receptions"][0], KMAX_C)
        y_lo, y_hi, c_lo, c_hi = bootstrap_bands(svc, row, model, bootstrap, boot_sims)
        g = svc.games[svc.games["game_id"] == gid].iloc[0]
        games[gid] = {"id": gid, "away": g["away_team"], "home": g["home_team"], "week": int(g["week"]),
                      "kickoff": pd.Timestamp(g["kickoff_utc"]).isoformat()}
        last = fc.recent_games[-1] if fc.recent_games else None
        players.append({
            "id": pid, "name": r["player_name"], "pos": r["position"], "team": r["team"], "opp": r["opponent"],
            "game": gid, "kickoff": pd.Timestamp(r["kickoff_utc"]).isoformat(),
            "evidence": fc.evidence_level, "flags": fc.flags,
            "stale": stale_games(svc.games, svc.tg, r["team"], r["kickoff_utc"]),
            "last_game": last["game_id"] if last else None,
            "games_prior": fc.sample_sizes["prior_games_played"],
            "exp_targets": round(fc.model_params["expected_targets"], 2),
            "yds": {"median": fc.yards["median"], "mean": fc.yards["mean"], "i80": fc.yards["interval_80"],
                    "ge": _r(y_ge), "lo": _r(y_lo), "hi": _r(y_hi),
                    "base": _r(workload_curve(bundle["b3_yards"], row, KMAX_Y))},
            "rec": {"median": fc.receptions["median"], "mean": fc.receptions["mean"], "i80": fc.receptions["interval_80"],
                    "ge": _r(c_ge), "lo": _r(c_lo), "hi": _r(c_hi),
                    "base": _r(workload_curve(bundle["b3_rec"], row, KMAX_C))},
        })
        if (i + 1) % 25 == 0:
            log.info("slate: %d / %d players", i + 1, len(q))
    data_dir = config.PROCESSED_DIR if svc.data_mode == "real" else config.DEMO_DIR
    with_stats = set(svc.tg.loc[svc.tg["has_stats"], "game_id"])
    done = svc.games[svc.games["game_id"].isin(with_stats)]
    return {
        "generated_at": now.isoformat(),
        "data_mode": svc.data_mode,
        "data_through": pd.Timestamp(done["kickoff_utc"].max()).isoformat() if len(done) else None,
        "model_version": config.MODEL_VERSION,
        "kmax": {"yds": KMAX_Y + 1, "rec": KMAX_C + 1},
        "games": sorted(games.values(), key=lambda g: g["kickoff"]),
        "players": players,
        "calibration": calibration_summary(data_dir),
        "notes": {
            "conditional": "Probabilities assume the player plays (pick'em apps usually void a pick if he does not).",
            "coverage": "Wide receivers and tight ends; receiving yards and receptions only.",
            "validation": "Not tested against pick'em lines. On the 2025 held-out season the model showed no "
                          "demonstrated advantage over a simple workload baseline for 100+ yard milestones.",
        },
    }


def write_slate(slate: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slate, separators=(",", ":"), default=str))
    return out
