"""Command line: python -m nfl_lab <command> ...

  ingest     download nflverse files into data/raw (cached; --refresh re-downloads completed seasons)
  build      build data/tables from the cache
  inventory  rows per source and season, tag timestamps, missing files
  features   pregame features for one game, e.g. features --home LAR --away BUF --week 5 --season 2026
  devig      fair probabilities from a price pair, e.g. devig -- -175 +145
  props      parse data/props.csv and show fair probabilities
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import config, teams


def _game_id(season: int, week: int, away: str, home: str) -> str:
    return f"{season}_{week:02d}_{teams.code(away)}_{teams.code(home)}"


def cmd_ingest(a):
    from .data import ingest
    rs = ingest.ingest(refresh=a.refresh)
    bad = [r for r in rs if r.status not in ("ok", "cached", "empty")]
    print(f"{len(rs)} files: " + ", ".join(f"{s}={sum(r.status == s for r in rs)}"
                                             for s in sorted({r.status for r in rs})))
    for r in bad:
        print(f"  {r.key} {r.season}: {r.status} {r.message}")
    return 0


def cmd_build(a):
    from .data import pipeline
    t = pipeline.build()
    for k, v in t.items():
        print(f"{k:18s} {len(v):>8,d} rows")
    return 0


def cmd_inventory(a):
    from .data import ingest
    from .data.sources import INGEST_PLAN, SOURCES
    files = ingest.current_files()
    rows = []
    for (k, s), r in sorted(files.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        rows.append(dict(source=k, season=s or "all", rows=r["n_rows"], retrieved=r["retrieved_at"][:16],
                         nflverse_updated=(r.get("source_last_updated") or "")[:16], sha=r["sha256"][:8]))
    print(pd.DataFrame(rows).to_string(index=False))
    missing = [r for r in ingest.read_manifest() if r["status"] == "not_published"]
    for r in missing:
        print(f"not published: {r['key']} {r['season']}")
    print(f"\nlive season {config.LIVE_SEASON}; participation is published after the postseason "
          f"(latest season with a file: {max((s for (k, s) in files if k == 'participation'), default=None)})")
    return 0


def cmd_features(a):
    from .data import pipeline
    from .features import bundle
    gid = _game_id(a.season, a.week, a.away, a.home)
    names = ["games", "team_view", "context", "team_games", "team_pressure", "player_games", "player_pfr_rush"]
    T = {n: pipeline.load_table(n) for n in names}
    f = bundle.game_features(T, gid)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"{gid}  asof {f['asof']['asof'].iloc[0]}  (features use only rows available before this time)\n")
    print(f["team_view"][["team", "opponent", "is_home", "rest_days", "team_spread", "implied_points"]].to_string(index=False))
    print()
    cols = ["team", "eff_games", "off_pass_epa", "off_rush_epa", "def_pass_epa", "def_rush_epa", "off_proe",
            "off_sec_per_play", "off_sack_rate", "off_pressure_rate_pfr", "def_pressure_rate_pfr", "off_rz_td_rate",
            "off_giveaway_rate"]
    print(f["team"][[c for c in cols if c in f["team"]]].round(3).to_string(index=False))
    print()
    p = f["players"]
    p = p[(p["games_this_season"] > 0) & (p["position"] != "QB")]
    p = p.sort_values(["team", "target_share"], ascending=[True, False])
    cols = ["team", "player_name", "position", "games_this_season", "snap_share", "route_part_est", "target_share",
            "tprr_est", "adot", "air_yards_share", "rz_target_share", "carry_share", "i10_carry_share", "catch_rate",
            "yprr_est", "yds_per_carry", "yaco_per_att"]
    print(p[[c for c in cols if c in p]].round(3).to_string(index=False))
    return 0


def cmd_devig(a):
    from .features import market
    q = [market.parse_price(x)[0] for x in a.prices]
    for m in ("multiplicative", "additive", "power", "shin"):
        print(f"{m:15s}", " ".join(f"{v:.4f}" for v in market.devig(q, m)))
    return 0


def cmd_props(a):
    from .data import lines
    p = lines.fair_probs(lines.load_props())
    pd.set_option("display.width", 220)
    print(p[["player", "team", "stat", "line", "source", "odds_kind", "fair_over", "fair_under", "market_note"]]
          .round(3).to_string(index=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nfl_lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(fn=cmd_ingest)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("inventory").set_defaults(fn=cmd_inventory)
    s = sub.add_parser("features")
    for k in ("home", "away"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--week", type=int, required=True)
    s.add_argument("--season", type=int, default=config.LIVE_SEASON)
    s.set_defaults(fn=cmd_features)
    s = sub.add_parser("devig")
    s.add_argument("prices", nargs="+")
    s.set_defaults(fn=cmd_devig)
    sub.add_parser("props").set_defaults(fn=cmd_props)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
