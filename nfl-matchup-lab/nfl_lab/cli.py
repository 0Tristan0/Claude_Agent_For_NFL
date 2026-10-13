"""Command line: python -m nfl_lab <command> ...

  ingest     download nflverse files into data/raw (cached; --refresh re-downloads completed seasons)
  build      build data/tables from the cache
  inventory  rows per source and season, tag timestamps, missing files
  features   pregame features for one game, e.g. features --home LAR --away BUF --week 5 --season 2026
  devig      fair probabilities from a price pair, e.g. devig -- -175 +145
  props      parse data/props.csv and show fair probabilities
  prizepicks import the PrizePicks NFL board (or --file board.json saved from a browser)
  project    v0 model for one game, e.g. project --home LAR --away BUF --week 5 --season 2026
  slate      v0 model for a whole week, e.g. slate --week 6
  check      v0 calibration check on 2024 (validation season)
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


def find_game(games: pd.DataFrame, season: int, week: int, home: str, away: str) -> str | None:
    """Game id for two teams in a week, whichever of them is listed at home."""
    pair = {teams.code(home), teams.code(away)}
    g = games[(games["season"] == season) & (games["week"] == week)]
    hit = g[g.apply(lambda r: {r["home_team"], r["away_team"]} == pair, axis=1)] if len(g) else g
    return None if hit.empty else str(hit["game_id"].iloc[0])


def cmd_features(a):
    from .data import lines, pipeline
    from .features import bundle
    names = ["games", "team_view", "context", "team_games", "team_pressure", "player_games", "player_pfr_rush"]
    T = {n: pipeline.load_table(n) for n in names}
    T["game_lines"] = lines.load_game_lines()
    gid = find_game(T["games"], a.season, a.week, a.home, a.away)
    if gid is None:
        print(f"No {a.season} week {a.week} game between {a.home} and {a.away} in the schedule.")
        return 1
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
    print("\nOpponent-adjusted (ridge; def = what the defense allows after adjusting for offenses faced)")
    print(f["opponent_adjusted"].round(3).to_string(index=False))
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
    if len(a.prices) < 2:
        print("Give every side of the market, e.g. devig -- -175 +145 (one-sided prices: see props.csv).")
        return 1
    try:
        q = [market.parse_price(x)[0] for x in a.prices]
    except ValueError as exc:
        print(exc)
        return 1
    if any(v is None for v in q):
        print("Every side needs a price.")
        return 1
    if sum(q) <= 1.0:
        print(f"Prices sum to {sum(q):.4f} (no vig), so every method just normalizes them.")
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


def cmd_prizepicks(a):
    import json
    from .data import prizepicks as PP
    try:
        feed = json.loads(open(a.file).read()) if a.file else PP.fetch()
    except (RuntimeError, OSError, ValueError) as exc:
        print(exc)
        return 1
    rows, skipped = PP.parse(feed, pd.Timestamp.now(tz="UTC"))
    for why, n in skipped.most_common():
        print(f"  skipped {n}: {why}")
    if rows.empty:
        print("No NFL lines the model can use were found on the board.")
        return 1
    PP.write(rows, raw=feed)
    print(f"{len(rows)} PrizePicks lines written to data/props.csv for {rows['team'].nunique()} teams:")
    print(rows.groupby(["stat", "line_type"]).size().to_string())
    print("Next: python -m nfl_lab slate --week <week>, then python3 src/site.py")
    return 0


def _print_run(res, show_players=True):
    from .sim import v0 as S
    pd.set_option("display.width", 220)
    if res.get("results"):
        note = next(iter(res["results"].values()))["proj"].get("injury_note")
        if note:
            print("injuries:", note)
    for gid, r in res["results"].items():
        g = r["proj"]["game"]
        print(f"\n=== {gid}  ({g.get('line_source', 'schedule')} lines: spread {g['spread_line']:+g} home, total {g['total_line']:g})")
        for t, tp in r["proj"]["teams"].items():
            pts = r["sim"].teams[t]["points"]
            print(f"  {teams.display(t):4s} implied {tp.implied_points:.1f}  sim mean {pts.mean():.1f}  plays {tp.plays:.0f}  pass rate {tp.pass_rate:.0%}")
        if show_players:
            sm = S.summary(r["sim"])
            keep = sm[sm["stat"].isin(["rec_yds", "receptions", "rush_yds", "pass_yds", "anytime_td"]) & (sm["mean"] > 0.3)]
            print(keep.sort_values(["team", "stat", "mean"], ascending=[True, True, False])
                  [["team", "player", "position", "stat", "mean", "q10", "median", "q90"]].round(2).to_string(index=False))
    legs = res["legs"]
    if len(legs):
        print("\nYour lines (model probability vs no-vig market):")
        print(legs[["player", "stat", "line", "side", "p_model", "p_market", "market_edge", "qualified_any", "source"]]
              .sort_values("p_model", ascending=False).round(3).to_string(index=False))
    for n, lst in sorted(res["best"].items()):
        for L in lst[:1]:
            legs_txt = "; ".join(f"{x['player']} {x['stat']} {x['side']} {x['line']:g} ({x['p_model']:.0%})" for x in L["legs"])
            tag = "RECOMMENDED" if L.get("recommended", True) else "not recommended (below break-even)"
            print(f"\n{n}-pick {L['play_as']} [{tag}]: all hit {L['p_all']:.1%} (independent {L['p_independent']:.1%}), "
                  f"Power ${(L['power_ev'] or 0):.2f}, Flex ${(L['flex_ev'] or 0):.2f}, skip under {L['min_power_mult']:.1f}x\n  {legs_txt}")
    if not any(res["best"].values()):
        print("\nNo PrizePicks lineup can be built: add PrizePicks lines (source prizepicks) to data/props.csv.")
    for p in res["pages"]:
        print("page:", p)


def cmd_project(a):
    from . import v0run
    from .data import pipeline
    gid = find_game(pipeline.load_table("games"), a.season, a.week, a.home, a.away)
    if gid is None:
        print(f"No {a.season} week {a.week} game between {a.home} and {a.away} in the schedule.")
        return 1
    _print_run(v0run.run_games([gid], a.sims))
    return 0


def cmd_slate(a):
    from . import v0run
    gids = v0run.week_games(a.season, a.week)
    if not gids:
        print(f"No {a.season} week {a.week} games with market lines yet.")
        return 1
    _print_run(v0run.run_games(gids, a.sims), show_players=False)
    return 0


def cmd_check(a):
    from . import v0run
    from .backtest import quick
    from .picks import calibration
    from .project import v0 as P
    T = v0run.load_tables()
    df = quick.run(T, a.season, n_sims=a.sims, progress=print, calibrate=not a.raw)
    rep = quick.report(df)
    cal = calibration.load()
    v0run.save_check(rep, a.season, dict(season=a.season, fit_seasons=P.fit_seasons(a.season), n_sims=a.sims,
                                         calibrated=not a.raw, calibration_fitted_on=(cal or {}).get("fitted_on"),
                                         weeks="3-18", lines="stand-in (median of last 6 games)"))
    pd.set_option("display.width", 220)
    print(rep["overall"])
    print(rep["by_stat"].round(4).to_string(index=False))
    print(rep["calibration"].round(3).to_string(index=False))
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
    s = sub.add_parser("prizepicks", help="import the PrizePicks NFL board into data/props.csv")
    s.add_argument("--file", help="a board saved from the PrizePicks feed in a browser (.json)")
    s.set_defaults(fn=cmd_prizepicks)
    s = sub.add_parser("project", help="v0 model for one game: projections, your lines, lineups, page")
    for k in ("home", "away"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--week", type=int, required=True)
    s.add_argument("--season", type=int, default=config.LIVE_SEASON)
    s.add_argument("--sims", type=int, default=20000)
    s.set_defaults(fn=cmd_project)
    s = sub.add_parser("slate", help="v0 model for every game of a week, with cross-game lineups")
    s.add_argument("--week", type=int, required=True)
    s.add_argument("--season", type=int, default=config.LIVE_SEASON)
    s.add_argument("--sims", type=int, default=20000)
    s.set_defaults(fn=cmd_slate)
    s = sub.add_parser("check", help="calibration check of v0 on a validation season (2024 or earlier)")
    s.add_argument("--season", type=int, default=2024)
    s.add_argument("--sims", type=int, default=2000)
    s.add_argument("--raw", action="store_true", help="skip calibration (report raw sim probabilities)")
    s.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
