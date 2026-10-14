"""The Guesser method: how the Bills-Rams $20 entries were built, applied to every game. No game simulation.

For each standard PrizePicks line in a game:
  recent    games over (or under) the line in the player's last 4 games this season, pulled toward 50%:
            (hits + 2.5) / (games + 5)                                         (the original src/ev.py)
  market    the no-vig sportsbook chance if a price is on the row, else 50% (a PrizePicks line is set near even)
  matchup   this game's matchups (picks/matchups.py): pass rush vs offensive line, deep receivers under
            pressure, receiving yards the defense allows to the position, pass and run defense, game script,
            red-zone defense; capped at 8 points (the original's hand nudges ran -1 to +8), reasons written out
  estimate  (recent + matchup + market) / 2, exactly the original formula ("half our read, half the market");
            the side with the higher estimate is the pick
Confidence tiers come from the estimate. Players with fewer than 3 games this season are shown but kept out of
lineups (too small a sample, like Parkinson on the original board).

Entries use one game only: the top picks by estimate (Lean, 52.5%, or better), one per player, both teams. Like the
original, an entry's chance multiplies the picks' estimates. The Guesser puts a pretend $20 on each game, split by
plan() across up to 3 entries (2 to 6 picks, Power or Flex) to grow a pretend bankroll fastest; the play is frozen
at kickoff, and the Lineups tab shows exactly that play.
Only stats from before kickoff are used.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, teams
from .data import lines, live
from .features import bundle
from .features.asof import visible
from .picks import matchups, oldmodel, payouts

RECENT_N, PRIOR = 4, 5
MATCHUP_CAP = 0.08
NEWS_CAP = 0.10                            # matchup + news, before the estimate is halved with the market
MIN_GAMES = 3
MIN_PICK = 0.525                           # Lean or better: a coin flip is not a pick
TIERS = [(0.60, "High"), (0.575, "Med-High"), (0.55, "Medium"), (0.525, "Lean"), (0.0, "Coin flip")]
BUDGET = 20                                # the Guesser puts $20 on each game
REF_BANKROLL = 60                          # ...splitting it as if it were a third of a pretend $60 roll (see plan)
MAX_ENTRIES, MIN_STAKE = 3, 2
OUT_DIR = config.ROOT / "output" / "games"
LIVE_DIR = config.BACKTESTS_DIR / "live"
PP_STAT = {"rec_yds": "rec yds", "rush_yds": "rush yds", "pass_yds": "pass yds", "receptions": "receptions",
           "anytime_td": "rush+rec TDs", "rush_rec_tds": "rush+rec TDs", "rush_rec_yds": "rush+rec yds",
           "pass_tds": "pass TDs", "pass_att": "pass attempts", "completions": "completions", "rush_att": "rush attempts",
           "ints": "INTs", "pass_rush_yds": "pass+rush yds"}


def recent_rate(hits: int, games: int) -> float:
    """Hits at the line over his last games, pulled toward 50% (src/ev.py shrink)."""
    return (hits + PRIOR * 0.5) / (games + PRIOR)


def estimate(hits: int, games: int, matchup: float, market: float) -> float:
    """The original formula: (recent + matchup + market) / 2, "half our read, half the market"."""
    return (recent_rate(hits, games) + matchup + market) / 2


def tier(p: float) -> str:
    return next(name for cut, name in TIERS if p >= cut)


def game_label(g) -> str:
    return f"{teams.display(g['away_team'])} at {teams.display(g['home_team'])}"


def slug(g) -> str:
    return (f"{int(g['season'])}-w{int(g['week']):02d}-{teams.display(g['away_team']).lower()}-"
            f"{teams.display(g['home_team']).lower()}")


def _board(props: pd.DataFrame, g) -> pd.DataFrame:
    """Standard PrizePicks lines for this game, captured in the week before kickoff."""
    if props is None or props.empty:
        return pd.DataFrame()
    ko = g["kickoff_utc"]
    pp = props[props["source"].str.lower().eq("prizepicks") & props["line_type"].eq("standard")
               & props["team"].isin([g["home_team"], g["away_team"]]) & props["stat"].isin(list(oldmodel.STAT_COLS))
               & props["game_id"].isin(["", g["game_id"]])
               & (props["captured_at"] < ko) & (props["captured_at"] > ko - pd.Timedelta(days=7))]
    return pp.sort_values("captured_at").drop_duplicates(["player", "team", "stat", "line"], keep="last")


def rate_game(T: dict, eng, props: pd.DataFrame, game_id: str, ov: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """One row per line: the pick, its estimate and the reasons. Returns (picks, injury note)."""
    from . import v0run
    games = T["games"].set_index("game_id")
    g = games.loc[game_id].copy()
    g["game_id"] = game_id
    ko, season = g["kickoff_utc"], int(g["season"])
    board = _board(props, g)
    if board.empty:
        return pd.DataFrame(), ""
    out, note = v0run._outs(T, g, ko)
    ov = ov if ov is not None else live.overrides(game_id)
    bo, starters = v0run.board_overrides(props, g)
    # a player on the saved board who has since been ruled out (injury report or your overrides) stays out
    hist = visible(T["player_games"], ko)
    out_names = set(hist.loc[hist["player_id"].isin(out), "player_name"].map(lines.norm_name))
    if len(ov):
        out_names |= set(ov.loc[ov["status"] == "out", "player"].map(lines.norm_name))
    bo = bo[~bo["player"].map(lines.norm_name).isin(out_names)]
    board = board[~board["player"].map(lines.norm_name).isin(out_names)]
    books = _book_prices(props, g)
    nudges = live.news(game_id, ko)
    act = v0run._active_with_overrides(T, g, ko, out, pd.concat([ov, bo], ignore_index=True), starters,
                                       quiet=set(bo["player"]))
    cands = pd.concat([a.assign(team=t) for t, a in act.items()], ignore_index=True)
    res = lines.resolve_players(board, cands)
    res = res.dropna(subset=["player_id"])
    pos = dict(zip(cands["player_id"], cands["position"]))
    pg = visible(T["player_games"], ko)
    pg = pg[(pg["season"] == season) & (pg["any_snaps"] > 0)]
    count_stats = set(config.load("stats")["count_stats"]["keys"])
    rows = []
    for _, r in res.iterrows():
        if r["stat"] in count_stats and float(r["line"]).is_integer():
            continue                                             # a push would void the leg; skip whole numbers
        h = pg[pg["player_id"] == r["player_id"]].sort_values("kickoff_utc").tail(RECENT_N)
        vals = oldmodel._value(h, r["stat"]).to_numpy() if len(h) else np.array([])
        n = len(vals)
        for side in ("over", "under"):
            if side == "under" and r["stat"] == "anytime_td":
                continue
            hits = int((vals > r["line"]).sum() if side == "over" else (vals < r["line"]).sum())
            recent = recent_rate(hits, n)
            fo = r.get("fair_over", np.nan)
            src = "price on the line"
            if fo is None or fo != fo:
                fo, src = books.get((lines.norm_name(r["player"]), r["stat"], float(r["line"])), (np.nan, "50%: no price"))
            mkt = (float(fo) if side == "over" else 1 - float(fo)) if fo == fo and fo is not None else 0.5
            rows.append(dict(game_id=game_id, player_id=r["player_id"], player=r["player"], team=r["team"],
                             position=pos.get(r["player_id"], ""), stat=r["stat"], line=float(r["line"]), side=side,
                             hits=hits, games=n, log=[float(x) for x in vals], recent=recent, market=mkt,
                             market_src=src, p_model=recent, captured_at=str(r["captured_at"])))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, note
    df = matchups.adjust(df, matchups.context(T, game_id, eng), pos, cap=MATCHUP_CAP)   # p_model = recent + matchup
    df = _apply_news(df, nudges)                                                         # + news (capped with matchup)
    df["p_model"] = (df["p_model"] + df["market"]) / 2                                   # half our read, half the market
    df = df.sort_values("p_model", ascending=False, kind="mergesort").drop_duplicates(["player_id", "stat", "line"])
    df["tier"] = df["p_model"].map(tier)
    return df.reset_index(drop=True), note


def _book_prices(props: pd.DataFrame, g) -> dict:
    """No-vig chance of More from sportsbook rows saved before kickoff, by (player, stat, line). Several books at the
    same line are averaged. Only exact line matches are used (a different line is a different bet)."""
    if props is None or props.empty or "fair_over" not in props:
        return {}
    ko = g["kickoff_utc"]
    bk = props[~props["source"].str.lower().isin(["prizepicks", "legacy_ev_py"])
               & props["team"].isin([g["home_team"], g["away_team"]]) & props["game_id"].isin(["", g["game_id"]])
               & (props["captured_at"] < ko) & (props["captured_at"] > ko - pd.Timedelta(days=7)) & props["fair_over"].notna()]
    if bk.empty:
        return {}
    bk = bk.sort_values("captured_at").drop_duplicates(["player", "stat", "line", "source"], keep="last")
    out = {}
    for (pl, st, ln), grp in bk.groupby([bk["player"].map(lines.norm_name), "stat", "line"]):
        out[(pl, st, float(ln))] = (float(grp["fair_over"].mean()), "no-vig " + ", ".join(sorted(set(grp["source"]))))
    return out


def _apply_news(df: pd.DataFrame, nudges: pd.DataFrame) -> pd.DataFrame:
    """Add hand nudges from data/news.csv to p_model (recent + matchup), keeping matchup + news within NEWS_CAP."""
    if nudges is None or nudges.empty or df.empty:
        return df
    df = df.copy()
    adj, reasons = [], []
    for _, r in df.iterrows():
        m = nudges[(nudges["player"].map(lines.norm_name) == lines.norm_name(r["player"]))
                   & nudges["stat"].isin(["*", r["stat"]])]
        sign = 1 if r["side"] == "over" else -1
        a = float((m["more_adj"] * sign).sum()) if len(m) else 0.0
        a = float(np.clip(r["p_matchup"] + a, -NEWS_CAP, NEWS_CAP) - r["p_matchup"])
        adj.append(a)
        reasons.append(list(r["reasons"]) + [("For: " if s * sign > 0 else "Against: ") + "News: " + why
                                             for s, why in zip(m["more_adj"], m["reason"]) if s])
    df["p_news"] = adj
    df["p_model"] = np.clip(df["p_model"] + df["p_news"], 0.01, 0.99)
    df["reasons"] = reasons
    return df


def _note(r) -> str:
    log = ", ".join(f"{v:g}" for v in r["log"]) if r["stat"] != "anytime_td" else \
        ", ".join("TD" if v else "no TD" for v in r["log"])
    word = "More" if r["side"] == "over" else "Less"
    return f"{r['tier']} · {word} in {r['hits']} of last {r['games']}" + (f" ({log})" if log else "")


def _flex_ev(ps: list[float]) -> float | None:
    _, flex = payouts.tables()
    n = len(ps)
    if n not in flex:
        return None
    return float(sum(np.prod([p if c else 1 - p for c, p in zip(combo, ps)]) * flex[n].get(sum(combo), 0)
                     for combo in itertools.product([0, 1], repeat=n)))


def _pool(picks: pd.DataFrame, min_pick: float | None = None) -> pd.DataFrame:
    """Picks a lineup may use: 3+ games this season, Lean or better, the best line per player."""
    if picks is None or picks.empty:
        return pd.DataFrame()
    pool = picks[(picks["games"] >= MIN_GAMES) & (picks["p_model"] >= (MIN_PICK if min_pick is None else min_pick))]
    return pool.sort_values("p_model", ascending=False, kind="mergesort").drop_duplicates("player_id")


def _legs_for(pool: pd.DataFrame, n: int) -> pd.DataFrame | None:
    """The top n picks; if they are all one team, the other team's best pick takes the last spot (PrizePicks rule)."""
    pick = pool.head(n)
    if len(pick) < n:
        return None
    if pick["team"].nunique() < 2:
        other = pool[~pool["team"].isin(pick["team"])].head(1)
        if other.empty:
            return None
        pick = pd.concat([pick.head(n - 1), other])
    return pick


def _leg(r) -> dict:
    return dict(player=r.player, player_id=r.player_id, team=teams.display(r.team), stat=r.stat, side=r.side,
                line=r.line, p_model=float(r.p_model), game_id=r.game_id, note=_note(r._asdict()), reasons=list(r.reasons))


def _compositions(total: int, k: int, lo: int):
    if k == 1:
        if total >= lo:
            yield (total,)
        return
    for a in range(lo, total - lo * (k - 1) + 1):
        for rest in _compositions(total - a, k - 1, lo):
            yield (a,) + rest


def plan(picks: pd.DataFrame, label: str = "", budget: int = BUDGET) -> dict:
    """How the Guesser splits its $20 on one game. Candidates: the top 2 to 6 picks, each as a Power or a Flex entry.
    Chances come from the picks' estimates, treated as independent like the original. The split (whole dollars, up
    to 3 entries, at least $2 each) maximizes expected log growth of a pretend $60 bankroll that must put $20 on this
    game: so most of the money goes on the entry most likely to pay and only a little on bigger payouts."""
    power, flex = payouts.tables()
    pool = _pool(picks)
    cands = []
    for n in range(2, 7):
        legs = _legs_for(pool, n)
        if legs is None:
            continue
        for play in ("Power", "Flex"):
            if play == "Flex" and n not in flex:
                continue
            cands.append((play, n, legs))
    if not cands:
        return {}
    keys = list(dict.fromkeys((r.player_id, r.stat, r.line) for _, _, legs in cands for r in legs.itertuples()))
    pk = {(r.player_id, r.stat, r.line): float(r.p_model) for _, _, legs in cands for r in legs.itertuples()}
    P = np.array([pk[k] for k in keys])
    outs = np.array(list(itertools.product([0, 1], repeat=len(keys))))
    prob = np.prod(np.where(outs == 1, P, 1 - P), axis=1)
    M = []
    for play, n, legs in cands:
        hits = outs[:, [keys.index((r.player_id, r.stat, r.line)) for r in legs.itertuples()]].sum(1)
        M.append(np.where(hits == n, power[n], 0.0) if play == "Power" else
                 np.array([flex[n].get(int(h), 0.0) for h in hits]))
    M = np.array(M)
    best = None
    for k in range(1, MAX_ENTRIES + 1):
        splits = np.array(list(_compositions(budget, k, MIN_STAKE)))
        if not len(splits):
            continue
        for sub in itertools.combinations(range(len(cands)), k):
            R = splits @ M[list(sub)]
            obj = np.log(REF_BANKROLL - budget + R) @ prob
            i = int(np.argmax(obj))
            if best is None or obj[i] > best[0] + 1e-12:
                best = (obj[i], sub, splits[i], R[i])
    _, sub, stakes, R = best
    entries = []
    for j, st in sorted(zip(sub, stakes), key=lambda t: (-t[1], cands[t[0]][1])):
        play, n, legs = cands[j]
        p_all = float(np.prod(legs["p_model"]))
        ev = float(M[j] @ prob)
        entries.append(dict(n=n, play_as=play, stake=int(st), game=label, basis="Guesser method", p_all=p_all,
                            p_paid=float(prob[M[j] > 0].sum()), best_ev=ev, recommended=bool(ev > 1.0),
                            min_power_mult=1 / p_all, legs=[_leg(r) for r in legs.itertuples()]))
    for e_, letter in zip(entries, "ABC"):
        e_["entry"] = letter
    return dict(budget=budget, entries=entries, expected_back=float(R @ prob), p_profit=float(prob[R > budget].sum()),
                p_lose_all=float(prob[R == 0].sum()), p_any_back=float(prob[R > 0].sum()),
                rule=f"split ${budget} to maximize expected log growth of a pretend ${REF_BANKROLL} bankroll "
                     f"(up to {MAX_ENTRIES} entries, ${MIN_STAKE}+ each)")


def lineups(picks: pd.DataFrame, label: str = "", sizes=(3, 4, 5, 6), min_pick: float = None) -> dict:
    """Entry A (3-pick Flex), Entry B (4-pick Power), then 5 and 6 picks: the top estimates, one per player."""
    power, _ = payouts.tables()
    out = {}
    pool = _pool(picks, min_pick)
    for n in sizes:
        pick = _legs_for(pool, n)
        if pick is None:
            continue
        ps = pick["p_model"].tolist()
        p_all = float(np.prod(ps))
        pw, fx = p_all * power[n], _flex_ev(ps)
        play = {3: "Flex", 4: "Power"}.get(n) or ("Power" if pw >= (fx or 0) else "Flex")   # A and B as on the original slip
        best = fx if play == "Flex" else pw
        entry = {3: "A", 4: "B"}.get(n)
        out[str(n)] = [dict(
            n=n, entry=entry, recommended=bool(best > 1.0), basis="Guesser method", play_as=play, game=label,
            p_all=p_all, p_independent=p_all, power_ev=pw, flex_ev=fx, best_ev=best, min_power_mult=1 / p_all,
            legs=[_leg(r) for r in pick.itertuples()])]
    return out


def leg_text(x: dict) -> str:
    return f"{x['player']} {PP_STAT.get(x['stat'], x['stat'])} {'more' if x['side'] == 'over' else 'less'} {x['line']:g}"


def write_bankroll(g, pl: dict, now: pd.Timestamp) -> Path | None:
    """The Guesser's $20 on this game, split as plan() says. Never rewritten once the game has started."""
    path = LIVE_DIR / f"{slug(g)}-bankroll.json"
    ko = pd.Timestamp(g["kickoff_utc"])
    if path.exists() and now >= ko:
        return None                                        # frozen: the game has started
    if not pl or not pl.get("entries"):
        path.unlink(missing_ok=True)                       # no play any more (before kickoff): no bet
        return None
    power, flex = payouts.tables()
    entries = [dict(id=x["entry"], type=x["play_as"].lower(), stake=float(x["stake"]), legs=[leg_text(l) for l in x["legs"]],
                    legs_detail=x["legs"], pays=({str(k): v for k, v in flex[x["n"]].items()} if x["play_as"] == "Flex"
                                                 else {str(x["n"]): power[x["n"]]}))
               for x in pl["entries"]]
    d = dict(game=g["game_id"], label=game_label(g), kickoff_utc=f"{ko:%Y-%m-%d %H:%M} UTC",
             frozen_at=f"{now:%Y-%m-%d %H:%M} UTC", bankroll=float(pl["budget"]), reserve_not_bet=0.0, paper_money=True,
             picks_source="Guesser method (nfl_lab/guesser.py): recent form + market + matchups, same game; "
                          "stakes from guesser.plan",
             split_rule=pl["rule"], entries=entries,
             expected_return_per_dollar={f"{x['entry']}_{x['play_as'].lower()}{x['n']}": dict(estimate=round(x["best_ev"], 3))
                                         for x in pl["entries"]},
             expected_back=round(pl["expected_back"], 2), p_profit=round(pl["p_profit"], 3),
             p_lose_all=round(pl["p_lose_all"], 3), result=None)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, default=str))
    return path


def run_week(season: int, week: int, write: bool = True, only: str | None = None) -> dict:
    """Rate every game of the week that has PrizePicks lines; write per-game results and Guesser entries."""
    from . import v0run
    T = v0run.load_tables()
    eng = bundle.engines(T)
    props = lines.load_props()
    props = lines.fair_probs(props) if len(props) else props
    games = T["games"]
    gl = games[(games["season"] == season) & (games["week"] == week)].sort_values("kickoff_utc")
    now = pd.Timestamp.now(tz="UTC")
    out = {}
    for _, g in gl.iterrows():
        gid = g["game_id"]
        if only and only not in (gid, slug(g)):
            continue
        if now >= g["kickoff_utc"] and (OUT_DIR / f"{slug(g)}.json").exists():
            out[gid] = json.loads((OUT_DIR / f"{slug(g)}.json").read_text())   # started: keep what was frozen
            continue
        picks, note = rate_game(T, eng, props, gid)
        lu = lineups(picks, game_label(g))
        pl = plan(picks, game_label(g))
        ctx = matchups.context(T, gid, eng)
        rec = dict(game_id=gid, season=season, week=week, label=game_label(g), away=teams.display(g["away_team"]),
                   home=teams.display(g["home_team"]), kickoff_utc=str(g["kickoff_utc"]),
                   home_spread=float(-g["spread_line"]) if g["spread_line"] == g["spread_line"] else None,
                   total=float(g["total_line"]) if g["total_line"] == g["total_line"] else None,
                   injury_note=note, built_at=str(now), units=unit_table(ctx, g),
                   picks=[] if picks.empty else json.loads(picks.drop(columns=["reasons"]).assign(
                       reasons=picks["reasons"]).to_json(orient="records")),
                   lineups=lu, plan=pl)
        out[gid] = rec
        if write:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"{slug(g)}.json").write_text(json.dumps(rec, indent=1, default=str))
            write_bankroll(g, pl, now)
    return out


def unit_table(ctx: dict, g) -> list[dict]:
    """Unit vs unit: each offense against the defense it faces, ranks out of 32 (1 = best for that unit)."""
    tf = ctx["tf"]
    rows = []
    for team, opp in ((g["away_team"], g["home_team"]), (g["home_team"], g["away_team"])):
        if team not in tf.index or opp not in tf.index:
            continue
        o, d = tf.loc[team], tf.loc[opp]
        def rk(col, val, high_good):
            s = tf[col].dropna()
            return int((s > val).sum() + 1) if high_good else int((s < val).sum() + 1)
        for label, ocol, dcol, o_high_good in (
                ("EPA per dropback", "off_pass_epa", "def_pass_epa", True),
                ("EPA per rush", "off_rush_epa", "def_rush_epa", True),
                ("Explosive pass rate", "off_expl_pass_rate", "def_expl_pass_rate", True),
                ("Pressure rate (PFR)", "off_pressure_rate_pfr", "def_pressure_rate_pfr", False),
                ("Red-zone TD rate", "off_rz_td_rate", "def_rz_td_rate", True)):
            if ocol not in tf or dcol not in tf:
                continue
            # offense rank: 1 = best offense; defense rank: 1 = best defense (allows least / pressures most)
            orank = rk(ocol, o[ocol], o_high_good)
            drank = rk(dcol, d[dcol], not o_high_good) if label != "Pressure rate (PFR)" else rk(dcol, d[dcol], True)
            rows.append(dict(offense=teams.display(team), defense=teams.display(opp), stat=label,
                             off_value=float(o[ocol]), def_value=float(d[dcol]), off_rank=orank, def_rank=drank))
    return rows


def grade_week(season: int, week: int, stats: pd.DataFrame) -> list[str]:
    """Fill in results for this week's Guesser bets whose games have official stats. Returns graded game ids."""
    from .backtest import retro
    done = []
    for path in sorted(LIVE_DIR.glob(f"{season}-w{week:02d}-*-bankroll.json")):
        d = json.loads(path.read_text())
        if d.get("result") and not d["result"].get("provisional"):
            continue
        legs, missing = {}, False
        for en in d["entries"]:
            for x, text in zip(en.get("legs_detail", []), en["legs"]):
                v = retro.actual(stats, x["player_id"], week, x["stat"])
                if v is None:
                    missing = True
                    continue
                hit = v > x["line"] if x["side"] == "over" else v < x["line"]
                shown = ("scored" if v else "no TD") if x["stat"] == "anytime_td" else v
                legs[text] = dict(actual=shown, hit=bool(hit))
        if not legs or missing:
            continue
        d["result"] = dict(final="", graded_at=f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC", provisional=False,
                           source=f"nflverse stats_player_week_{season}", legs=legs)
        path.write_text(json.dumps(d, indent=2, default=str))
        done.append(d["game"])
    return done


RETRO_PATH = LIVE_DIR / "{slug}-guesser-retro.json"


def retro_board(game_id: str, prices: bool) -> pd.DataFrame:
    """The lines on file for a finished game (data/props.csv rows with its game_id, all captured before kickoff), as
    standard PrizePicks lines. prices=False drops any sportsbook price (market = 50%, as on a PrizePicks board);
    prices=True keeps the prices the original report collected."""
    import tempfile
    raw = pd.read_csv(lines.PROPS_PATH, dtype=str, keep_default_na=False)
    g = raw[raw["game_id"] == game_id].copy()
    g["has_price"] = (g["over_odds"] != "") | (g["under_odds"] != "")
    g = g.sort_values("has_price", ascending=False, kind="mergesort").drop_duplicates(["player", "team", "stat", "line"])
    g = g.drop(columns="has_price")
    g["source"], g["line_type"] = "prizepicks", "standard"
    if not prices:
        g["over_odds"], g["under_odds"], g["odds_kind"] = "", "", ""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "props.csv"
        g.to_csv(p, index=False)
        out = lines.load_props(p)
    return lines.fair_probs(out) if len(out) else out


def retro_predict(game_id: str) -> Path:
    """Before-kickoff test, step 1: rate a finished game with only what was known before kickoff and freeze the
    lineups to backtests/live/<slug>-guesser-retro.json. No results are read here."""
    from . import v0run
    T = v0run.load_tables()
    if (T["player_games"]["game_id"] == game_id).any():
        raise RuntimeError(f"{game_id} is already in data/tables: the test would see the result")
    eng = bundle.engines(T)
    g = T["games"].set_index("game_id").loc[game_id].copy()
    g["game_id"] = game_id
    variants = {}
    for name, prices in (("no_prices", False), ("with_prices", True)):
        board = retro_board(game_id, prices)
        picks, note = rate_game(T, eng, board, game_id)
        lu = lineups(picks, game_label(g))
        filled = {n: [dict(L, filled=True, basis="Guesser method, filled with coin flips")]
                  for n, (L,) in lineups(picks, game_label(g), min_pick=0.5).items() if n not in lu}
        variants[name] = dict(
            market=("50% for every line, as on a PrizePicks board (how Week 6 is rated)" if not prices else
                    "the sportsbook / Kalshi / Polymarket prices the original report collected before kickoff"),
            picks=[] if picks.empty else json.loads(picks.to_json(orient="records")), lineups=lu, filled=filled,
            filled_rule="sizes the model would not build (fewer players at Lean, 52.5%, or better), filled with the "
                        "next-best picks down to 50%, same order, one per player, 3+ games",
            injury_note=note)
    rec = dict(game_id=game_id, label=game_label(g), kickoff_utc=str(g["kickoff_utc"]),
               predicted_at=f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC",
               data_check=f"{game_id} not in data/tables (latest game in tables kicked off "
                          f"{T['player_games']['kickoff_utc'].max()}); stats, matchups and actives taken as of kickoff",
               lines=f"{len(retro_board(game_id, False))} lines on file in data/props.csv for this game, saved before kickoff",
               method="Guesser method (nfl_lab/guesser.py), same code as the Week 6 lineups", variants=variants, graded=None)
    path = Path(str(RETRO_PATH).format(slug=slug(g)))
    path.write_text(json.dumps(rec, indent=1, default=str))
    return path


def retro_grade(path: Path, stats: pd.DataFrame, final: str = "") -> dict:
    """Before-kickoff test, step 2: grade the frozen lineups against nflverse's official weekly stats."""
    from .backtest import retro
    rec = json.loads(Path(path).read_text())
    week = int(rec["game_id"].split("_")[1])
    for v in rec["variants"].values():
        v["lineups"] = retro.grade(v["lineups"], stats, week)
        v["filled"] = retro.grade(v.get("filled") or {}, stats, week)
    rec["graded"] = dict(at=f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC", final=final,
                         source=f"nflverse stats_player_week_{rec['game_id'][:4]}")
    Path(path).write_text(json.dumps(rec, indent=1, default=str))
    return rec
