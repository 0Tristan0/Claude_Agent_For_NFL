"""NFL receiving forecast lab: local Streamlit dashboard.

Educational, no-money forecasting of WR/TE receiving production. No sportsbook
data, wagering advice or bet placement. Run with:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nfl_agent import config  # noqa: E402
from nfl_agent.agent import router  # noqa: E402
from nfl_agent.data import imports as user_imports  # noqa: E402
from nfl_agent.data import inventory, pipeline  # noqa: E402
from nfl_agent.evaluation import walkforward as W  # noqa: E402
from nfl_agent.explain import narrative as N  # noqa: E402
from nfl_agent.features.roles import role_change_flags  # noqa: E402
from nfl_agent.forecast import service as S  # noqa: E402

st.set_page_config(page_title="NFL Receiving Forecast Lab", page_icon=":material/sports_football:", layout="wide")

# Validated categorical slots (see dataviz palette): model, baseline, third series. Muted for reference marks.
C_MODEL, C_BASE, C_THIRD, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
HELP = N.GLOSSARY


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading data and computing pregame features...")
def get_service() -> S.ForecastService:
    return S.ForecastService.load()


@st.cache_data(show_spinner="Simulating 20,000 games...", max_entries=64)
def get_forecast(player_id: str, game_id: str, pt: float, vol: float, built_at: str) -> S.Forecast:
    return get_service().forecast(player_id, game_id, pt, vol)


@st.cache_data
def load_eval(built_at: str, data_mode: str) -> dict:
    d = (config.PROCESSED_DIR if data_mode == "real" else config.DEMO_DIR) / "eval"
    out: dict = {}
    if not d.exists():
        return out
    for p in d.glob("*.parquet"):
        out[p.stem] = pd.read_parquet(p)
    if (d / "info.json").exists():
        out["info"] = json.loads((d / "info.json").read_text())
    return out


def style(fig: go.Figure, height: int = 320, ytitle: str = "", xtitle: str = "") -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), bargap=0.08)
    fig.update_xaxes(title=xtitle, showgrid=False)
    fig.update_yaxes(title=ytitle, gridwidth=1)
    return fig


def demo_banner(meta: dict) -> None:
    if meta.get("data_mode") == "demo":
        st.error("**DEMO MODE: synthetic data.** Teams 'DM*' and players 'DEMO ...' are fabricated for "
                 "testing. Nothing on this screen is a real NFL observation or forecast.", icon=":material/warning:")


def fmt_pct(p):
    return N.pct(p)


# ---------------------------------------------------------------- pages
def page_forecast(svc: S.ForecastService, ev: dict) -> None:
    st.header("Player forecast")
    st.caption("Receiving yards and receptions for wide receivers and tight ends. Forecasts assume the player "
               "is active and plays (participation is not modelled).")
    games = svc.game_options()
    seasons = sorted(games["season"].unique(), reverse=True)
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    season = c1.selectbox("Season", seasons, index=0)
    g_s = games[games["season"] == season]
    now = pd.Timestamp.now(tz="UTC")
    future = g_s[g_s["kickoff_utc"] > now]
    weeks = sorted(g_s["week"].unique())
    default_week = int(future["week"].min()) if len(future) else int(max(weeks))
    week = c2.selectbox("Week", weeks, index=weeks.index(default_week))
    g_w = g_s[g_s["week"] == week].sort_values("kickoff_utc")

    def status(r) -> str:
        if r.completed:
            return ""
        return " · upcoming" if r.kickoff_utc > now else " · kicked off, not final"
    labels = {r.game_id: f"{r.away_team} @ {r.home_team} · {pd.Timestamp(r.kickoff_utc).tz_convert(config.KICKOFF_TZ):%a %b %d %H:%M} ET"
                         f"{status(r)}" for r in g_w.itertuples()}
    gids = list(labels)
    first_up = next((r.game_id for r in g_w.itertuples() if r.kickoff_utc > now), gids[0])
    game_id = c3.selectbox("Game", gids, index=gids.index(first_up), format_func=labels.get)
    players = svc.players_for_game(game_id)
    if players.empty:
        st.info("No forecastable WR/TE for this game.")
        return
    plabels = {r.player_id: f"{r.player_name} ({r.position}, {r.team})" for r in players.itertuples()}
    default_pid = st.session_state.get("player_id")
    pids = list(plabels)
    pid = c4.selectbox("Player", pids, index=pids.index(default_pid) if default_pid in pids else 0,
                       format_func=plabels.get)
    st.session_state["player_id"] = pid

    with st.expander("Hypothetical scenario (analyst assumption, not a validated causal effect)"):
        s1, s2 = st.columns(2)
        pt = s1.slider("Playing time vs. his usual", 0.3, 1.3, 1.0, 0.05,
                       help="Scales expected targets proportionally. Example: 0.7 = he plays about 70% of his usual "
                            "share. Assumes targets scale with playing time, which has NOT been validated.")
        vol = s2.slider("Team pass volume vs. usual", 0.7, 1.3, 1.0, 0.05,
                        help="Scales expected targets proportionally to team pass attempts. Hypothetical.")
    try:
        fc = get_forecast(pid, game_id, 1.0, 1.0, svc.meta["built_at"])
        sfc = get_forecast(pid, game_id, pt, vol, svc.meta["built_at"]) if (pt != 1.0 or vol != 1.0) else None
    except ValueError as exc:
        st.warning(str(exc))
        return

    if not fc.game_completed and pd.Timestamp(fc.kickoff_utc) <= now:
        st.warning("This game has kicked off. The forecast uses only pre-kickoff information, but a snapshot saved "
                   "now is recorded as **post-kickoff** (not a genuine pregame forecast).", icon=":material/schedule:")
    if fc.game_completed:
        st.info("This game has been played. The forecast below is what the model would have produced before "
                "kickoff (features from earlier games only; model fitted on earlier seasons). The actual result is "
                "shown for comparison.", icon=":material/history:")
    if fc.evidence_level == "limited":
        st.warning("**Limited evidence.** " + " ".join(fc.flags), icon=":material/info:")

    # --- ask
    st.subheader("Ask a question")
    q_default = st.selectbox("Example questions", router.SUPPORTED, index=None, placeholder="Pick one or type below")
    q = st.text_input("Your question", value=q_default or "", placeholder="e.g. What is the chance he reaches 100 yards?")
    if q:
        ans = router.answer(q, fc, ev, sfc)
        with st.container(border=True):
            st.markdown(ans["markdown"])
            st.caption("Answers are assembled from computed results by a rule-based router (no language model); "
                       "no number is generated by text.")

    # --- headline tiles
    shown = sfc or fc
    if sfc:
        st.warning("**HYPOTHETICAL SCENARIO** — " + sfc.scenario["assumption"], icon=":material/science:")
    st.markdown(N.headline(shown))
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Median yards", shown.yards["median"], help="Half of simulated games are above, half below.")
    m2.metric("Mean yards", f"{shown.yards['mean']:.0f}", help="Average over simulated games; pulled up by big games.")
    m3.metric("80% interval", f"{shown.yards['interval_80'][0]}–{shown.yards['interval_80'][1]}", help=HELP["prediction interval"])
    m4.metric("P(100+ yards)", fmt_pct(shown.milestone_probabilities["100"]),
              help="Model probability, conditional on playing. Workload baseline: "
                   f"{fmt_pct(fc.baseline['workload_milestones']['100'])}.")
    m5.metric("Expected targets", f"{shown.model_params['expected_targets']:.1f}", help=HELP["target"])

    # --- distribution
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Receiving-yard distribution** (share of 20,000 simulated games)")
        h = shown.yards_histogram
        x = np.array(h["bin_start"]) + h["width"] / 2
        fig = go.Figure(go.Bar(x=x, y=h["share"], marker_color=C_MODEL, name="model",
                               hovertemplate="%{x:.0f} yds: %{y:.1%}<extra></extra>"))
        for val in (shown.yards["median"], shown.yards["interval_80"][0], shown.yards["interval_80"][1]):
            fig.add_vline(x=val, line_width=1, line_color=C_MUTED)
        actual_txt = ""
        if fc.game_completed:
            actual = svc.pg[(svc.pg.player_id == pid) & (svc.pg.game_id == game_id)]["receiving_yards"]
            if len(actual):
                fig.add_vline(x=float(actual.iloc[0]), line_width=2, line_color=C_BASE)
                actual_txt = f" Orange line: actual result, {actual.iloc[0]:.0f} yards (not known at forecast time)."
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 330, "share of simulations", "receiving yards"), width="stretch")
        st.caption(f"Gray lines: 10th percentile ({shown.yards['interval_80'][0]}), median ({shown.yards['median']}) "
                   f"and 90th percentile ({shown.yards['interval_80'][1]}).{actual_txt}")
    with right:
        st.markdown("**Milestone probabilities** (conditional on playing)")
        rows = []
        for t, p in shown.milestone_probabilities.items():
            rng = fc.milestone_probability_ranges.get(t, [None, None]) if not sfc else [None, None]
            if rng[0] is None:
                rtxt = "—"
            elif rng[1] < 0.01:
                rtxt = "<1%"
            else:
                rtxt = f"{fmt_pct(rng[0])}–{fmt_pct(rng[1])}"
            rows.append({"Milestone": f"{t}+ yds", "Model": fmt_pct(p), "Range": rtxt,
                         "Workload": fmt_pct(fc.baseline["workload_milestones"][t]),
                         "Last 16": fmt_pct(fc.baseline["empirical_milestones_last16"][t])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Model": st.column_config.TextColumn(help="Structured model, conditional on playing."),
            "Range": st.column_config.TextColumn(help=HELP["probability range"]),
            "Workload": st.column_config.TextColumn(help="Workload baseline: recent targets x yards per target, "
                                                         "with an empirical spread. Not scenario-adjusted."),
            "Last 16": st.column_config.TextColumn(help="Share of his last 16 games reaching the milestone, shrunk "
                                                        "toward the position average."),
        })
        st.caption("*Range*: 10th-90th percentile of the model probability when his own game history is resampled "
                   "(uncertainty about the probability itself, not game-to-game randomness). Held-out testing showed "
                   "**no demonstrated advantage over the workload baseline for 100+ yard milestones**; read both.")

    c_t, c_r = st.columns(2)
    for col, pmf, title, color in ((c_t, shown.target_pmf, "Targets", C_MODEL), (c_r, shown.reception_pmf, "Receptions", C_MODEL)):
        with col:
            st.markdown(f"**Projected {title.lower()}** (share of simulated games)")
            fig = go.Figure(go.Bar(x=list(pmf.keys()), y=list(pmf.values()), marker_color=color,
                                   hovertemplate="%{x}: %{y:.1%}<extra></extra>", name=title))
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 240, "share", title.lower()), width="stretch")
    with st.expander("Table view of the charts above"):
        st.dataframe(pd.DataFrame({"yards_bin_start": shown.yards_histogram["bin_start"],
                                   "share": shown.yards_histogram["share"]}), hide_index=True)
        st.dataframe(pd.DataFrame({"targets": list(shown.target_pmf), "share": list(shown.target_pmf.values())}), hide_index=True)

    # --- explanation
    ex = N.full_explanation(fc, ev)
    st.subheader("Why this forecast, and what could make it wrong")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Main observed factors** (associations in the model, not causes)")
        for line in ex["main_factors"]:
            st.markdown(f"- {line}")
        st.markdown("**Recent role vs. history**")
        for line in ex["role_change"]:
            st.markdown(f"- {line}")
        st.markdown("**Historical accuracy for comparable forecasts**")
        for line in ex["accuracy"]:
            st.markdown(f"- {line}")
    with e2:
        st.markdown("**Factors that increase uncertainty**")
        for line in ex["uncertainty"]:
            st.markdown(f"- {line}")
        st.markdown("**Missing information that could matter**")
        for line in ex["missing"]:
            st.markdown(f"- {line}")
        lab = ex["labels"]
        st.markdown("**Model outputs vs. assumptions**")
        st.markdown("\n".join([f"- *Model:* {x}" for x in lab["model outputs"]] +
                              [f"- *Assumption:* {x}" for x in lab["analyst assumptions"]]))
    with st.container(border=True):
        st.markdown("**Why his average alone cannot give the chance of a 150-yard game**")
        st.markdown(ex["average_vs_tail"])

    with st.expander("Forecast record: identifiers, data cutoff, model version, sample sizes, inputs"):
        meta = {k: getattr(fc, k) for k in ("forecast_created_at", "player_id", "game_id", "kickoff_utc", "data_mode",
                                            "data_cutoff", "data_built_at", "model_version", "model_train_seasons",
                                            "conditional_on_participation", "participation_probability",
                                            "evidence_level", "sample_sizes", "model_params", "inputs", "context")}
        st.json(meta)
    sv1, sv2 = st.columns([1, 3])
    if sv1.button("Save snapshot to journal", type="primary", help="Stores an immutable copy. Later saves become "
                                                                    "revisions; nothing is overwritten."):
        target = sfc or fc
        fid = S.save_forecast(target)
        sv2.success(f"Saved snapshot {fid[:8]}... ({'pregame' if pd.Timestamp(target.forecast_created_at) < pd.Timestamp(target.kickoff_utc) else 'post-kickoff: marked as not pregame'}).")


def page_usage(svc: S.ForecastService) -> None:
    st.header("Usage history")
    pg = svc.pg
    last = pg.sort_values("kickoff_utc").groupby("player_id").tail(1).sort_values("kickoff_utc", ascending=False)
    opts = {r.player_id: f"{r.player_name} ({r.position}, last team {r.team})" for r in last.itertuples()}
    keys = list(opts)
    cur = st.session_state.get("player_id")
    pid = st.selectbox("Player", keys, index=keys.index(cur) if cur in keys else 0, format_func=opts.get)
    st.session_state["player_id"] = pid
    h = pg[pg["player_id"] == pid].sort_values("kickoff_utc").copy()
    f = svc.features[(svc.features["player_id"] == pid) & svc.features["played"]][["game_id", "tpg_ew", "tshare_ew"]]
    h = h.merge(f, on="game_id", how="left")
    seasons = sorted(h["season"].unique())
    if len(seasons) > 1:
        lo, hi = st.select_slider("Seasons", options=seasons, value=(seasons[max(0, len(seasons) - 3)], seasons[-1]))
        h = h[h["season"].between(lo, hi)]
    h["label"] = h["season"].astype(str) + " wk " + h["week"].astype(str) + " " + h["opponent"].fillna("")
    h["yards_roll4"] = h["receiving_yards"].rolling(4, min_periods=1).mean()
    x = h["label"]
    step = max(1, len(h) // 18)
    ticks = dict(tickmode="array", tickvals=list(x.iloc[::step]), tickangle=-45)

    flags = pd.concat([role_change_flags(h, "snap_share", min_diff=0.15),
                       role_change_flags(h, "target_share", min_diff=0.08)], ignore_index=True)
    team_changes = h[h["team"] != h["team"].shift(1)].iloc[1:]

    fig = go.Figure()
    fig.add_bar(x=x, y=h["targets"], name="targets", marker_color=C_MODEL, hovertemplate="%{x}: %{y} targets<extra></extra>")
    fig.add_scatter(x=x, y=h["tpg_ew"], name="pregame expectation (weighted history)", mode="lines",
                    line=dict(color=C_BASE, width=2), hovertemplate="%{x}: %{y:.1f}<extra></extra>")
    st.markdown("**Targets per game** vs. the weighted history the model saw before each game")
    fig.update_xaxes(**ticks)
    st.plotly_chart(style(fig, 280, "targets"), width="stretch")

    fig = go.Figure()
    fig.add_bar(x=x, y=h["receiving_yards"], name="receiving yards", marker_color=C_MODEL,
                customdata=h["receptions"], hovertemplate="%{x}: %{y} yds on %{customdata} rec<extra></extra>")
    fig.add_scatter(x=x, y=h["yards_roll4"], name="4-game rolling mean", mode="lines", line=dict(color=C_BASE, width=2))
    st.markdown("**Receiving yards per game** (hover for receptions)")
    fig.update_xaxes(**ticks)
    st.plotly_chart(style(fig, 280, "yards"), width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=x, y=h["snap_share"], name="offensive snap share", mode="lines+markers",
                    line=dict(color=C_MODEL, width=2), marker=dict(size=8))
    fig.add_scatter(x=x, y=h["target_share"], name="target share", mode="lines+markers",
                    line=dict(color=C_THIRD, width=2), marker=dict(size=8))
    for r in flags.itertuples():
        lab = h.loc[h["game_id"] == r.game_id, "label"]
        if len(lab):
            fig.add_vline(x=lab.iloc[0], line_width=1, line_color=C_MUTED)
    for r in team_changes.itertuples():
        fig.add_vline(x=r.label, line_width=2, line_color=C_BASE, annotation_text=f"joined {r.team}")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(**ticks)
    st.markdown("**Snap share and target share** (gray lines = statistically flagged role changes)")
    st.plotly_chart(style(fig, 300, "share"), width="stretch")
    st.caption("Snap share counts all offensive snaps, including runs. It is **not** route participation, which "
               "is unavailable in-season from the public sources used. " + HELP["target share"])

    imp, metas = user_imports.load_imports()
    if len(imp) and (imp["player_id"] == pid).any():
        u = imp[imp["player_id"] == pid].merge(h[["game_id", "label"]], on="game_id", how="inner")
        if len(u):
            col = "route_participation" if "route_participation" in u and u["route_participation"].notna().any() else "routes_run"
            fig = go.Figure(go.Scatter(x=u["label"], y=u[col], mode="lines+markers", name=col,
                                       line=dict(color=C_THIRD, width=2), marker=dict(size=8)))
            st.markdown(f"**{col.replace('_', ' ')}: USER-SUPPLIED** ({u['provenance'].iloc[0]}) — not verified, not a model input")
            st.plotly_chart(style(fig, 240, col), width="stretch")

    st.markdown("**Role-change flags**")
    if flags.empty:
        st.caption("No statistically clear role changes in the selected seasons.")
    else:
        flags = flags.merge(h[["game_id", "label"]], on="game_id")
        st.dataframe(flags[["label", "metric", "prior_mean", "recent_mean", "diff", "direction"]], hide_index=True,
                     column_config={"prior_mean": st.column_config.NumberColumn("previous 8 games", format="%.2f"),
                                    "recent_mean": st.column_config.NumberColumn("last 3 games", format="%.2f"),
                                    "diff": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("Flag rule: last-3-game mean differs from the previous 8 games by a minimum amount and by more "
                   "than 2 standard errors. A flag is evidence of a change, not proof of a depth-chart move.")
    with st.expander("Table view"):
        st.dataframe(h[["season", "week", "team", "opponent", "targets", "receptions", "receiving_yards",
                        "snap_share", "target_share", "tpg_ew", "stats_zero_filled"]], hide_index=True,
                     column_config={"stats_zero_filled": st.column_config.CheckboxColumn(
                         "zero from snaps", help="Played (snap counts) but no stat row in nflverse: recorded as 0 "
                                                 "targets. Not an imputation.")})


def page_evaluation(ev: dict) -> None:
    st.header("Model evaluation")
    if not ev:
        st.warning("No evaluation results yet. Run `python -m nfl_agent.cli evaluate`.")
        return
    info = ev.get("info", {})
    st.caption(f"Walk-forward evaluation generated {info.get('generated_at')} · data cutoff {info.get('data_cutoff')} · "
               f"model {info.get('model_version')}. Each season is forecast by models fitted only on earlier seasons.")
    sets = {f"Final test {info.get('test_season')} (held out)": "test",
            f"Validation {info.get('validation_seasons', ['?'])[0]}–{info.get('validation_seasons', ['?'])[-1]}": "val"}
    if "live_preds" in ev:
        sets[f"Live {info.get('live_season')} (small sample)"] = "live"
    choice = st.radio("Evaluation set", list(sets), horizontal=True)
    key = sets[choice]
    preds = ev.get(f"{key}_preds")
    if preds is None or preds.empty:
        st.info("No predictions stored for this set.")
        return
    n = int((preds["model"] == "structured").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Player-games scored", f"{n:,}")
    c2.metric("Seasons", ", ".join(str(s) for s in sorted(preds["season"].unique())))
    c3.metric("Games (clusters)", f"{preds['game_id'].nunique():,}")
    sc = W.score(preds, n_boot=200) if key == "live" else {"summary": ev[f"{key}_summary"], "diff_vs_B3": ev[f"{key}_diff"]}
    s = sc["summary"]
    metrics = ["MAE (median)", "CRPS (approx.)", "coverage 50%", "coverage 80%", "coverage 95%", "width 80%",
               "Brier >=50", "LogLoss >=50", "Brier >=100", "LogLoss >=100", "Brier >=150", "LogLoss >=150"]
    t = s[(s["outcome"] == "yards") & s["metric"].isin(metrics)].pivot(index="model", columns="metric", values="value")
    st.markdown("**Receiving yards: models vs. baselines** (lower is better except coverage, which should match its nominal level)")
    st.dataframe(t[[m for m in metrics if m in t]].round(4), width="stretch",
                 column_config={"MAE (median)": st.column_config.NumberColumn(help="Mean absolute error of the median forecast, yards."),
                                "Brier >=100": st.column_config.NumberColumn(help=HELP["Brier score"]),
                                "LogLoss >=100": st.column_config.NumberColumn(help=HELP["log loss"])})
    st.caption("B1B2 = rolling mean + shrunk empirical milestone frequency; B3 = workload baseline; "
               "B3n = workload mean with normal errors (shown to illustrate tail failure).")

    d = sc["diff_vs_B3"]
    if len(d):
        d = d[d["model"] == "structured"].set_index("metric")
        st.markdown("**Structured model minus workload baseline** (negative = structured better; 95% game-cluster "
                    "bootstrap interval)")
        k1, k2 = st.columns([1, 3])
        if "MAE yards" in d.index:
            r = d.loc["MAE yards"]
            k1.metric("MAE difference (yards)", f"{r['estimate']:+.2f}",
                      help=f"95% interval {r['ci_low']:+.2f} to {r['ci_high']:+.2f}. Negative = smaller errors.")
            k1.caption(f"95% interval {r['ci_low']:+.2f} to {r['ci_high']:+.2f}")
        ll = d[d.index.str.startswith("LogLoss")]
        ll = ll.reindex([f"LogLoss >={t}" for t in config.RECEIVING_YARD_MILESTONES]).dropna()
        fig = go.Figure(go.Scatter(x=[i.replace("LogLoss ", "") for i in ll.index], y=ll["estimate"], mode="markers",
                                   marker=dict(size=10, color=C_MODEL),
                                   error_y=dict(type="data", symmetric=False, array=ll["ci_high"] - ll["estimate"],
                                                arrayminus=ll["estimate"] - ll["ci_low"], color=C_MODEL, thickness=1.5),
                                   hovertemplate="%{x}: %{y:+.4f}<extra></extra>", name="structured − B3"))
        fig.add_hline(y=0, line_width=1, line_color=C_MUTED)
        with k2:
            st.plotly_chart(style(fig, 260, "log-loss difference", "milestone (yards)"), width="stretch")
        st.caption("Intervals crossing zero mean no demonstrated difference. Brier-score differences are in the table "
                   "view below.")
        with st.expander("Table view: all differences"):
            st.dataframe(d.reset_index().round(4), hide_index=True)

    st.markdown("**Calibration**")
    thr = st.select_slider("Milestone", options=list(config.RECEIVING_YARD_MILESTONES), value=100)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C_MUTED, width=1), name="perfect calibration",
                    hoverinfo="skip")
    tables = []
    for model, color in (("structured", C_MODEL), ("B3_workload", C_BASE)):
        g = preds[preds["model"] == model]
        if g.empty:
            continue
        c = W.M.calibration_table(g[f"yards_p_ge_{thr}"].to_numpy(float), (g["receiving_yards"].to_numpy() >= thr).astype(float))
        c["model"] = model
        tables.append(c)
        fig.add_scatter(x=c["mean_pred"], y=c["observed"], mode="lines+markers", name=model,
                        line=dict(color=color, width=2), marker=dict(size=9),
                        error_y=dict(type="data", symmetric=False, array=c["obs_ci_high"] - c["observed"],
                                     arrayminus=c["observed"] - c["obs_ci_low"], color=color, thickness=1),
                        customdata=c["n"], hovertemplate="forecast %{x:.1%} → observed %{y:.1%} (n=%{customdata})<extra></extra>")
    mx = max(0.05, float(pd.concat(tables)["mean_pred"].max()) * 1.15) if tables else 1
    fig.update_xaxes(range=[0, min(1, mx)], tickformat=".0%")
    fig.update_yaxes(range=[0, min(1, mx * 1.3)], tickformat=".0%")
    st.plotly_chart(style(fig, 380, "observed frequency", f"forecast probability of {thr}+ yards"), width="stretch")
    with st.expander("Calibration table (with sample counts)"):
        st.dataframe(pd.concat(tables)[["model", "bin_low", "bin_high", "n", "mean_pred", "observed", "obs_ci_low", "obs_ci_high"]].round(4), hide_index=True)

    st.markdown("**Prediction-interval coverage** (randomized PIT; target = nominal level)")
    cov = s[(s["outcome"] == "yards") & s["metric"].str.match(r"^coverage \d")].copy()
    cov["nominal"] = cov["metric"].str.extract(r"(\d+)").astype(float) / 100
    fig = go.Figure()
    for model, color in (("structured", C_MODEL), ("B3_workload", C_BASE), ("B3n_workload_normal", C_THIRD)):
        g = cov[cov["model"] == model]
        fig.add_bar(x=g["metric"], y=g["value"], name=model, marker_color=color,
                    hovertemplate="%{x}: %{y:.1%}<extra>" + model + "</extra>")
    fig.update_layout(bargap=0.35, bargroupgap=0.06)
    for lvl in (0.5, 0.8, 0.95):
        fig.add_hline(y=lvl, line_width=1, line_color=C_MUTED)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(style(fig, 300, "share of outcomes inside interval"), width="stretch")
    w = s[(s["outcome"] == "yards") & s["metric"].str.startswith("width")].pivot(index="model", columns="metric", values="value")
    st.caption("Mean interval width (yards): " + "; ".join(f"{m}: " + ", ".join(f"{c.split()[1]} {v:.0f}" for c, v in r.items())
                                                       for m, r in w.iterrows()))

    st.markdown("**High milestones** (rare: treat with caution)")
    from nfl_agent.evaluation.report import high_milestone_table
    st.dataframe(high_milestone_table(preds, [m for m in ("structured", "B3_workload", "B1B2_rolling_empirical", "B3n_workload_normal")
                                              if m in set(preds["model"])]).round(4), hide_index=True)
    st.caption("A model is not reliable because of a handful of correct calls. With ~20-40 150-yard games per "
               "season, small calibration errors are within noise.")
    if key == "val" and "ablation_summary" in ev:
        st.markdown("**Ablation: does adding a feature group improve held-out performance?** (validation seasons)")
        a = ev["ablation_summary"]
        am = ["MAE (median)", "CRPS (approx.)", "LogLoss >=50", "LogLoss >=100", "LogLoss >=150", "coverage 80%"]
        st.dataframe(a[(a["outcome"] == "yards") & a["metric"].isin(am)].pivot(index="model", columns="metric", values="value")[am].round(4), width="stretch")
        st.caption("Team volume, opponent and context groups did not improve held-out scores; snap share did. "
                   "The default model uses player history + snaps.")


def page_health(svc: S.ForecastService) -> None:
    st.header("Data health")
    meta = svc.meta
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data mode", meta["data_mode"].upper())
    c2.metric("Built (UTC)", pd.Timestamp(meta["built_at"]).strftime("%b %d %H:%M"), help=meta["built_at"])
    cut = meta.get("data_cutoff")
    c3.metric("Latest completed game", pd.Timestamp(cut).strftime("%b %d %H:%M") if cut else "none",
              help=f"Kickoff (UTC) of the latest completed game in the schedule: {cut}")
    c4.metric("WR/TE player-games", f"{meta['n_player_games']:,}")
    if meta.get("zero_filled_share") is not None:
        st.caption(f"{meta['zero_filled_share']:.1%} of player-games had snaps but no stat row and were recorded as "
                   "0 targets from snap counts (they would otherwise be silently missing).")
    if st.button("Download latest data (cached; ~1 minute)"):
        with st.spinner("Downloading and rebuilding..."):
            m = pipeline.run(mode="auto")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success(f"Rebuilt ({m['data_mode']}). Failures: {m['fetch_failures'] or 'none'}")
    try:
        inv, miss = inventory.build_inventory()
        st.markdown("**Sources** (freshness, coverage, pregame availability)")
        st.dataframe(inv, hide_index=True, width="stretch", column_config={
            "pregame_available": st.column_config.TextColumn("usable before a historical kickoff?"),
            "limitations": st.column_config.TextColumn(width="large")})
        st.markdown("**Missing fields**")
        st.dataframe(miss, hide_index=True, column_config={"missing_share": st.column_config.ProgressColumn(
            "missing share", min_value=0.0, max_value=1.0, format="%.3f")})
        log = inventory.ingestion_log(200)
        st.markdown("**Ingestion log** (latest 200; failures first)")
        if len(log):
            log = log.sort_values(["status", "id"], ascending=[True, False])
            st.dataframe(log[["retrieved_at", "source_key", "season", "status", "n_rows", "source_last_updated", "message"]],
                         hide_index=True, width="stretch")
        else:
            st.caption("No downloads logged (demo mode or offline build).")
    except Exception as exc:  # inventory is informational; never break the page
        st.warning(f"Inventory unavailable: {type(exc).__name__}")
    issues = {k: v for k, v in meta.get("validation_issues", {}).items() if v}
    st.markdown("**Validation issues**")
    st.json(issues or {"none": "all checks passed"})
    st.markdown("**Coverage limitations**")
    st.markdown("""
- Route participation / routes run: **not available in-season** (FTN participation is released after the postseason).
- Injury reports: no publication timestamp → context only, not a model input.
- Weather: schedule temp/wind are observed after the game → not used; no archived pregame forecasts.
- Starting QB in the schedule: retrospective for completed games → context only.
- Next Gen Stats weekly rows exist only for qualifying receivers (not missing at random) → context only.
- Snap counts: the 2012 file is published but empty, so snap data start in 2013; 2012 serves only as feature warm-up.
""")
    st.markdown("**Import user-supplied data (e.g. routes run)**")
    st.caption("Imported values are labelled as user-supplied, validated against the schema, and NOT used as model "
               "inputs. Blank cells stay missing. Schema: player_id, game_id, routes_run and/or route_participation, "
               "optional team_dropbacks, notes.")
    up = st.file_uploader("CSV file", type=["csv"])
    label = st.text_input("Source label (required: where did these numbers come from?)")
    if up is not None and st.button("Validate and import"):
        raw = up.getvalue()
        df, errs = user_imports.validate_routes_csv(raw, set(svc.pg["player_id"]), set(svc.games["game_id"]))
        if errs:
            st.error("Import rejected:\n\n" + "\n".join(f"- {e}" for e in errs))
        elif not label.strip():
            st.error("Please provide a source label.")
        else:
            m = user_imports.save_import(df, raw, up.name, label)
            st.success(f"Imported {m['rows']} rows (sha256 {m['sha256'][:12]}...).")
    _, metas = user_imports.load_imports()
    if metas:
        st.dataframe(pd.DataFrame(metas), hide_index=True)


def page_journal(svc: S.ForecastService) -> None:
    st.header("Forecast journal")
    st.caption("Immutable forecast snapshots. The database rejects edits and deletions; refreshed forecasts are "
               "stored as new revisions. Actual results are attached in a separate table after games.")
    if st.button("Attach results for completed games"):
        n = S.record_results(svc)
        st.success(f"Recorded results for {n} player-games.")
    j = S.journal()
    if j.empty:
        st.info("No snapshots yet. Use 'Save snapshot to journal' on the Player forecast page.")
        return
    j["integrity_ok"] = [hashlib.sha256(p.encode()).hexdigest() == h for p, h in zip(j["payload_json"], j["payload_sha256"])]
    pay = [json.loads(p) for p in j["payload_json"]]
    j["median_yds"] = [p["yards"]["median"] for p in pay]
    j["p100"] = [N.pct(p["milestone_probabilities"]["100"]) for p in pay]
    j["interval_80"] = [f"{p['yards']['interval_80'][0]}–{p['yards']['interval_80'][1]}" for p in pay]
    j["pregame"] = j["is_pregame"].astype(bool)
    j["actual"] = np.where(j["played"].isna(), "pending", np.where(j["played"] == 0, "did not play",
                                                                    j["actual_yards"].astype(str)))
    st.dataframe(j[["created_at", "player_name", "game_id", "pregame", "revision", "scenario", "model_version", "data_cutoff",
                    "data_mode", "median_yds", "interval_80", "p100", "actual", "integrity_ok"]], hide_index=True,
                 width="stretch", column_config={
                     "pregame": st.column_config.CheckboxColumn(help="Created before kickoff. Only pregame snapshots "
                                                                     "count as genuine forecasts."),
                     "integrity_ok": st.column_config.CheckboxColumn("hash ok", help="Stored payload still matches its "
                                                                                      "SHA-256 fingerprint.")})
    sel = st.selectbox("Inspect snapshot", j["forecast_id"], format_func=lambda f: f"{f[:8]} · " +
                       j.loc[j.forecast_id == f, "player_name"].iloc[0] + " · " + j.loc[j.forecast_id == f, "game_id"].iloc[0])
    st.json(json.loads(j.loc[j.forecast_id == sel, "payload_json"].iloc[0]), expanded=False)


def page_about() -> None:
    st.header("About, methods and glossary")
    st.markdown("""
This is an **educational forecasting lab**. It estimates distributions of receiving production for wide receivers
and tight ends and measures how accurate those estimates have been. It has no sportsbook data, makes no wager
recommendations and places no bets.

**How a forecast is made**
1. *Target volume*: a gradient-boosted Poisson model of expected targets from the player's recent targets,
   target share, snap share, experience, draft position and team tenure; game-to-game spread follows a
   negative binomial whose dispersion depends on the expected volume.
2. *Catches given targets*: a logistic model of catch probability, with a small game-level catch-rate shock.
3. *Yards given catches*: a gamma model of yards per catch; each catch's yardage is drawn separately.
4. *Simulation*: 20,000 games are simulated; every number on the forecast page is read from those simulations.

All model inputs are built from games played **before** kickoff. Rookies and players with few games are pulled
toward position averages (shrinkage), and the model receives their sample size and draft position. A player's first
game with a new team is flagged; his history from the old team is still used but with lower confidence.
See `docs/METHODOLOGY.md` and `docs/EVALUATION.md` for details and results.
""")
    st.markdown("**Glossary**")
    st.dataframe(pd.DataFrame({"term": list(HELP), "definition": list(HELP.values())}), hide_index=True, width="stretch")


# ---------------------------------------------------------------- main
def main() -> None:
    try:
        svc = get_service()
    except FileNotFoundError:
        st.title("NFL Receiving Forecast Lab")
        st.error("No processed data found. In a terminal run `python -m nfl_agent.cli build` (real data) or "
                 "`python -m nfl_agent.cli build --demo` (synthetic demo data), then reload.")
        return
    ev = load_eval(svc.meta["built_at"], svc.data_mode)
    pages = {"Player forecast": lambda: page_forecast(svc, ev), "Usage history": lambda: page_usage(svc),
             "Model evaluation": lambda: page_evaluation(ev), "Data health": lambda: page_health(svc),
             "Forecast journal": lambda: page_journal(svc), "About & glossary": page_about}
    with st.sidebar:
        st.title("Receiving Forecast Lab")
        choice = st.radio("View", list(pages), label_visibility="collapsed")
        st.caption(f"Data: **{svc.data_mode}** · cutoff {str(svc.meta.get('data_cutoff'))[:16]} UTC · model "
                   f"{config.MODEL_VERSION}")
        st.caption("Educational, no-money forecasting. Not betting advice.")
    demo_banner(svc.meta)
    pages[choice]()


main()
