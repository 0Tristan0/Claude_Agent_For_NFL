# NFL Receiving Forecast Lab

A local research tool for **educational, no-money forecasting** of NFL wide receiver and tight end production.
Pick a player and a scheduled game, ask plain-English questions, and get a full probability distribution for
receiving yards and receptions. Each forecast comes with milestone probabilities (25, 50, 75, 100, 125, 150 yards),
uncertainty, an evidence-based explanation, and an honest record of how accurate comparable forecasts have been.

The research dashboard uses no betting lines. A separate page, the **Receiving Line Checker** (below), scores
pick'em lines that *you* supply from a screenshot. Nothing connects to PrizePicks, Underdog or Sleeper (no
scraping, no account access, no bet placement). No API key is required; every probability comes from fitted
models and simulation, never from a language model.

## Quick start (exact commands)

Tested with Python 3.11 on Linux.

```bash
# 1. environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. data: downloads ~25 MB of public nflverse release files (cached under data/cache), ~1 minute
python -m nfl_agent.cli build

# 3. evaluation: walk-forward backtest 2019-2025 + live season, ~6-10 minutes.
#    Needed for the Model evaluation page and the "historical accuracy" statements.
python -m nfl_agent.cli evaluate

# 4. dashboard
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. The first forecast for a season takes about 20 seconds while that
season's model is fitted and cached; later forecasts take a few seconds.

**No internet / blocked downloads?** `python -m nfl_agent.cli build --demo` builds a synthetic fixture (teams
`DMA`–`DMH`, players `DEMO …`). The dashboard shows a red **DEMO MODE** banner on every page, and demo data are
never mixed with real data. `python -m nfl_agent.cli evaluate --demo` fills the evaluation page for the demo.

Other commands:

```bash
python -m nfl_agent.cli build --offline      # rebuild tables from the local cache only
python -m nfl_agent.cli build --refresh      # re-download completed seasons too (stat corrections)
python -m nfl_agent.cli inventory            # regenerate docs/DATA_INVENTORY.md
python -m nfl_agent.cli forecast --player "Ja'Marr Chase" --game 2026_05_CIN_MIA   # prints + journals
python -m nfl_agent.cli record-results       # attach actual results to journal forecasts
python -m nfl_agent.cli slate                # export curves for the Receiving Line Checker page
python -m pytest                             # 60 tests, ~40 s, no network needed
```

## Receiving Line Checker (screenshot → scored picks)

Live page: <https://claude.ai/artifact/AdVscj6DYUebaAxU9T3Q9W> (private to the owner until shared from its Share menu).

1. Upload, drop or paste a screenshot from PrizePicks, Underdog or Sleeper, then press **Read lines**. Claude reads
   the player names, stats and lines from the image on your own Claude account; it does not produce any
   probabilities. You can also type lines by hand.
2. Each WR/TE **receiving yards** or **receptions** line gets the model's chance of More and Less, a range
   (how much that chance moves when the player's history is resampled), the workload baseline's chance, and an
   evidence level. Other stats (rushing, passing, fantasy score, combos, first-half props) are listed as not covered.
3. **Best entry** combines the most likely picks (2–6, at most one per player, optionally one per game) into the
   chance that every pick hits. Enter your app's payout to compare that chance with break-even; when the edge is
   smaller than the model's typical error the page says "too close to call".
4. **Track record** shows how often forecasts in each probability range came true in the held-out 2025 season.

Limits: the model was never tested against pick'em lines; forecasts assume the player plays; combined chances
treat picks as independent; QB changes, injuries and weather are not inputs.

**Refreshing the data** (the page carries a snapshot; it flags forecasts that miss a team's earlier game):

```bash
python -m nfl_agent.cli build     # pull the latest nflverse stats (nightly after game days)
python -m nfl_agent.cli slate     # ~9 min: curves for every WR/TE kicking off in the next 9 days
```

Then republish `app/linechecker/index.html` with `data/processed/linechecker/slate.json` as `slate.json`
(ask Claude Code to republish the Receiving Line Checker).

## What the dashboard does

| View | Contents |
|---|---|
| **Player forecast** | Season / week / game / player selection; question box; median, mean, 50/80/95% intervals; yardage distribution; milestone probabilities with a *probability range* and two baselines; target and reception distributions; explanation (main factors, uncertainty, role change, missing inputs, model output vs. assumption); hypothetical playing-time and pass-volume scenarios; "save snapshot" to the journal |
| **Usage history** | Targets vs. the pregame expectation the model saw, yards, snap share and target share over time; statistically flagged role changes; team changes; user-supplied route data (labelled) |
| **Model evaluation** | Final test season, validation seasons and live season; baselines vs. model; differences with game-clustered bootstrap intervals; calibration plots with counts; interval coverage and width; high-milestone table; ablation |
| **Data health** | Source freshness and coverage, missing fields, ingestion log and failures, validation issues, coverage limitations, CSV import |
| **Forecast journal** | Immutable pregame snapshots, revisions, model versions, data cutoffs, actual results, integrity hashes |

Questions the question box answers (deterministic keyword routing to computed results):
the projected receiving-yard distribution; how uncertain target volume is; milestone probabilities;
how reduced playing time changes the forecast; the evidence for the forecast and what could make it wrong;
historical accuracy for comparable forecasts; and why average yardage cannot determine the chance of a
150-yard game.

## Results in one paragraph

On the untouched **2025 test season** (4,649 player-games), the structured model's median forecast missed by
**15.2 yards** on average, against 15.9 for the workload baseline: −0.68 yards, 95% game-cluster interval
[−0.84, −0.51]. It also had better log loss for 25+ and 50+ yards. Its 50/80/95% intervals covered **50.7 / 80.9 /
95.2%** of outcomes. **For 100+ yard milestones it showed no demonstrated advantage over the simple baseline**, and
both over-predicted 150-yard games in 2025 (about 30 expected vs. 20 observed), so the dashboard always shows the
baseline beside the model. A normal-distribution model under-predicted 150-yard games roughly three-fold. Team
volume, opponent defense and game-context features did not improve held-out accuracy and are not in the default
model. Full details: [docs/EVALUATION.md](docs/EVALUATION.md).

## Project layout

```
nfl_agent/
  config.py              paths, seasons, model version, milestones
  db.py                  SQLite: ingestion log, immutable forecast journal (UPDATE/DELETE blocked by triggers)
  data/                  sources.py (registry + pregame classification), ingest.py (cached downloads),
                         validate.py, build.py (participation spine & joins), pipeline.py, inventory.py,
                         imports.py (user CSV), demo.py (synthetic fixture)
  features/              pregame.py (shifted, shrunk, exponentially-weighted features), roles.py
  models/                baselines.py, structured.py (targets -> catches -> yards), simulate.py
  evaluation/            walkforward.py, metrics.py, report.py
  explain/narrative.py   plain-English explanations from computed numbers
  agent/router.py        question -> computed answer
  forecast/service.py    forecast assembly, probability ranges, journal save / results
  forecast/slate.py      probability curves for upcoming games (Receiving Line Checker data)
  cli.py
app/streamlit_app.py     dashboard (UI only)
app/linechecker/         Receiving Line Checker page (published as a claude.ai artifact)
tests/                   pytest suite (joins, leakage, simulation, journal, imports, metrics, router)
docs/                    METHODOLOGY.md, DATA_DICTIONARY.md, DATA_INVENTORY.md, EVALUATION.md
data/                    local cache, processed tables, SQLite (git-ignored)
```

## Key design rules

* **No leakage.** Features use only earlier games (verified by a test that perturbs the current game's outcome).
  Priors use earlier seasons only. Each evaluated season's models are fitted on earlier seasons, including
  imputation and dispersion estimates.
* **Missing stays missing.** Zero-target games are recovered from snap counts (a recorded fact). Inactive games
  are excluded, never zero-filled. Snap share is never relabelled as route participation.
* **Conditional on playing.** Participation is not modelled; every forecast says so.
* **Provenance.** Each source is classified by whether it was available before a historical kickoff. Weather,
  starting QB, injury reports and route data are not model inputs, for the reasons listed in
  [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).
* **Immutable journal.** Snapshots record creation time, data cutoff, model version and a SHA-256 of the payload;
  a refreshed forecast is a new revision; results live in a separate table.

## Data and license

Data: [nflverse](https://github.com/nflverse/nflverse-data) release files (CC-BY 4.0; underlying statistics from the
NFL, snap counts from Pro-Football-Reference). Please credit nflverse when sharing outputs. The app downloads only
public release assets, caches them, and pauses between requests.

## Known limitations

See the end of [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md). In short:
no participation model; no in-season route data; QB and teammate availability are not inputs; the extreme right
tail (150+) is slightly over-predicted; model-parameter uncertainty is not included in the probability range;
training data reflect later stat corrections.
