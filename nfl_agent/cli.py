"""Command-line entry point.

    python -m nfl_agent.cli build [--demo] [--refresh] [--offline]
    python -m nfl_agent.cli evaluate
    python -m nfl_agent.cli inventory
    python -m nfl_agent.cli forecast --player "Name" --game GAME_ID
    python -m nfl_agent.cli record-results
"""
from __future__ import annotations

import argparse
import json
import logging


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nfl_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download (cached) data and build modelling tables")
    b.add_argument("--demo", action="store_true", help="build the synthetic DEMO fixture instead")
    b.add_argument("--refresh", action="store_true", help="re-download even completed seasons")
    b.add_argument("--offline", action="store_true", help="use only the local cache")
    e = sub.add_parser("evaluate", help="walk-forward evaluation + final test; writes docs/EVALUATION.md")
    e.add_argument("--demo", action="store_true")
    e.add_argument("--quick", action="store_true", help="fewer seasons, for smoke testing")
    e.add_argument("--render-only", action="store_true", help="re-render the report from stored results")
    sub.add_parser("inventory", help="write the data inventory (docs/DATA_INVENTORY.md)")
    f = sub.add_parser("forecast", help="print a forecast and store a journal snapshot")
    f.add_argument("--player", required=True, help="player name or GSIS id")
    f.add_argument("--game", required=True, help="nflverse game_id, e.g. 2026_05_CIN_MIA")
    f.add_argument("--no-save", action="store_true")
    sub.add_parser("record-results", help="attach actual results to journal forecasts for completed games")
    sl = sub.add_parser("slate", help="export upcoming-game probability curves for the Receiving Line Checker page")
    sl.add_argument("--days", type=float, default=9.0, help="include games kicking off within this many days")
    sl.add_argument("--bootstrap", type=int, default=20, help="history resamples per player for probability ranges")
    sl.add_argument("--out", default=None, help="output JSON path (default: <data>/processed/linechecker/slate.json)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "build":
        from .data import pipeline
        mode = "demo" if args.demo else "auto"
        meta = pipeline.run(mode=mode, download=not (args.demo or args.offline), refresh=args.refresh)
        print(json.dumps({k: v for k, v in meta.items() if k != "validation_issues"}, indent=2))
        for k, v in meta["validation_issues"].items():
            for issue in v:
                print("validation:", issue)
    elif args.cmd == "evaluate":
        from .evaluation import report
        if args.render_only:
            report.render_from_saved(demo=args.demo)
        else:
            report.run(demo=args.demo, quick=args.quick)
        if args.demo:
            print("demo evaluation tables written (docs/EVALUATION.md is only written for real data)")
        else:
            print(f"wrote {report.config.DOCS_DIR / 'EVALUATION.md'}")
    elif args.cmd == "inventory":
        from .data import inventory
        path = inventory.write_inventory()
        print(f"wrote {path}")
    elif args.cmd == "forecast":
        from .forecast import service
        svc = service.ForecastService.load()
        pid = svc.resolve_player(args.player)
        fc = svc.forecast(pid, args.game)
        print(json.dumps(fc.to_display_dict(), indent=2, default=str))
        if not args.no_save:
            fid = svc.save(fc)
            print(f"journal snapshot: {fid}")
    elif args.cmd == "slate":
        from pathlib import Path

        from . import config
        from .forecast import service, slate
        svc = service.ForecastService.load()
        data = slate.build_slate(svc, days=args.days, bootstrap=args.bootstrap)
        out = Path(args.out) if args.out else config.PROCESSED_DIR / "linechecker" / "slate.json"
        slate.write_slate(data, out)
        print(f"wrote {out}: {len(data['players'])} players in {len(data['games'])} games "
              f"(data through {data['data_through']})")
    elif args.cmd == "record-results":
        from .forecast import service
        svc = service.ForecastService.load()
        n = svc.record_results()
        print(f"recorded results for {n} journal forecasts")


if __name__ == "__main__":
    main()
