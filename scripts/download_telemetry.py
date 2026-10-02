"""Download one driver's laps of a Formula 1 session with FastF1, one CSV per lap.

    python scripts/download_telemetry.py --year 2025 --event "Italian Grand Prix" --driver VER \\
        --laps 1 4-12 --out telemetry/monza
"""
import argparse
from pathlib import Path

import fastf1


def parse_laps(items: list[str]) -> list[int]:
    laps = set()
    for item in items:
        first, _, last = item.partition("-")
        first, last = int(first), int(last or first)
        if last < first:
            raise ValueError(item)
        laps.update(range(first, last + 1))
    return sorted(laps)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--event", required=True, help='event name or round number, e.g. "Italian Grand Prix" or 16')
    p.add_argument("--session", default="R", help="session identifier (default: R = race)")
    p.add_argument("--driver", required=True, help="three-letter driver code, e.g. VER")
    p.add_argument("--laps", nargs="+", required=True, help="lap numbers and ranges, e.g. 1 4-12")
    p.add_argument("--out", required=True, help="output folder for lap<NN>.csv")
    p.add_argument("--cache", default="cache", help="FastF1 cache folder (default: cache)")
    args = p.parse_args()

    try:
        laps_wanted = parse_laps(args.laps)
    except ValueError:
        raise SystemExit(f"cannot read --laps {' '.join(args.laps)}; use numbers and ranges like 1 4-12")
    event = int(args.event) if args.event.isdigit() else args.event

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)
    session = fastf1.get_session(args.year, event, args.session)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    laps = session.laps.pick_drivers(args.driver)
    if laps.empty:
        drivers = ", ".join(sorted(session.laps["Driver"].unique()))
        raise SystemExit(f"no laps for driver {args.driver}; drivers in this session: {drivers}")
    available = sorted(int(n) for n in laps["LapNumber"].dropna())
    for number in laps_wanted:
        lap = laps[laps["LapNumber"] == number]
        if lap.empty:
            raise SystemExit(f"{args.driver} has no lap {number}; available laps: {available[0]}-{available[-1]}")
        telemetry = lap.iloc[0].get_telemetry()
        telemetry.to_csv(out / f"lap{number:02d}.csv", index=False)
        print(f"lap {number:2d}: {len(telemetry)} samples")
    print(f"wrote {len(laps_wanted)} laps to {out}")


if __name__ == "__main__":
    main()
