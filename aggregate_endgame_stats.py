#!/usr/bin/env python3
"""
aggregate_endgame_stats.py

Aggregate endgame_stats_*.tsv (produced by endgame_stats.py) across months, grouped by Elo bucket.

- Bucket is identified by (elo_min, elo_max) from the TSV header lines.
- Overall counters are summed exactly from headers.
- Per-material aggregation now reads explicit integer columns from the new TSV format.
"""

from __future__ import annotations

import argparse
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


META_RE = re.compile(r"^#\s*([^=]+)=(.*)$")
DEFAULT_GLOB = "endgame_stats_*.tsv"


@dataclass
class FileStats:
    path: Path
    month: str
    elo_min: int
    elo_max: int
    meta: Dict[str, str]
    rows: List[List[str]]  # raw tab-split rows
    header: List[str]


@dataclass
class MaterialAgg:
    games: int = 0
    plies: int = 0
    games_with_error: int = 0
    errors_total: int = 0
    time_losses: int = 0


@dataclass
class BucketAgg:
    elo_min: int
    elo_max: int
    months: List[str] = field(default_factory=list)
    files: List[Path] = field(default_factory=list)

    # Exact header-level sums.
    raw_seen: int = 0
    games_seen: int = 0
    games_used: int = 0
    games_skipped_short: int = 0
    games_skipped_parse: int = 0
    games_with_any_phase: int = 0
    games_ended_in_3to5: int = 0
    plies_total: int = 0
    errors_total: int = 0
    time_loss_games_total: int = 0

    # Per-material aggregation.
    per_material: Dict[str, MaterialAgg] = field(default_factory=dict)

    def add_file(self, fs: FileStats) -> None:
        self.months.append(fs.month)
        self.files.append(fs.path)

        def geti(name: str, default: int = 0) -> int:
            v = fs.meta.get(name)
            if v is None:
                return default
            try:
                return int(v)
            except ValueError:
                return default

        self.raw_seen += geti("raw_seen")
        self.games_seen += geti("games_seen")
        self.games_used += geti("games_used")
        self.games_skipped_short += geti("games_skipped_short_plycount<35")
        self.games_skipped_parse += geti("games_skipped_parse")
        self.games_with_any_phase += geti("games_with_any_phase")
        self.games_ended_in_3to5 += geti("games_ended_in_3to5")
        self.plies_total += geti("plies_total")
        self.errors_total += geti("errors_total")
        self.time_loss_games_total += geti("time_loss_games_total")

        # Per-material rows.
        col = {name: i for i, name in enumerate(fs.header)}

        def f(name: str, row: List[str]) -> str:
            idx = col.get(name)
            return row[idx] if idx is not None and idx < len(row) else ""

        def val(name: str) -> int:
            try:
                return int(f(name, row))
            except ValueError:
                return 0

        for row in fs.rows:
            mat = f("material", row)
            if not mat:
                continue

            # In new format, we have explicit integers.
            g = val("games")
            plies = val("plies")
            gerr = val("games_with_error")
            errs = val("errors")
            tl = val("time_losses")

            agg = self.per_material.get(mat)
            if agg is None:
                agg = MaterialAgg()
                self.per_material[mat] = agg

            agg.games += g
            agg.plies += plies
            agg.games_with_error += gerr
            agg.errors_total += errs
            agg.time_losses += tl

    def finalize(self) -> None:
        self.months = sorted(set(self.months))
        self.files = sorted(set(self.files), key=str)

    def errors_per_ply_pct_total(self) -> float:
        return ((self.errors_total / self.plies_total) * 100.0) if self.plies_total else 0.0

    def pct_any_phase_over_games_used(self) -> float:
        return ((self.games_with_any_phase / self.games_used) * 100.0) if self.games_used else 0.0

    def pct_ended_in_3to5_over_games_used(self) -> float:
        return ((self.games_ended_in_3to5 / self.games_used) * 100.0) if self.games_used else 0.0

    def pct_time_loss_over_games_used(self) -> float:
        return ((self.time_loss_games_total / self.games_used) * 100.0) if self.games_used else 0.0


def parse_tsv(path: Path) -> FileStats:
    meta: Dict[str, str] = {}
    header: Optional[List[str]] = None
    rows: List[List[str]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith("#"):
            m = META_RE.match(line)
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
            continue
        if line.startswith("material\t"):
            header = line.split("\t")
            continue
        if header is not None:
            rows.append(line.split("\t"))

    if header is None:
        raise ValueError(f"Missing TSV header row (material\t...) in {path}")

    month = meta.get("month", "")
    try:
        elo_min = int(meta.get("elo_min", "0"))
        elo_max = int(meta.get("elo_max", "0"))
    except ValueError as e:
        raise ValueError(f"Invalid elo_min/elo_max in {path}: {e}") from e

    if not month or elo_min <= 0 or elo_max <= 0:
        raise ValueError(f"Missing required meta (month/elo_min/elo_max) in {path}")

    return FileStats(
        path=path,
        month=month,
        elo_min=elo_min,
        elo_max=elo_max,
        meta=meta,
        rows=rows,
        header=header,
    )


def iter_input_paths(args: argparse.Namespace) -> List[Path]:
    if args.paths:
        paths = [Path(p) for p in args.paths]
        # Robustness: If user ran `script.py --glob file1 file2`, argparse assigns glob=file1, paths=[file2].
        # We detect this if glob is not default and paths is non-empty. We add glob back to paths.
        if args.glob != DEFAULT_GLOB:
            paths.insert(0, Path(args.glob))
    else:
        paths = sorted(Path().glob(args.glob))

    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob(DEFAULT_GLOB)))
        else:
            out.append(p)

    return [p for p in out if p.exists() and p.is_file()]


def print_summary(buckets: List[BucketAgg]) -> None:
    cols = [
        ("bucket", 16),
        ("months", 7),
        ("games_used", 12),
        ("%any", 8),
        ("err/ply%", 10),
        ("%time_loss", 10),
    ]
    fmt = " ".join([f"{{:{w}}}" for _, w in cols])

    print(fmt.format(*[c[0] for c in cols]))
    print(fmt.format(*["-" * c[1] for c in cols]))

    for b in buckets:
        bucket_s = f"{b.elo_min}-{b.elo_max}"
        months_s = str(len(b.months))
        games_used_s = str(b.games_used)
        pct_any_s = f"{b.pct_any_phase_over_games_used():.2f}"
        err_s = f"{b.errors_per_ply_pct_total():.4f}"
        tl_s = f"{b.pct_time_loss_over_games_used():.4f}"
        print(fmt.format(bucket_s, months_s, games_used_s, pct_any_s, err_s, tl_s))


def print_bucket_full(b: BucketAgg, top: int, min_games: int) -> None:
    print()
    print(f"### AGG bucket elo{b.elo_min}-{b.elo_max}")
    print("# month=AGG")
    print(f"# months={','.join(b.months)}")
    print(f"# files={len(b.files)}")
    print(f"# elo_min={b.elo_min}")
    print(f"# elo_max={b.elo_max}")
    print("# elo_rule=soft")
    print(f"# raw_seen={b.raw_seen}")
    print(f"# games_seen={b.games_seen}")
    print(f"# games_used={b.games_used}")
    print(f"# games_skipped_short_plycount<35={b.games_skipped_short}")
    print(f"# games_skipped_parse={b.games_skipped_parse}")
    print(f"# games_with_any_phase={b.games_with_any_phase}")
    print(f"# pct_any_phase_over_games_used={b.pct_any_phase_over_games_used():.6f}")
    print(f"# games_ended_in_3to5={b.games_ended_in_3to5}")
    print(f"# pct_ended_in_3to5_over_games_used={b.pct_ended_in_3to5_over_games_used():.6f}")
    print(f"# plies_total={b.plies_total}")
    print(f"# errors_total={b.errors_total}")
    print(f"# errors_per_ply_pct_total={b.errors_per_ply_pct_total():.8f}")
    print(f"# time_loss_games_total={b.time_loss_games_total}")
    print(f"# pct_time_loss_over_games_used={b.pct_time_loss_over_games_used():.6f}")

    print(
        "material\t"
        "games\tgames_pct_over_used\t"
        "plies\tavg_plies_per_game\t"
        "games_with_error\terror_game_pct\t"
        "errors\terrors_per_ply_pct\t"
        "time_losses\ttime_loss_pct"
    )

    denom_used = b.games_used if b.games_used > 0 else 1

    items = list(b.per_material.items())
    items.sort(key=lambda kv: (-kv[1].games, kv[0]))

    shown = 0
    for mat, a in items:
        if a.games < min_games:
            continue
        if top and shown >= top:
            break

        games = a.games
        pct_used = (games / denom_used) * 100.0
        avg_plies = (a.plies / games) if games else 0.0
        err_game_pct = (a.games_with_error / games * 100.0) if games else 0.0
        err_per_ply_pct = (a.errors_total / a.plies * 100.0) if a.plies else 0.0
        tl_pct = (a.time_losses / games * 100.0) if games else 0.0

        print(
            f"{mat}\t"
            f"{games}\t{pct_used:.6f}\t"
            f"{a.plies}\t{avg_plies:.6f}\t"
            f"{a.games_with_error}\t{err_game_pct:.6f}\t"
            f"{a.errors_total}\t{err_per_ply_pct:.6f}\t"
            f"{a.time_losses}\t{tl_pct:.6f}"
        )
        shown += 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate endgame_stats_*.tsv by Elo bucket (elo_min/elo_max)."
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="TSV files and/or directories. If omitted, uses --glob.",
    )
    ap.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="Glob used when no positional paths are provided.",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Also print an aggregated per-material TSV block per bucket.",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=0,
        help="When --full, show only the top N materials by games (0=all).",
    )
    ap.add_argument(
        "--min-games",
        type=int,
        default=0,
        help="When --full, hide materials with fewer than this many games.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    paths = iter_input_paths(args)
    if not paths:
        print(
            f"No input TSV files found (paths={args.paths!r}, glob={args.glob!r}).",
            file=sys.stderr,
        )
        return 2

    buckets: Dict[Tuple[int, int], BucketAgg] = {}

    for p in paths:
        try:
            fs = parse_tsv(p)
        except Exception as e:
            print(f"skip: {p}: {e}", file=sys.stderr)
            continue

        key = (fs.elo_min, fs.elo_max)
        b = buckets.get(key)
        if b is None:
            b = BucketAgg(elo_min=fs.elo_min, elo_max=fs.elo_max)
            buckets[key] = b
        b.add_file(fs)

    if not buckets:
        print("No valid TSV files after parsing.", file=sys.stderr)
        return 2

    bucket_list = list(buckets.values())
    for b in bucket_list:
        b.finalize()
    bucket_list.sort(key=lambda x: (x.elo_min, x.elo_max))

    print_summary(bucket_list)

    if args.full:
        for b in bucket_list:
            print_bucket_full(b, top=args.top, min_games=args.min_games)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
