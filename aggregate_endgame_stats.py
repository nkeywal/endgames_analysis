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
from typing import Dict, List, Optional, Tuple, Set


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
    can_win: int = 0
    can_draw: int = 0
    missed_win_to_draw: int = 0
    missed_win_to_loss: int = 0
    missed_draw: int = 0


@dataclass
class BucketAgg:
    elo_min: int
    elo_max: int
    months: List[str] = field(default_factory=list)
    files: List[Path] = field(default_factory=list)

    # Exact header-level sums.
    games_seen: int = 0
    games_used: int = 0
    games_skipped_short: int = 0
    games_skipped_parse: int = 0
    relevant_games: int = 0
    plies_total: int = 0
    errors_total: int = 0
    can_win_total: int = 0
    can_draw_total: int = 0
    missed_win_to_draw_total: int = 0
    missed_win_to_loss_total: int = 0
    missed_draw_total: int = 0
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

        self.games_seen += geti("games_seen")
        self.games_used += geti("games_used")
        self.games_skipped_short += geti("games_skipped_short_plycount<35")
        self.games_skipped_parse += geti("games_skipped_parse")
        self.relevant_games += geti("relevant_games")
        self.plies_total += geti("plies_total")
        self.errors_total += geti("errors_total")
        
        self.can_win_total += geti("can_win_total")
        self.can_draw_total += geti("can_draw_total")
        self.missed_win_to_draw_total += geti("missed_win_to_draw_total")
        self.missed_win_to_loss_total += geti("missed_win_to_loss_total")
        self.missed_draw_total += geti("missed_draw_total")
        
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
            
            cw = val("can_win")
            cd = val("can_draw")
            mw2d = val("missed_win_to_draw")
            mw2l = val("missed_win_to_loss")
            md = val("missed_draw")

            agg = self.per_material.get(mat)
            if agg is None:
                agg = MaterialAgg()
                self.per_material[mat] = agg

            agg.games += g
            agg.plies += plies
            agg.games_with_error += gerr
            agg.errors_total += errs
            agg.time_losses += tl
            
            agg.can_win += cw
            agg.can_draw += cd
            agg.missed_win_to_draw += mw2d
            agg.missed_win_to_loss += mw2l
            agg.missed_draw += md

    def finalize(self) -> None:
        self.months = sorted(set(self.months))
        self.files = sorted(set(self.files), key=str)

    def errors_per_ply_pct_total(self) -> float:
        return ((self.errors_total / self.plies_total) * 100.0) if self.plies_total else 0.0

    def pct_relevant_over_games_used(self) -> float:
        return ((self.relevant_games / self.games_used) * 100.0) if self.games_used else 0.0

    def pct_time_loss_over_relevant_games(self) -> float:
        return ((self.time_loss_games_total / self.relevant_games) * 100.0) if self.relevant_games else 0.0


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
        raise ValueError(f"Missing TSV header row (material\t ...) in {path}")

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
        ("%rel", 8),
        ("err/ply%", 10),
        ("err/win%", 10),
        ("err/draw%", 10),
        ("%time_loss", 10),
    ]
    fmt = " ".join([f"{{:{w}}}" for _, w in cols])

    print(fmt.format(*[c[0] for c in cols]))
    print(fmt.format(*["-" * c[1] for c in cols]))

    for b in buckets:
        bucket_s = f"{b.elo_min}-{b.elo_max}"
        months_s = str(len(b.months))
        games_used_s = str(b.games_used)
        pct_rel_s = f"{b.pct_relevant_over_games_used():.3f}"
        err_s = f"{b.errors_per_ply_pct_total():.3f}"
        
        missed_win_total = b.missed_win_to_draw_total + b.missed_win_to_loss_total
        err_win_s = f"{(missed_win_total / b.can_win_total * 100.0) if b.can_win_total else 0.0:.3f}"
        err_draw_s = f"{(b.missed_draw_total / b.can_draw_total * 100.0) if b.can_draw_total else 0.0:.3f}"
        
        tl_s = f"{b.pct_time_loss_over_relevant_games():.3f}"
        print(fmt.format(bucket_s, months_s, games_used_s, pct_rel_s, err_s, err_win_s, err_draw_s, tl_s))


def print_bucket_full(b: BucketAgg, top: int, min_games: int, keep_types: Optional[Set[str]], raw_only: bool = False) -> None:
    print()
    print(f"### AGG bucket elo{b.elo_min}-{b.elo_max}")
    print("# month=AGG")
    print(f"# months={','.join(b.months)}")
    print(f"# files={len(b.files)}")
    print(f"# elo_min={b.elo_min}")
    print(f"# elo_max={b.elo_max}")
    print(f"# games_seen={b.games_seen}")
    print(f"# games_used={b.games_used}")
    print(f"# games_skipped_short_plycount<35={b.games_skipped_short}")
    print(f"# games_skipped_parse={b.games_skipped_parse}")
    print(f"# relevant_games={b.relevant_games}")
    print(f"# pct_relevant_over_games_used={b.pct_relevant_over_games_used():.3f}")
    print(f"# plies_total={b.plies_total}")
    print(f"# errors_total={b.errors_total}")
    print(f"# errors_per_ply_pct_total={b.errors_per_ply_pct_total():.3f}")
    
    print(f"# can_win_total={b.can_win_total}")
    print(f"# can_win_pct_over_plies_total={((b.can_win_total / b.plies_total) * 100.0) if b.plies_total else 0.0:.3f}")
    print(f"# can_draw_total={b.can_draw_total}")
    print(f"# can_draw_pct_over_plies_total={((b.can_draw_total / b.plies_total) * 100.0) if b.plies_total else 0.0:.3f}")

    print(f"# missed_win_to_draw_total={b.missed_win_to_draw_total}")
    print(f"# missed_win_to_draw_pct_over_can_win_total={((b.missed_win_to_draw_total / b.can_win_total) * 100.0) if b.can_win_total else 0.0:.3f}")
    print(f"# missed_win_to_loss_total={b.missed_win_to_loss_total}")
    print(f"# missed_win_to_loss_pct_over_can_win_total={((b.missed_win_to_loss_total / b.can_win_total) * 100.0) if b.can_win_total else 0.0:.3f}")
    print(f"# missed_draw_total={b.missed_draw_total}")
    print(f"# missed_draw_pct_over_can_draw_total={((b.missed_draw_total / b.can_draw_total) * 100.0) if b.can_draw_total else 0.0:.3f}")

    missed_win_total = b.missed_win_to_draw_total + b.missed_win_to_loss_total
    err_win_opp_total = (missed_win_total / b.can_win_total * 100.0) if b.can_win_total else 0.0
    err_draw_opp_total = (b.missed_draw_total / b.can_draw_total * 100.0) if b.can_draw_total else 0.0

    print(f"# error_per_win_opportunity_total={err_win_opp_total:.3f}")
    print(f"# error_per_draw_opportunity_total={err_draw_opp_total:.3f}")

    print(f"# time_loss_games_total={b.time_loss_games_total}")
    print(f"# pct_time_loss_over_relevant_games={b.pct_time_loss_over_relevant_games():.3f}")

    if raw_only:
        print(
            "material\t"
            "games\t"
            "plies\t"
            "can_win\t"
            "can_draw\t"
            "games_with_error\t"
            "errors\t"
            "missed_win_to_draw\t"
            "missed_win_to_loss\t"
            "missed_draw\t"
            "time_losses"
        )
    else:
        print(
            "material\t"
            "games\tgames_pct_over_used\t"
            "plies\tavg_plies_per_game\t"
            "can_win\tcan_win_pct_over_plies\t"
            "can_draw\tcan_draw_pct_over_plies\t"
            "games_with_error\terror_game_pct\t"
            "errors\terrors_per_ply_pct\t"
            "missed_win_to_draw\tmissed_win_to_draw_pct_over_can_win\t"
            "missed_win_to_loss\tmissed_win_to_loss_pct_over_can_win\t"
            "missed_draw\tmissed_draw_pct_over_can_draw\t"
            "err_win_opp_pct\terr_draw_opp_pct\t"
            "time_losses\ttime_loss_pct"
        )

    denom_used = b.games_used if b.games_used > 0 else 1

    items = list(b.per_material.items())
    items.sort(key=lambda kv: (-kv[1].games, kv[0]))

    shown = 0
    for mat, a in items:
        if keep_types is not None and mat not in keep_types:
            continue
        if a.games < min_games:
            continue
        if top and shown >= top:
            break

        if raw_only:
            print(
                f"{mat}\t"
                f"{a.games}\t"
                f"{a.plies}\t"
                f"{a.can_win}\t"
                f"{a.can_draw}\t"
                f"{a.games_with_error}\t"
                f"{a.errors_total}\t"
                f"{a.missed_win_to_draw}\t"
                f"{a.missed_win_to_loss}\t"
                f"{a.missed_draw}\t"
                f"{a.time_losses}"
            )
        else:
            games = a.games
            pct_used = (games / denom_used) * 100.0
            avg_plies = (a.plies / games) if games else 0.0
            
            can_win_pct = (a.can_win / a.plies * 100.0) if a.plies else 0.0
            can_draw_pct = (a.can_draw / a.plies * 100.0) if a.plies else 0.0
            
            err_game_pct = (a.games_with_error / games * 100.0) if games else 0.0
            err_per_ply_pct = (a.errors_total / a.plies * 100.0) if a.plies else 0.0
            
            mw2d_pct = (a.missed_win_to_draw / a.can_win * 100.0) if a.can_win > 0 else 0.0
            mw2l_pct = (a.missed_win_to_loss / a.can_win * 100.0) if a.can_win > 0 else 0.0
            md_pct = (a.missed_draw / a.can_draw * 100.0) if a.can_draw > 0 else 0.0
            
            missed_win = a.missed_win_to_draw + a.missed_win_to_loss
            err_win_opp_pct = (missed_win / a.can_win * 100.0) if a.can_win > 0 else 0.0
            err_draw_opp_pct = (a.missed_draw / a.can_draw * 100.0) if a.can_draw > 0 else 0.0
            
            tl_pct = (a.time_losses / games * 100.0) if games else 0.0

            print(
                f"{mat}\t"
                f"{games}\t{pct_used:.3f}\t"
                f"{a.plies}\t{avg_plies:.3f}\t"
                f"{a.can_win}\t{can_win_pct:.3f}\t"
                f"{a.can_draw}\t{can_draw_pct:.3f}\t"
                f"{a.games_with_error}\t{err_game_pct:.3f}\t"
                f"{a.errors_total}\t{err_per_ply_pct:.3f}\t"
                f"{a.missed_win_to_draw}\t{mw2d_pct:.3f}\t"
                f"{a.missed_win_to_loss}\t{mw2l_pct:.3f}\t"
                f"{a.missed_draw}\t{md_pct:.3f}\t"
                f"{err_win_opp_pct:.3f}\t{err_draw_opp_pct:.3f}\t"
                f"{a.time_losses}\t{tl_pct:.3f}"
            )
        shown += 1


def print_metrics_explanation() -> None:
    text = """
METRICS

Counts (games)
- games_seen: games encountered in the input.
- games_used: games kept after filtering (Elo bucket, Standard, non-Bullet) and successfully parsed.
- games_skipped_short_plycount<35: games rejected because total plycount < 35.

Relevance
- relevant_games: games that contain at least one analyzed ply in a non-trivial 3–5 piece position (kings included).
- %rel: relevant_games / games_used.

Non-trivial / excluded positions
- Positions are analyzed only if they have 3–5 total pieces and are not excluded (insufficient material, “obvious” patterns you skip, rare promotion/bishop edge-cases).

Counts (plies)
- plies_total: total number of analyzed plies across all relevant games.

Mover status before each analyzed move (perfect play)
- can_win_total: analyzed plies where the mover had a forced win.
- can_draw_total: analyzed plies where the mover could force a draw but not a win.
- can_win_pct_over_plies_total: can_win_total / plies_total.
- can_draw_pct_over_plies_total: can_draw_total / plies_total.

Errors (a move reduces what the mover can still guarantee)
- missed_win_to_draw_total: win → draw.
- missed_win_to_loss_total: win → loss.
- missed_draw_total: draw → loss.
- errors_total: sum of the three counters above.

Key rates
- missed_win_to_draw_pct_over_can_win_total: missed_win_to_draw_total / can_win_total (share of winning opportunities downgraded to a draw).
- missed_win_to_loss_pct_over_can_win_total: missed_win_to_loss_total / can_win_total (share of winning opportunities downgraded to a loss).
- missed_draw_pct_over_can_draw_total: missed_draw_total / can_draw_total (share of drawing opportunities downgraded to a loss).

Aggregate rates
- err/ply% (errors_per_ply_pct_total): errors_total / plies_total.
- err/win% (error_per_win_opportunity_total): (missed_win_to_draw_total + missed_win_to_loss_total) / can_win_total (overall failure rate to convert when a win was available).
- err/draw% (error_per_draw_opportunity_total): missed_draw_total / can_draw_total (overall failure rate to hold when a draw was available).

Time forfeits (conditioned on relevance)
- time_loss_games_total: relevant games that ended by time forfeit (counted separately from errors).
- %time_loss: time_loss_games_total / relevant_games.

Per-material table (same definitions, restricted to one oriented key)
- material: LEFT_RIGHT where LEFT is the side to move.
- games: relevant games that contain at least one analyzed ply with this key.
- plies: analyzed plies with this key.
- avg_plies_per_game: plies / games.
- games_with_error: games with at least one error in this key.
- error_game_pct: games_with_error / games.
"""
    print(text.strip())


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
    # --full is now default behavior, flag removed.
    ap.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only the top N materials by games (0=all).",
    )
    ap.add_argument(
        "--types",
        help="Comma-separated list of material keys to show (e.g. KR_KP,KQ_KQP). Symmetric (A_B implies B_A).",
    )
    ap.add_argument(
        "--min-games",
        type=int,
        default=0,
        help="Hide materials with fewer than this many games.",
    )
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="Explain the meaning of each metric and exit.",
    )
    ap.add_argument(
        "--notes",
        action="store_true",
        help="Print notes about inferred denominators for per-material aggregation.",
    )
    ap.add_argument(
        "--raw-only",
        action="store_true",
        help="Output only raw integer counts in the per-material table (suppress percentages and rates).",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.metrics:
        print_metrics_explanation()
        return 0

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

    keep_types: Optional[Set[str]] = None
    if args.types:
        keep_types = set()
        for t in args.types.split(","):
            t = t.strip()
            if not t:
                continue
            keep_types.add(t)
            # Handle symmetry A_B -> B_A
            if "_" in t:
                parts = t.split("_", 1)
                rev = f"{parts[1]}_{parts[0]}"
                keep_types.add(rev)

    print_summary(bucket_list)

    for b in bucket_list:
        print_bucket_full(b, top=args.top, min_games=args.min_games, keep_types=keep_types, raw_only=args.raw_only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())