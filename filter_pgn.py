# filter_pgn.py
# All code/comments in English as requested.

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, Optional, TextIO, Tuple, List


def _parse_tag_line(line: str) -> Optional[Tuple[str, str]]:
    if not (line.startswith("[") and line.endswith("]")):
        return None
    inner = line[1:-1].strip()
    if " " not in inner:
        return None
    key, rest = inner.split(" ", 1)
    rest = rest.strip()
    if len(rest) < 2 or rest[0] != '"' or rest[-1] != '"':
        return None
    return key, rest[1:-1]


def read_games_raw(stream: TextIO) -> Iterable[Tuple[Dict[str, str], str]]:
    """
    Yield (headers_dict, raw_pgn_string) for each game.
    Flush the previous game when we see the next header block.
    """
    headers: Dict[str, str] = {}
    header_lines: List[str] = []
    movetext_lines: List[str] = []
    in_headers = False
    in_movetext = False

    def flush_game() -> Optional[Tuple[Dict[str, str], str]]:
        nonlocal headers, header_lines, movetext_lines, in_headers, in_movetext
        if not header_lines and not movetext_lines:
            return None
        raw = "\n".join(header_lines) + "\n\n" + "\n".join(movetext_lines) + "\n\n"
        out_headers = headers
        headers = {}
        header_lines = []
        movetext_lines = []
        in_headers = False
        in_movetext = False
        return out_headers, raw

    for line in stream:
        line = line.rstrip("\n")

        if line.startswith("["):
            # New header while we already have a game -> flush.
            if in_movetext and header_lines:
                out = flush_game()
                if out is not None:
                    yield out

            in_headers = True
            in_movetext = False
            header_lines.append(line)

            kv = _parse_tag_line(line.strip())
            if kv is not None:
                k, v = kv
                headers[k] = v
            continue

        if line.strip() == "":
            if in_headers:
                in_headers = False
                in_movetext = True
            continue

        if in_movetext or (header_lines and not in_headers):
            in_movetext = True
            movetext_lines.append(line)
            continue

    out = flush_game()
    if out is not None:
        yield out


def parse_elo(v: Optional[str]) -> Optional[int]:
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def is_rated(headers: Dict[str, str]) -> bool:
    return "rated" in (headers.get("Event", "").lower())


def is_standard(headers: Dict[str, str]) -> bool:
    v = (headers.get("Variant") or "").strip().lower()
    return (not v) or (v == "standard")


def is_blitz(headers: Dict[str, str]) -> bool:
    speed = (headers.get("Speed") or "").strip().lower()
    if speed:
        return speed == "blitz"
    return "blitz" in (headers.get("Event") or "").lower()


def is_excluded_termination(headers: Dict[str, str]) -> bool:
    term = (headers.get("Termination") or "").strip().lower()
    return term in {"rules infraction", "unterminated"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter Lichess PGN dumps by headers and output filtered PGN.")
    ap.add_argument("--elo-min", type=int, required=True, help="Keep only games where both players Elo >= this")
    ap.add_argument("--in", dest="inp", default="-", help="Input PGN (text), '-' for stdin")
    ap.add_argument("--out", dest="out", default="-", help="Output PGN, '-' for stdout")
    args = ap.parse_args()

    if args.inp == "-":
        src = sys.stdin
    else:
        src = open(args.inp, "r", encoding="utf-8", errors="replace")

    if args.out == "-":
        dst = sys.stdout
    else:
        dst = open(args.out, "w", encoding="utf-8")

    kept = 0
    seen = 0

    try:
        for headers, raw in read_games_raw(src):
            seen += 1

            if not is_rated(headers):
                continue
            if not is_standard(headers):
                continue
            if not is_blitz(headers):
                continue
            if is_excluded_termination(headers):
                continue

            we = parse_elo(headers.get("WhiteElo"))
            be = parse_elo(headers.get("BlackElo"))
            if we is None or be is None:
                continue
            if we < args.elo_min or be < args.elo_min:
                continue

            dst.write(raw)
            kept += 1

            # periodic lightweight progress to stderr
            if kept % 100000 == 0:
                print(f"kept={kept} seen={seen}", file=sys.stderr, flush=True)

    finally:
        if src is not sys.stdin:
            src.close()
        if dst is not sys.stdout:
            dst.close()

    print(f"done kept={kept} seen={seen}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
