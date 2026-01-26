#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Endgame stats from large PGN streams (Lichess exports), with tablebase-based WDL error rates.
#
# Key conventions (explicit):
# - material key is oriented: LEFT = side-to-move material, RIGHT = opponent material.
# - games[type]  = number of games where this type appears with side-to-move (i.e., encountered on a ply we analyzed).
# - plies[type]  = number of half-moves played from positions of this type (plies == steps).
# - errors[type] = number of move-errors on those plies (WDL change after the move, from mover POV).
#
# Notes:
# - --month is an output label; it does NOT filter by PGN dates.
# - Tablebase probe failures are fatal (abort the run).
# - If a move IMPROVES mover's WDL (loss->draw or draw->win, etc.), we log context and abort (this should not happen).

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Tuple
from collections import defaultdict

import chess
import chess.pgn
import chess.gaviota


# ----------------------------
# Constants
# ----------------------------

TRACK_TOTAL_PIECES: Set[int] = {3, 4, 5}
EXCLUDE_TOTAL_PIECES: Set[int] = {2}

MIN_PLYCOUNT = 35  # estimated from headers or movetext

# IMPORTANT: relative to current working directory.
GAVIOTA_ROOT = Path("../chess/gaviota")

NON_KING_PIECES = ["Q", "R", "B", "N", "P"]
PIECE_SYMBOL_TO_LETTER = {
    "k": "K",
    "q": "Q",
    "r": "R",
    "b": "B",
    "n": "N",
    "p": "P",
}


# ----------------------------
# Fast PGN raw reader (headers-first)
# ----------------------------

def _parse_header_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not (line.startswith("[") and line.endswith("]")):
        return None
    inner = line[1:-1].strip()
    if " " not in inner:
        return None
    key, rest = inner.split(" ", 1)
    rest = rest.strip()
    if not (rest.startswith('"') and rest.endswith('"')):
        return None
    return key, rest[1:-1]


def _int_or_none(tag_value: Optional[str]) -> Optional[int]:
    if not tag_value:
        return None
    try:
        return int(tag_value)
    except ValueError:
        return None


def _fast_ply_count_from_movetext(movetext: str) -> int:
    # Lightweight ply estimate: count SAN-like tokens that are not move numbers, results, obvious comments, or NAGs.
    tokens = movetext.replace("\n", " ").split()
    ply = 0
    for t in tokens:
        if t.endswith(".") and t[:-1].isdigit():
            continue
        if t in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        if t.startswith("{") or t.endswith("}"):
            continue
        if t.startswith("$"):
            continue
        ply += 1
    return ply


def read_games_raw(stream: TextIO) -> Iterable[Tuple[Dict[str, str], int, str]]:
    """Yield (headers, ply_est, raw_pgn) for each game, without SAN parsing.

    This is a streaming reader that flushes the previous game when a new header block begins.
    """
    headers: Dict[str, str] = {}
    header_lines: List[str] = []
    movetext_lines: List[str] = []

    state: str = "idle"  # idle | headers | movetext

    def flush_current() -> Optional[Tuple[Dict[str, str], int, str]]:
        if not header_lines:
            return None
        raw_pgn = "\n".join(header_lines) + "\n\n" + "\n".join(movetext_lines).rstrip() + "\n"
        ply_est = _int_or_none(headers.get("PlyCount")) or _fast_ply_count_from_movetext("\n".join(movetext_lines))
        return headers.copy(), ply_est, raw_pgn

    for line in stream:
        if line.startswith("["):
            if state == "movetext" and header_lines:
                flushed = flush_current()
                if flushed is not None:
                    yield flushed
                headers = {}
                header_lines = []
                movetext_lines = []
                state = "headers"
            elif state == "idle":
                headers = {}
                header_lines = []
                movetext_lines = []
                state = "headers"

            # state == headers: continue accumulating header lines
            header_lines.append(line.rstrip("\n"))
            parsed = _parse_header_line(line)
            if parsed:
                k, v = parsed
                headers[k] = v
            continue

        if state == "headers" and line.strip() == "":
            # End of headers; movetext begins.
            state = "movetext"
            continue

        if state == "movetext":
            movetext_lines.append(line.rstrip("\n"))
            continue

        # Ignore stray lines outside any game.

    flushed = flush_current()
    if flushed is not None:
        yield flushed


# ----------------------------
# Filters / metadata
# ----------------------------

def is_rated_event(event: str) -> bool:
    return "rated" in (event or "").lower()


def is_standard_variant_tag(variant: str) -> bool:
    v = (variant or "").strip().lower()
    return (not v) or (v == "standard")


def termination_is_time_forfeit(headers: Dict[str, str]) -> bool:
    return "time" in (headers.get("Termination") or "").lower()


def has_increment(headers: Dict[str, str]) -> bool:
    """Check if the game has a time increment (TimeControl 'seconds+increment' where increment > 0)."""
    tc = headers.get("TimeControl", "")
    if "+" in tc:
        try:
            parts = tc.split("+")
            if len(parts) >= 2:
                inc = int(parts[1])
                return inc > 0
        except ValueError:
            pass
    return False


def in_bucket(we: int, be: int, elo_min: int, elo_max: int) -> bool:
    s = we + be
    if not (2 * elo_min <= s < 2 * elo_max):
        return False
    if not (elo_min - 100 <= we < elo_max + 100):
        return False
    if not (elo_min - 100 <= be < elo_max + 100):
        return False
    return True


def out_path_for(month: str, elo_min: int, elo_max: int, out_dir: Path) -> Path:
    return out_dir / f"endgame_stats_{month}_elo{elo_min}-{elo_max}.tsv"


# ----------------------------
# Gaviota loading (compatible across python-chess versions)
# ----------------------------

def find_gaviota_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Gaviota root does not exist: {root}")
    dirs: List[Path] = []
    for p in root.rglob("*.gtb.cp4"):
        d = p.parent
        if d not in dirs:
            dirs.append(d)
    if not dirs:
        raise FileNotFoundError(f"No gaviota *.gtb.cp4 files found under: {root}")
    return sorted(dirs)


def _load_gtb_ctypes() -> Any:
    import ctypes
    import ctypes.util

    env = os.environ.get("GAVIOTA_LIB") or os.environ.get("GTB_LIB")
    candidates: List[str] = []
    if env:
        candidates.append(env)

    # Try local path found in the parent project structure
    local_lib = Path("../chess/Gaviota-Tablebases/libgtb.so.1.0.1")
    if local_lib.exists():
        candidates.append(str(local_lib))

    found = ctypes.util.find_library("gtb")
    if found:
        candidates.append(found)

    # Common names
    candidates.extend([
        "libgtb.so",
        "libgtb.so.0",
        "libgtb.dylib",
        "gtb.dll",
    ])

    last_err: Optional[Exception] = None
    for c in candidates:
        try:
            return ctypes.cdll.LoadLibrary(c)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not load Gaviota native library (libgtb). Last error: {last_err}")


def open_tablebase_native_fixed(dirs: List[Path]) -> chess.gaviota.NativeTablebase:
    """Open NativeTablebase with best-effort compatibility.

    We patch tb_restart to pass a null-terminated list if the underlying handle is accessible.
    """
    tb: chess.gaviota.NativeTablebase
    
    # Prioritize loading our specific lib via ctypes to avoid segfaults from system libs
    try:
        lib = _load_gtb_ctypes()
        tb = chess.gaviota.NativeTablebase(lib)
    except Exception:
        try:
            tb = chess.gaviota.NativeTablebase()  # fallback to python-chess default
        except TypeError:
            # Re-raise if we can't load it at all
            raise

    # Patch tb_restart only if the internal handle is accessible (best-effort).
    try:
        import ctypes  # noqa: F401

        if hasattr(tb, "libgtb") and hasattr(tb.libgtb, "tb_restart"):
            tb.libgtb.tb_restart.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            tb.libgtb.tb_restart.restype = ctypes.c_char_p

            def _tb_restart_null_terminated() -> None:
                if not hasattr(tb, "paths"): 
                    return
                n = len(tb.paths)
                c_paths = (ctypes.c_char_p * (n + 1))()
                c_paths[:n] = [p.encode("utf-8") for p in tb.paths]
                c_paths[n] = None

                verbosity = ctypes.c_int(1)
                compression_scheme = ctypes.c_int(4)
                _ = tb.libgtb.tb_restart(verbosity, compression_scheme, c_paths)
                tb.c_paths = c_paths

            tb._tb_restart = _tb_restart_null_terminated  # type: ignore[attr-defined]
    except Exception:
        pass

    # Add directories first.
    for d in dirs:
        tb.add_directory(str(d))

    return tb


# ----------------------------
# Material keying and exclusions
# ----------------------------

def total_pieces(board: chess.Board) -> int:
    return len(board.piece_map())


def _sq_color_parity(sq: int) -> int:
    # 0 for dark, 1 for light; a1 is dark.
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    return (file + rank) & 1


def _count_side(board: chess.Board, color: bool) -> Dict[str, int]:
    counts = {p: 0 for p in "KQRBNP"}
    for piece in board.piece_map().values():
        if piece.color != color:
            continue
        sym = chess.piece_symbol(piece.piece_type)
        counts[PIECE_SYMBOL_TO_LETTER[sym]] += 1
    return counts


def _extras(counts: Dict[str, int]) -> int:
    return counts["Q"] + counts["R"] + counts["B"] + counts["N"] + counts["P"]


def is_trivial_win_against_bare_king(board: chess.Board) -> bool:
    """Exclude trivial 'K vs heavy' positions from analysis.

    - If one side is bare king and opponent has any rook/queen, exclude.
    - If one side is bare king and opponent has >=3 non-king units (pieces/pawns), exclude.
    - Keep KBN vs K (special mate) and similar "not trivial" cases.
    """
    wc = _count_side(board, chess.WHITE)
    bc = _count_side(board, chess.BLACK)

    def bare_king(c: Dict[str, int]) -> bool:
        return _extras(c) == 0

    def is_kbn(c: Dict[str, int]) -> bool:
        return (c["B"] == 1 and c["N"] == 1 and c["Q"] == 0 and c["R"] == 0 and c["P"] == 0)

    def opponent_trivial(c: Dict[str, int]) -> bool:
        # KBN vs K should not be excluded.
        if is_kbn(c):
            return False
        # "K vs Q/R" excluded.
        if c["Q"] > 0 or c["R"] > 0:
            return True
        # "K vs 3+ units" excluded.
        if _extras(c) >= 3:
            return True
        return False

    if bare_king(wc) and opponent_trivial(bc):
        return True
    if bare_king(bc) and opponent_trivial(wc):
        return True
    return False


def _bishop_token_and_check(board: chess.Board, color: bool) -> Optional[str]:
    bishops = list(board.pieces(chess.BISHOP, color))
    n = len(bishops)
    if n == 0:
        return ""
    if n == 1:
        return "B"
    if n == 2:
        # Exclude same-colored bishops (promotion oddity).
        c0 = _sq_color_parity(bishops[0])
        c1 = _sq_color_parity(bishops[1])
        if c0 == c1:
            return None
        return "BB"
    return None  # 3+ bishops not supported


def _side_counts_no_bishops(board: chess.Board, color: bool) -> Tuple[int, int, int, int]:
    q = r = n = p = 0
    for piece in board.piece_map().values():
        if piece.color != color:
            continue
        pt = piece.piece_type
        if pt == chess.QUEEN:
            q += 1
        elif pt == chess.ROOK:
            r += 1
        elif pt == chess.KNIGHT:
            n += 1
        elif pt == chess.PAWN:
            p += 1
    return q, r, n, p


def build_key_for_side_to_move(board: chess.Board) -> Optional[str]:
    tot = total_pieces(board)
    if tot in EXCLUDE_TOTAL_PIECES:
        return None
    if tot not in TRACK_TOTAL_PIECES:
        return None

    if board.is_insufficient_material():
        return None

    if is_trivial_win_against_bare_king(board):
        return None

    left = board.turn
    right = not left

    left_tok = _bishop_token_and_check(board, left)
    if left_tok is None:
        return None
    right_tok = _bishop_token_and_check(board, right)
    if right_tok is None:
        return None

    # Opposite-colored bishops: exactly one bishop each side => normalize to "D"/"D".
    if left_tok == "B" and right_tok == "B":
        lb = next(iter(board.pieces(chess.BISHOP, left)))
        rb = next(iter(board.pieces(chess.BISHOP, right)))
        if _sq_color_parity(lb) != _sq_color_parity(rb):
            left_tok = "D"
            right_tok = "D"

    lq, lr, ln, lp = _side_counts_no_bishops(board, left)
    rq, rr, rn, rp = _side_counts_no_bishops(board, right)

    left_s = "K" + ("Q" * lq) + ("R" * lr) + left_tok + ("N" * ln) + ("P" * lp)
    right_s = "K" + ("Q" * rq) + ("R" * rr) + right_tok + ("N" * rn) + ("P" * rp)
    return f"{left_s}_{right_s}"


# ----------------------------
# Tablebase probing
# ----------------------------

def wdl_white_from_dtm(dtm_stm: int, stm_is_white: bool) -> int:
    if dtm_stm == 0:
        return 0
    wdl_stm = 1 if dtm_stm > 0 else -1
    return wdl_stm if stm_is_white else -wdl_stm


def probe_wdl_white(tb: Any, board: chess.Board) -> int:
    if board.is_checkmate():
        return -1 if board.turn == chess.WHITE else 1
    if board.is_stalemate():
        return 0
    dtm_stm = tb.probe_dtm(board)
    return wdl_white_from_dtm(dtm_stm, board.turn == chess.WHITE)


# ----------------------------
# Game analysis
# ----------------------------

def result_to_white_outcome(res: str) -> Optional[int]:
    r = (res or "").strip()
    if r == "1-0":
        return 1
    if r == "0-1":
        return -1
    if r == "1/2-1/2":
        return 0
    return None


@dataclass
class GameDeltas:
    keys_seen: Set[str]
    keys_with_error: Set[str]
    per_key_plies: Dict[str, int]
    per_key_errors: Dict[str, int]

    # Per-ply "capability" counters (evaluated at each analyzed ply, before the move).
    # can_draw means exactly-draw (WDL == 0) from the mover point of view.
    per_key_can_win: Dict[str, int]
    per_key_can_draw: Dict[str, int]

    # Missed opportunities (evaluated on the transition caused by the played move).
    per_key_missed_win_to_draw: Dict[str, int]
    per_key_missed_win_to_loss: Dict[str, int]
    per_key_missed_draw: Dict[str, int]

    time_loss_key: Optional[str]
    time_draw_key: Optional[str]


def analyze_game(game: chess.pgn.Game, headers: Dict[str, str], tb: Any) -> GameDeltas:
    board = game.board()

    keys_seen: Set[str] = set()
    keys_with_error: Set[str] = set()
    per_key_plies: Dict[str, int] = defaultdict(int)
    per_key_errors: Dict[str, int] = defaultdict(int)

    per_key_can_win: Dict[str, int] = defaultdict(int)
    per_key_can_draw: Dict[str, int] = defaultdict(int)
    per_key_missed_win_to_draw: Dict[str, int] = defaultdict(int)
    per_key_missed_win_to_loss: Dict[str, int] = defaultdict(int)
    per_key_missed_draw: Dict[str, int] = defaultdict(int)

    actual_white = result_to_white_outcome(headers.get("Result", ""))
    time_forfeit = termination_is_time_forfeit(headers)

    site = (headers.get("Site") or "").strip()
    site_tag = f'[Site "{site}"]' if site else "[Site \"<missing>\"]"

    # Avoid double TB probe: reuse "after" as next "before" when we remain in-track.
    cached_wdl_white: Optional[int] = None

    # Log the castling-rights anomaly once per game (only when we hit 3/4/5 pieces).
    castling_warned = False

    def probe_wdl_white_or_die(ctx: str) -> int:
        nonlocal castling_warned
        b = board.copy(stack=False)

        # In 3/4/5-piece endgames, castling rights should not exist; normalize and log once.
        if total_pieces(b) in TRACK_TOTAL_PIECES and b.castling_rights:
            if not castling_warned:
                cr = chess.Board(None).castling_xfen().replace("-", "")
                print(
                    f'WARNING: {site_tag} Someone managed to reach a {total_pieces(b)}-piece endgame while still being allowed to castle; '
                    f'zeroing castling rights for TB probe. ctx={ctx} fen="{b.fen()}"',
                    file=sys.stderr,
                    flush=True,
                )
                castling_warned = True
            b.castling_rights = 0

        try:
            return probe_wdl_white(tb, b)
        except Exception as e:
            raise RuntimeError(f"Tablebase probe failed ({ctx}). {site_tag} FEN={b.fen()}") from e

    # Mainline plies
    node = game
    ply_idx = 0
    for move in game.mainline_moves():
        ply_idx += 1

        key = build_key_for_side_to_move(board)
        if key is None:
            board.push(move)
            cached_wdl_white = None
            continue

        # We count this as the type "appearing at side-to-move".
        keys_seen.add(key)
        per_key_plies[key] += 1

        fen_before = board.fen()
        try:
            san = board.san(move)
        except Exception:
            san = "<san-unavailable>"
        uci = move.uci()

        had_cached_before = (cached_wdl_white is not None)

        w_before = cached_wdl_white if cached_wdl_white is not None else probe_wdl_white_or_die(f"before ply={ply_idx}")
        mover_is_white = (board.turn == chess.WHITE)

        # WDL from the mover's perspective, before making the move.
        before_mover = w_before if mover_is_white else -w_before

        # Capability counters: evaluated at each analyzed ply (before the move).
        if before_mover == 1:
            per_key_can_win[key] += 1
        if before_mover == 0:
            per_key_can_draw[key] += 1

        board.push(move)

        fen_after = board.fen()
        w_after = probe_wdl_white_or_die(f"after ply={ply_idx}")
        cached_wdl_white = w_after

        after_mover = w_after if mover_is_white else -w_after

        # Missed opportunities: classify WDL drops.
        if before_mover == 1 and after_mover == 0:
            per_key_missed_win_to_draw[key] += 1
        elif before_mover == 1 and after_mover == -1:
            per_key_missed_win_to_loss[key] += 1
        elif before_mover == 0 and after_mover == -1:
            per_key_missed_draw[key] += 1

        # Invariant: with perfect TB opponent evaluation, a single move should not "improve" mover's WDL category.
        # If it does, something is inconsistent (probe / keying / caching / board normalization).
        if after_mover > before_mover:
            # Log maximum useful context, then abort.
            print("ERROR: WDL improvement detected; aborting.", file=sys.stderr)
            print(f"  {site_tag}", file=sys.stderr)
            print(f"  ply={ply_idx} mover={'White' if mover_is_white else 'Black'}", file=sys.stderr)
            print(f"  key={key}", file=sys.stderr)
            print(f"  move_san={san} move_uci={uci}", file=sys.stderr)
            print(f"  fen_before={fen_before}", file=sys.stderr)
            print(f"  fen_after ={fen_after}", file=sys.stderr)
            print(f"  wdl_white_before={w_before} wdl_white_after={w_after}", file=sys.stderr)
            print(f"  wdl_mover_before={before_mover} wdl_mover_after={after_mover}", file=sys.stderr)
            print(f"  cached_before={'yes' if had_cached_before else 'no'}", file=sys.stderr)
            raise RuntimeError(f"WDL improved for mover at ply={ply_idx}. {site_tag}")

        if after_mover != before_mover:
            per_key_errors[key] += 1
            keys_with_error.add(key)

    # Time Outcome Attribution
    time_loss_key: Optional[str] = None
    time_draw_key: Optional[str] = None

    if time_forfeit:
        if actual_white in (-1, 1):
            loser_is_white = (actual_white == -1)
            b2 = board.copy(stack=False)
            b2.turn = chess.WHITE if loser_is_white else chess.BLACK
            k_loser = build_key_for_side_to_move(b2)
            if k_loser is not None:
                time_loss_key = k_loser
                keys_seen.add(k_loser)
        elif actual_white == 0:
            # Assume the side whose turn it was at end of PGN is the one who flagged.
            flagger_is_white = (board.turn == chess.WHITE)
            b2 = board.copy(stack=False)
            b2.turn = chess.WHITE if flagger_is_white else chess.BLACK
            k_flagger = build_key_for_side_to_move(b2)
            if k_flagger is not None:
                time_draw_key = k_flagger
                keys_seen.add(k_flagger)

    return GameDeltas(
        keys_seen=keys_seen,
        keys_with_error=keys_with_error,
        per_key_plies=dict(per_key_plies),
        per_key_errors=dict(per_key_errors),
        per_key_can_win=dict(per_key_can_win),
        per_key_can_draw=dict(per_key_can_draw),
        per_key_missed_win_to_draw=dict(per_key_missed_win_to_draw),
        per_key_missed_win_to_loss=dict(per_key_missed_win_to_loss),
        per_key_missed_draw=dict(per_key_missed_draw),
        time_loss_key=time_loss_key,
        time_draw_key=time_draw_key,
    )


# ----------------------------
# Aggregation / output
# ----------------------------

@dataclass
class Stats:
    games_seen: int = 0
    games_used: int = 0
    games_skipped_short: int = 0
    games_skipped_parse: int = 0

    relevant_games: int = 0
    relevant_games_with_increment: int = 0
    relevant_games_without_increment: int = 0

    plies_total: int = 0

    # Aggregate move-quality counters (mover POV)
    errors_total: int = 0
    can_win_total: int = 0
    can_draw_total: int = 0
    missed_win_to_draw_total: int = 0
    missed_win_to_loss_total: int = 0
    missed_draw_total: int = 0

    time_loss_games_total: int = 0
    time_draw_games_total: int = 0


def write_tsv(
    out_path: Path,
    month: str,
    elo_min: int,
    elo_max: int,
    s: Stats,
    per_key_games: Dict[str, int],
    per_key_games_with_error: Dict[str, int],
    per_key_plies_total: Dict[str, int],
    per_key_errors_total: Dict[str, int],
    per_key_can_win_total: Dict[str, int],
    per_key_can_draw_total: Dict[str, int],
    per_key_missed_win_to_draw_total: Dict[str, int],
    per_key_missed_win_to_loss_total: Dict[str, int],
    per_key_missed_draw_total: Dict[str, int],
    per_key_time_losses: Dict[str, int],
    per_key_time_draws: Dict[str, int],
) -> None:
    denom_used = s.games_used if s.games_used > 0 else 1

    lines: List[str] = []
    lines.append(f"# month={month}")
    lines.append(f"# elo_min={elo_min}")
    lines.append(f"# elo_max={elo_max}")
    lines.append(f"# games_seen={s.games_seen}")
    lines.append(f"# games_used={s.games_used}")
    lines.append(f"# games_skipped_short_plycount<{MIN_PLYCOUNT}={s.games_skipped_short}")
    lines.append(f"# games_skipped_parse={s.games_skipped_parse}")

    lines.append(f"# relevant_games={s.relevant_games}")
    lines.append(f"# relevant_games_with_increment={s.relevant_games_with_increment}")
    lines.append(f"# relevant_games_without_increment={s.relevant_games_without_increment}")
    lines.append(f"# pct_relevant_over_games_used={(s.relevant_games / denom_used) * 100.0:.6f}")

    lines.append(f"# plies_total={s.plies_total}")
    lines.append(f"# errors_total={s.errors_total}")
    lines.append(
        f"# errors_per_ply_pct_total={((s.errors_total / s.plies_total) * 100.0) if s.plies_total else 0.0:.8f}"
    )

    lines.append(f"# can_win_total={s.can_win_total}")
    lines.append(f"# can_win_pct_over_plies_total={((s.can_win_total / s.plies_total) * 100.0) if s.plies_total else 0.0:.8f}")
    lines.append(f"# can_draw_total={s.can_draw_total}")
    lines.append(f"# can_draw_pct_over_plies_total={((s.can_draw_total / s.plies_total) * 100.0) if s.plies_total else 0.0:.8f}")

    lines.append(f"# missed_win_to_draw_total={s.missed_win_to_draw_total}")
    lines.append(f"# missed_win_to_draw_pct_over_can_win_total={((s.missed_win_to_draw_total / s.can_win_total) * 100.0) if s.can_win_total else 0.0:.8f}")
    lines.append(f"# missed_win_to_loss_total={s.missed_win_to_loss_total}")
    lines.append(f"# missed_win_to_loss_pct_over_can_win_total={((s.missed_win_to_loss_total / s.can_win_total) * 100.0) if s.can_win_total else 0.0:.8f}")
    lines.append(f"# missed_draw_total={s.missed_draw_total}")
    lines.append(f"# missed_draw_pct_over_can_draw_total={((s.missed_draw_total / s.can_draw_total) * 100.0) if s.can_draw_total else 0.0:.8f}")


    lines.append(f"# time_loss_games_total={s.time_loss_games_total}")
    lines.append(f"# time_draw_games_total={s.time_draw_games_total}")
    lines.append(f"# pct_time_loss_over_relevant_games={(s.time_loss_games_total / s.relevant_games * 100.0) if s.relevant_games else 0.0:.6f}")

    lines.append(
        "material	"
        "games	games_pct_over_used	"
        "plies	avg_plies_per_game	"
        "can_win	can_win_pct_over_plies	"
        "can_draw	can_draw_pct_over_plies	"
        "games_with_error	error_game_pct	"
        "errors	errors_per_ply_pct	"
        "missed_win_to_draw	missed_win_to_draw_pct_over_can_win	"
        "missed_win_to_loss	missed_win_to_loss_pct_over_can_win	"
        "missed_draw	missed_draw_pct_over_can_draw	"
        "time_losses	time_loss_pct	"
        "time_draws	time_draw_pct"
    )

    # Only output lines with non-zero games.

    for k in sorted(per_key_games.keys()):
        g = per_key_games.get(k, 0)
        if g == 0:
            continue

        gerr = per_key_games_with_error.get(k, 0)
        plies = per_key_plies_total.get(k, 0)
        errs = per_key_errors_total.get(k, 0)

        can_win = per_key_can_win_total.get(k, 0)
        can_draw = per_key_can_draw_total.get(k, 0)
        mw2d = per_key_missed_win_to_draw_total.get(k, 0)
        mw2l = per_key_missed_win_to_loss_total.get(k, 0)
        md = per_key_missed_draw_total.get(k, 0)

        tl = per_key_time_losses.get(k, 0)
        td = per_key_time_draws.get(k, 0)

        pct_used = (g / denom_used) * 100.0
        avg_plies = (plies / g) if g > 0 else 0.0

        can_win_pct = (can_win / plies) * 100.0 if plies > 0 else 0.0
        can_draw_pct = (can_draw / plies) * 100.0 if plies > 0 else 0.0

        err_game_pct = (gerr / g) * 100.0 if g > 0 else 0.0
        err_per_ply_pct = (errs / plies) * 100.0 if plies > 0 else 0.0

        mw2d_pct = (mw2d / can_win) * 100.0 if can_win > 0 else 0.0
        mw2l_pct = (mw2l / can_win) * 100.0 if can_win > 0 else 0.0
        md_pct = (md / can_draw) * 100.0 if can_draw > 0 else 0.0

        tl_pct = (tl / g) * 100.0 if g > 0 else 0.0
        td_pct = (td / g) * 100.0 if g > 0 else 0.0

        lines.append(
            f"{k}	"
            f"{g}	{pct_used:.6f}	"
            f"{plies}	{avg_plies:.6f}	"
            f"{can_win}	{can_win_pct:.6f}	"
            f"{can_draw}	{can_draw_pct:.6f}	"
            f"{gerr}	{err_game_pct:.6f}	"
            f"{errs}	{err_per_ply_pct:.6f}	"
            f"{mw2d}	{mw2d_pct:.6f}	"
            f"{mw2l}	{mw2l_pct:.6f}	"
            f"{md}	{md_pct:.6f}	"
            f"{tl}	{tl_pct:.6f}	"
            f"{td}	{td_pct:.6f}"
        )

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(out_path)


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Input PGN file (or '-' for stdin).")
    ap.add_argument("--month", required=True, help="Month label for output (e.g., 2025-12).")
    ap.add_argument("--elo-min", type=int, required=True)
    ap.add_argument("--elo-max", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("."))

    ap.add_argument("--log-every", type=float, default=60.0, help="Seconds between progress logs; 0 disables.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    elo_min = args.elo_min
    elo_max = args.elo_max
    if elo_max <= elo_min:
        raise ValueError("elo-max must be strictly greater than elo-min.")

    out_path = out_path_for(args.month, elo_min, elo_max, args.out_dir)

    gaviota_dirs = find_gaviota_dirs(GAVIOTA_ROOT)
    tb = open_tablebase_native_fixed(gaviota_dirs)

    s = Stats()

    per_key_games: Dict[str, int] = defaultdict(int)
    per_key_games_with_error: Dict[str, int] = defaultdict(int)
    per_key_plies_total: Dict[str, int] = defaultdict(int)
    per_key_errors_total: Dict[str, int] = defaultdict(int)
    per_key_can_win_total: Dict[str, int] = defaultdict(int)
    per_key_can_draw_total: Dict[str, int] = defaultdict(int)
    per_key_missed_win_to_draw_total: Dict[str, int] = defaultdict(int)
    per_key_missed_win_to_loss_total: Dict[str, int] = defaultdict(int)
    per_key_missed_draw_total: Dict[str, int] = defaultdict(int)
    per_key_time_losses: Dict[str, int] = defaultdict(int)
    per_key_time_draws: Dict[str, int] = defaultdict(int)

    if args.pgn == "-":
        pgn_stream = sys.stdin
    else:
        pgn_stream = open(args.pgn, "r", encoding="utf-8", errors="replace")

    t0 = time.time()
    last_log = t0

    def dump_progress(now: float) -> None:
        denom = s.games_used if s.games_used > 0 else 1
        pct_any = (s.relevant_games / denom) * 100.0
        err_rate_pct = ((s.errors_total / s.plies_total) * 100.0) if s.plies_total else 0.0
        tl_pct = (s.time_loss_games_total / denom) * 100.0
        print(
            "progress:\n"
            f"  month={args.month} elo=[{elo_min},{elo_max}[ elapsed={(now - t0)/60:.1f}m "
            f"games_seen={fmt_int(s.games_seen)} games_used={fmt_int(s.games_used)} "
            f"skipped_short={fmt_int(s.games_skipped_short)} skipped_parse={fmt_int(s.games_skipped_parse)} "
            f"relevant_games={fmt_int(s.relevant_games)} pct_rel={pct_any:.3f}% "
            f"plies_total={fmt_int(s.plies_total)} errors_total={fmt_int(s.errors_total)} err/ply={err_rate_pct:.4f}% "
            f"time_losses={fmt_int(s.time_loss_games_total)} tl_pct={tl_pct:.3f}% "
            f"time_draws={fmt_int(s.time_draw_games_total)}\n",
            file=sys.stderr,
            flush=True,
        )

    def write_out() -> None:
        write_tsv(
            out_path=out_path,
            month=args.month,
            elo_min=elo_min,
            elo_max=elo_max,
            s=s,
            per_key_games=dict(per_key_games),
            per_key_games_with_error=dict(per_key_games_with_error),
            per_key_plies_total=dict(per_key_plies_total),
            per_key_errors_total=dict(per_key_errors_total),
            per_key_can_win_total=dict(per_key_can_win_total),
            per_key_can_draw_total=dict(per_key_can_draw_total),
            per_key_missed_win_to_draw_total=dict(per_key_missed_win_to_draw_total),
            per_key_missed_win_to_loss_total=dict(per_key_missed_win_to_loss_total),
            per_key_missed_draw_total=dict(per_key_missed_draw_total),
            per_key_time_losses=dict(per_key_time_losses),
            per_key_time_draws=dict(per_key_time_draws),
        )

    try:
        for headers, ply_est, raw_pgn in read_games_raw(pgn_stream):
            s.games_seen += 1

            # Early filters (headers-only).
            if not is_rated_event(headers.get("Event", "")):
                continue
            if not is_standard_variant_tag(headers.get("Variant", "")):
                continue

            we = _int_or_none(headers.get("WhiteElo"))
            be = _int_or_none(headers.get("BlackElo"))
            if we is None or be is None:
                continue
            if not in_bucket(we, be, elo_min, elo_max):
                continue

            if ply_est < MIN_PLYCOUNT:
                s.games_skipped_short += 1
                continue

            # Parse PGN (only now).
            try:
                game = chess.pgn.read_game(io.StringIO(raw_pgn))
                if game is None:
                    s.games_skipped_parse += 1
                    continue
            except Exception:
                s.games_skipped_parse += 1
                continue

            s.games_used += 1

            # Analyze (TB issues are fatal by design).
            deltas = analyze_game(game, headers, tb)

            if deltas.keys_seen:
                s.relevant_games += 1
                if has_increment(headers):
                    s.relevant_games_with_increment += 1
                else:
                    s.relevant_games_without_increment += 1

                for k in deltas.keys_seen:
                    per_key_games[k] += 1
                for k in deltas.keys_with_error:
                    per_key_games_with_error[k] += 1

            for k, v in deltas.per_key_plies.items():
                per_key_plies_total[k] += v
                s.plies_total += v

            for k, v in deltas.per_key_errors.items():
                per_key_errors_total[k] += v
                s.errors_total += v


            for k, v in deltas.per_key_can_win.items():
                per_key_can_win_total[k] += v
                s.can_win_total += v

            for k, v in deltas.per_key_can_draw.items():
                per_key_can_draw_total[k] += v
                s.can_draw_total += v

            for k, v in deltas.per_key_missed_win_to_draw.items():
                per_key_missed_win_to_draw_total[k] += v
                s.missed_win_to_draw_total += v

            for k, v in deltas.per_key_missed_win_to_loss.items():
                per_key_missed_win_to_loss_total[k] += v
                s.missed_win_to_loss_total += v

            for k, v in deltas.per_key_missed_draw.items():
                per_key_missed_draw_total[k] += v
                s.missed_draw_total += v

            if deltas.time_loss_key is not None:
                per_key_time_losses[deltas.time_loss_key] += 1
                s.time_loss_games_total += 1
            
            if deltas.time_draw_key is not None:
                per_key_time_draws[deltas.time_draw_key] += 1
                s.time_draw_games_total += 1

            now = time.time()
            if args.log_every > 0 and (now - last_log) >= args.log_every:
                dump_progress(now)
                write_out()
                last_log = now

    finally:
        if args.pgn != "-":
            pgn_stream.close()
        try:
            tb.close()
        except Exception:
            pass

    write_out()
    dump_progress(time.time())
    print(f"done: out={out_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
