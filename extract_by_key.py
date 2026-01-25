# extract_by_key.py
# -----------------------------------------------------------------------------
# Script to extract the N-th game from a compressed Lichess PGN (.zst)
# that reaches a specific material configuration (Key).
#
# Usage:
#   python extract_by_key.py KP_K lichess_file.pgn.zst --index 1
# -----------------------------------------------------------------------------

import argparse
import io
import sys
import zstandard as zstd
import chess
import chess.pgn
from typing import Optional, Dict, Tuple, List, Set

# ----------------------------
# Configuration & Constants
# ----------------------------

# Set of total piece counts to consider (Kings included).
TRACK_TOTAL_PIECES: Set[int] = {3, 4, 5}

# Map piece symbols to Uppercase letters only.
PIECE_SYMBOL_TO_LETTER = {
    "k": "K",
    "q": "Q",
    "r": "R",
    "b": "B",
    "n": "N",
    "p": "P",
}

# ----------------------------
# Key Generation Logic
# (Must match the logic in endgame_stats.py)
# ----------------------------

def total_pieces(board: chess.Board) -> int:
    """Returns the total number of pieces on the board."""
    return len(board.piece_map())

def _sq_color_parity(sq: int) -> int:
    """Returns 0 for dark squares, 1 for light squares."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    return (file + rank) & 1

def _bishop_token_or_none(board: chess.Board, color: bool) -> Optional[str]:
    """
    Determines the bishop token ('B', 'BB') for a side.
    Returns None if the configuration is excluded (e.g., 3 bishops).
    """
    bishops = list(board.pieces(chess.BISHOP, color))
    n = len(bishops)
    if n == 0:
        return ""
    if n == 1:
        return "B"
    if n == 2:
        c0 = _sq_color_parity(bishops[0])
        c1 = _sq_color_parity(bishops[1])
        if c0 == c1:
            # Two bishops on same color = likely promotion -> Exclude.
            return None
        return "BB"
    return None

def _side_counts_no_bishops(board: chess.Board, color: bool) -> Tuple[int, int, int, int]:
    """Counts Q, R, N, P for a specific side."""
    q = r = n = p = 0
    for _, piece in board.piece_map().items():
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
    """
    Generates the oriented material key (Left = Side to move).
    Format example: "KP_K" (Side to move has King+Pawn, Opponent has King).
    """
    tot = total_pieces(board)
    if tot not in TRACK_TOTAL_PIECES:
        return None
    
    left = board.turn
    right = not left

    left_tok = _bishop_token_or_none(board, left)
    if left_tok is None:
        return None
    right_tok = _bishop_token_or_none(board, right)
    if right_tok is None:
        return None

    # Handle Opposite-Colored Bishops ("D")
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
# Main Script
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Find the N-th game matching a specific material key in a compressed PGN.")
    parser.add_argument("key", help="Target key (e.g., KP_K, KB_KB, KD_KD). Must respect piece order KQRBNP.")
    parser.add_argument("file", help="Input .pgn.zst file.")
    parser.add_argument("--index", "-n", type=int, default=1, help="Which occurrence to extract (1=first, 2=second, etc.). Default is 1.")
    
    args = parser.parse_args()
    target_key = args.key
    filename = args.file
    target_index = args.index

    if target_index < 1:
        print("Error: Index must be >= 1.", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for occurrence #{target_index} of key '{target_key}' in {filename}...", file=sys.stderr)

    try:
        with open(filename, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                game_count = 0
                matches_found = 0
                
                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    
                    game_count += 1
                    if game_count % 1000 == 0:
                        print(f"\rGames scanned: {game_count}, Matches found: {matches_found}...", end="", file=sys.stderr)

                    board = game.board()
                    found_in_this_game = False
                    
                    # Check initial position
                    if build_key_for_side_to_move(board) == target_key:
                        found_in_this_game = True
                    
                    # Check moves
                    if not found_in_this_game:
                        for move in game.mainline_moves():
                            board.push(move)
                            if build_key_for_side_to_move(board) == target_key:
                                found_in_this_game = True
                                break 
                    
                    if found_in_this_game:
                        matches_found += 1
                        if matches_found == target_index:
                            print(f"\n\n--- FOUND Occurrence #{matches_found} in Game #{game_count} ---", file=sys.stderr)
                            print(game)
                            return

        print(f"\nEnd of file reached. Found {matches_found} matches total. Requested index {target_index} not found.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
