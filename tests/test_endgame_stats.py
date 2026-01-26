import io
import sys
from pathlib import Path

import pytest
import chess
import chess.pgn

# Ensure project root (parent of tests/) is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import endgame_stats as es  # noqa: E402


def fast_ply_count(movetext: str) -> int:
    try:
        return es._fast_ply_count_from_movetext(movetext)
    except Exception:
        return es._fast_ply_count_from_movetext([movetext])


class FakeTablebase:
    """
    Deterministic fake Gaviota-like tablebase.

    Map (board.board_fen(), board.turn) -> desired WDL from White perspective (-1/0/+1).
    probe_dtm returns only a sign-relevant integer (DTM sign).
    """
    def __init__(self, wdl_white_by_key):
        self.wdl_white_by_key = dict(wdl_white_by_key)

    def probe_dtm(self, board: chess.Board) -> int:
        key = (board.board_fen(), board.turn)
        wdl_white = self.wdl_white_by_key.get(key, 0)
        if wdl_white == 0:
            return 0
        # wdl_white = wdl_stm if stm white else -wdl_stm
        wdl_stm = wdl_white if board.turn == chess.WHITE else -wdl_white
        return 1 if wdl_stm > 0 else -1


def make_game_from_pgn(pgn: str) -> chess.pgn.Game:
    g = chess.pgn.read_game(io.StringIO(pgn))
    assert g is not None
    return g


def test_read_games_raw_yields_multiple_games():
    pgn = """\
[Event "Rated Blitz game"]
[Site "https://lichess.org/aaa"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 e5 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/bbb"]
[Result "0-1"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "180+0"]
[Termination "Normal"]

1. d4 d5 0-1
"""
    games = list(es.read_games_raw(io.StringIO(pgn)))
    assert len(games) == 2
    h1, ply1, raw1 = games[0]
    h2, ply2, raw2 = games[1]
    assert h1["Site"] == "https://lichess.org/aaa"
    assert h2["Site"] == "https://lichess.org/bbb"
    assert ply1 == 2
    assert ply2 == 2
    assert raw1.strip().startswith("[Event")
    assert "1. e4" in raw1
    assert "1. d4" in raw2


def test_fast_ply_count_ignores_move_numbers_results_nags_and_single_token_comments():
    # Use a single-token comment "{c}" to match the intended fast-skip behavior.
    movetext = "1. e4 {c} e5 $1 2. Nf3 Nc6 1-0"
    # Tokens counted: e4, e5, Nf3, Nc6 => 4
    assert fast_ply_count(movetext) == 4


def test_in_bucket():
    assert es.in_bucket(1700, 1750, 1600, 2100)
    assert not es.in_bucket(1200, 1300, 1600, 2100)
    assert not es.in_bucket(1600, 2400, 1600, 2100)


def test_build_key_opposite_colored_bishops_normalizes_to_D_when_not_insufficient():
    # Add a pawn to avoid python-chess classifying the position as insufficient material.
    # White: K + B(dark) + P, Black: K + B(light). Total pieces = 5.
    board = chess.Board("8/5b2/8/8/8/8/P7/2B1K2k w - - 0 1")
    assert es.total_pieces(board) == 5
    assert es.build_key_for_side_to_move(board) == "KDP_KD"


def test_build_key_same_color_bishops_keeps_B_when_not_insufficient():
    # Add a pawn to avoid insufficient material. Both bishops on dark squares.
    board = chess.Board("8/8/8/8/8/4b3/P7/2B1K2k w - - 0 1")
    assert es.total_pieces(board) == 5
    assert es.build_key_for_side_to_move(board) == "KBP_KB"


def test_two_bishops_same_color_excluded():
    # Synthetic: two bishops on same color is excluded (may also be insufficient, which is fine).
    board = chess.Board("8/8/8/8/8/4B3/8/2B1K2k w - - 0 1")
    assert es.total_pieces(board) == 4
    assert es.build_key_for_side_to_move(board) is None


def test_trivial_exclusions_kr_vs_k_excluded_kbn_vs_k_included():
    board_krk = chess.Board("8/8/8/8/8/8/8/R3K2k w - - 0 1")
    assert es.total_pieces(board_krk) == 3
    assert es.build_key_for_side_to_move(board_krk) is None

    board_kbnk = chess.Board("8/8/8/8/8/8/8/2BNK2k w - - 0 1")
    assert es.total_pieces(board_kbnk) == 4
    assert es.build_key_for_side_to_move(board_kbnk) == "KBN_K"


def test_castling_rights_in_5_piece_endgame_logs(capsys):
    fen = "1n2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"
    pgn = f"""\
[Event "Rated Blitz game"]
[Site "https://lichess.org/We7tFUS8"]
[Result "*"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "180+0"]
[Termination "Normal"]
[SetUp "1"]
[FEN "{fen}"]

1. Ra2 *
"""
    game = make_game_from_pgn(pgn)
    headers = dict(game.headers)
    tb = FakeTablebase({})
    es.analyze_game(game, headers, tb)
    err = capsys.readouterr().err
    assert "still being allowed to castle" in err
    assert '[Site "https://lichess.org/We7tFUS8"]' in err
    assert "zeroing castling rights" in err


def test_wdl_improvement_detection_aborts_and_logs(capsys):
    fen0 = "8/8/8/8/8/8/4P3/4K2k w - - 0 1"
    b0 = chess.Board(fen0)
    move = chess.Move.from_uci("e2e4")
    b1 = b0.copy(stack=False)
    b1.push(move)

    tb = FakeTablebase({
        (b0.board_fen(), b0.turn): -1,
        (b1.board_fen(), b1.turn): 0,
    })

    pgn = f"""\
[Event "Rated Blitz game"]
[Site "https://lichess.org/improve"]
[Result "*"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "180+0"]
[Termination "Normal"]
[SetUp "1"]
[FEN "{fen0}"]

1. e4 *
"""
    game = make_game_from_pgn(pgn)
    headers = dict(game.headers)

    with pytest.raises(RuntimeError, match="WDL improved"):
        es.analyze_game(game, headers, tb)

    err = capsys.readouterr().err
    assert "WDL improvement detected" in err
    assert '[Site "https://lichess.org/improve"]' in err
    assert "fen_before=" in err
    assert "fen_after" in err


def test_error_count_on_wdl_drop():
    fen0 = "8/8/8/8/8/8/4P3/4K2k w - - 0 1"
    b0 = chess.Board(fen0)
    move = chess.Move.from_uci("e2e4")
    b1 = b0.copy(stack=False)
    b1.push(move)

    tb = FakeTablebase({
        (b0.board_fen(), b0.turn): 1,
        (b1.board_fen(), b1.turn): 0,
    })

    pgn = f"""\
[Event "Rated Blitz game"]
[Site "https://lichess.org/error"]
[Result "*"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "180+0"]
[Termination "Normal"]
[SetUp "1"]
[FEN "{fen0}"]

1. e4 *
"""
    game = make_game_from_pgn(pgn)
    headers = dict(game.headers)

    deltas = es.analyze_game(game, headers, tb)
    assert len(deltas.per_key_plies) == 1
    key = next(iter(deltas.per_key_plies.keys()))
    assert deltas.per_key_plies[key] == 1
    assert deltas.per_key_errors[key] == 1
    assert key in deltas.keys_with_error
    assert key in deltas.keys_seen


def test_time_forfeit_attribution_to_loser_type():
    fen0 = "8/8/8/8/8/8/4P3/4K2k w - - 0 1"
    pgn = f"""\
[Event "Rated Blitz game"]
[Site "https://lichess.org/time"]
[Result "0-1"]
[WhiteElo "2000"]
[BlackElo "2000"]
[TimeControl "180+0"]
[Termination "Time forfeit"]
[SetUp "1"]
[FEN "{fen0}"]

1. e4 0-1
"""
    game = make_game_from_pgn(pgn)
    headers = dict(game.headers)
    tb = FakeTablebase({})

    deltas = es.analyze_game(game, headers, tb)
    assert deltas.time_loss_key is not None
    assert deltas.time_loss_key in deltas.keys_seen


def test_write_tsv_skips_zero_game_rows(tmp_path):
    out = tmp_path / "stats.tsv"
    s = es.Stats(games_seen=10, games_used=10)
    per_key_games = {"KQ_K": 0, "KP_K": 2}
    per_key_games_with_error = {"KP_K": 1}
    per_key_plies_total = {"KP_K": 5}
    per_key_errors_total = {"KP_K": 2}
    per_key_time_losses = {"KP_K": 0}
    
    # New metrics dicts
    per_key_can_win_total = {"KP_K": 1}
    per_key_can_draw_total = {"KP_K": 1}
    per_key_missed_win_to_draw_total = {"KP_K": 0}
    per_key_missed_win_to_loss_total = {"KP_K": 0}
    per_key_missed_draw_total = {"KP_K": 0}
    per_key_time_draws = {"KP_K": 0}

    es.write_tsv(
        out_path=out,
        month="2025-02",
        elo_min=2500,
        elo_max=5000,
        s=s,
        per_key_games=per_key_games,
        per_key_games_with_error=per_key_games_with_error,
        per_key_plies_total=per_key_plies_total,
        per_key_errors_total=per_key_errors_total,
        per_key_can_win_total=per_key_can_win_total,
        per_key_can_draw_total=per_key_can_draw_total,
        per_key_missed_win_to_draw_total=per_key_missed_win_to_draw_total,
        per_key_missed_win_to_loss_total=per_key_missed_win_to_loss_total,
        per_key_missed_draw_total=per_key_missed_draw_total,
        per_key_time_losses=per_key_time_losses,
        per_key_time_draws=per_key_time_draws,
    )

    txt = out.read_text(encoding="utf-8")
    assert "KP_K" in txt
    assert "KQ_K" not in txt
    assert "can_win" in txt
    assert "missed_win_to_loss" in txt
    assert "missed_win_to_loss_pct_over_can_win" in txt


def test_analyze_game_counts_missed_opportunities():
    # Setup: 
    # Ply 1: White to move. Win (1). Moves to Draw (0). -> Missed Win to Draw.
    # Ply 2: Black to move. Draw (0). Moves to Loss (-1). -> Missed Draw (Blunder).
    
    fen0 = "8/8/8/8/8/8/4P3/4K2k w - - 0 1" # White to move
    b0 = chess.Board(fen0)
    move1 = chess.Move.from_uci("e2e4") # White blunders win to draw
    
    b1 = b0.copy(stack=False)
    b1.push(move1) # Now Black to move
    move2 = chess.Move.from_uci("h1h2") # Black blunders draw to loss
    
    b2 = b1.copy(stack=False)
    b2.push(move2) # Result

    # Fake TB:
    # b0 (White to move): Win for White (+1). Mover=White. Mover WDL = +1.
    # b1 (Black to move): Draw (0). Mover=Black. Mover WDL = 0.
    # b2 (White to move): Win for White (+1). 
    # Wait, if b1->b2 makes b2 WinForWhite, then for Black (mover at b1), b2 is Loss (-1).
    
    tb = FakeTablebase({
        (b0.board_fen(), b0.turn): 1,  # White winning
        (b1.board_fen(), b1.turn): 0,  # Draw
        (b2.board_fen(), b2.turn): 1,  # White winning (so Black lost)
    })

    pgn = f"""\
[Event "Test"]
[Site "https://lichess.org/missed"]
[Result "*"]
[SetUp "1"]
[FEN "{fen0}"]

1. e4 Kh2 *
"""
    game = make_game_from_pgn(pgn)
    headers = dict(game.headers)

    deltas = es.analyze_game(game, headers, tb)
    
    # Ply 1 (White): +1 -> 0. Missed Win to Draw.
    # Key should be KP_K (White has P, Black has K)
    key1 = "KP_K" 
    
    # Ply 2 (Black): 0 -> -1. Missed Draw.
    # Key should be K_KP (Black has K, White has P)
    key2 = "K_KP"

    # Verify counts
    assert deltas.per_key_plies[key1] == 1
    assert deltas.per_key_can_win[key1] == 1
    assert deltas.per_key_missed_win_to_draw[key1] == 1
    assert deltas.per_key_errors[key1] == 1
    
    assert deltas.per_key_plies[key2] == 1
    assert deltas.per_key_can_draw[key2] == 1
    assert deltas.per_key_missed_draw[key2] == 1
    assert deltas.per_key_errors[key2] == 1


def test_parse_args_default_increment_and_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["endgame_stats.py", "--pgn", "-", "--month", "2025-02", "--elo-min", "1600", "--elo-max", "2100"],
    )
    args = es.parse_args()
    assert args.increment == "all"

    monkeypatch.setattr(
        sys,
        "argv",
        ["endgame_stats.py", "--pgn", "-", "--month", "2025-02", "--elo-min", "1600", "--elo-max", "2100", "--increment", "yes"],
    )
    args = es.parse_args()
    assert args.increment == "yes"
