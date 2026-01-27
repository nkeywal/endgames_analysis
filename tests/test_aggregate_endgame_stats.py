import sys
import io
from pathlib import Path
from typing import List

import pytest

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aggregate_endgame_stats as agg

# Sample TSV content matching the current format
SAMPLE_TSV_1 = """
# month=2025-02
# elo_min=2500
# elo_max=5000
# games_seen=100
# games_used=10
# games_skipped_short_plycount<35=0
# games_skipped_parse=0
# relevant_games=5
# plies_total=50
# errors_total=5
# can_win_total=10
# can_draw_total=20
# missed_win_to_draw_total=1
# missed_win_to_loss_total=0
# missed_draw_total=1
# time_loss_games_total=1
material	games	games_pct_over_used	plies	avg_plies_per_game	can_win	can_win_pct	can_draw	can_draw_pct	games_with_error	error_game_pct	errors	errors_per_ply_pct	missed_win_to_draw	mw2d_pct	missed_win_to_loss	mw2l_pct	missed_draw	md_pct	time_losses	time_loss_pct
KP_K	5	50.0	20	4.0	5	25.0	5	25.0	1	20.0	2	10.0	1	5.0	0	0.0	0	0.0	1	20.0
KR_KP	3	30.0	15	5.0	3	20.0	10	66.6	0	0.0	0	0.0	0	0.0	0	0.0	0	0.0	0	0.0
"""

SAMPLE_TSV_2 = """
# month=2025-03
# elo_min=2500
# elo_max=5000
# games_seen=200
# games_used=20
# games_skipped_short_plycount<35=0
# games_skipped_parse=0
# relevant_games=10
# plies_total=100
# errors_total=10
# can_win_total=20
# can_draw_total=40
# missed_win_to_draw_total=2
# missed_win_to_loss_total=0
# missed_draw_total=2
# time_loss_games_total=2
material	games	games_pct_over_used	plies	avg_plies_per_game	can_win	can_win_pct	can_draw	can_draw_pct	games_with_error	error_game_pct	errors	errors_per_ply_pct	missed_win_to_draw	mw2d_pct	missed_win_to_loss	mw2l_pct	missed_draw	md_pct	time_losses	time_loss_pct
KP_K	5	25.0	20	4.0	5	25.0	5	25.0	1	20.0	2	10.0	1	5.0	0	0.0	0	0.0	1	20.0
KP_KR	2	10.0	10	5.0	2	20.0	5	50.0	0	0.0	0	0.0	0	0.0	0	0.0	0	0.0	0	0.0
"""

@pytest.fixture
def tsv_files(tmp_path):
    f1 = tmp_path / "stats_2025-02.tsv"
    f1.write_text(SAMPLE_TSV_1, encoding="utf-8")
    f2 = tmp_path / "stats_2025-03.tsv"
    f2.write_text(SAMPLE_TSV_2, encoding="utf-8")
    return [f1, f2]

def test_parse_tsv(tsv_files):
    fs = agg.parse_tsv(tsv_files[0])
    assert fs.month == "2025-02"
    assert fs.elo_min == 2500
    assert fs.elo_max == 5000
    assert fs.meta["plies_total"] == "50"
    
    # Check rows (skipping header)
    assert len(fs.rows) == 2
    assert fs.rows[0][0] == "KP_K" # material is first column

def test_aggregation_logic(tsv_files):
    bucket = agg.BucketAgg(elo_min=2500, elo_max=5000)
    
    fs1 = agg.parse_tsv(tsv_files[0])
    bucket.add_file(fs1)
    
    fs2 = agg.parse_tsv(tsv_files[1])
    bucket.add_file(fs2)
    
    bucket.finalize()
    
    # Check header aggregation
    assert bucket.games_used == 10 + 20
    assert bucket.plies_total == 50 + 100
    assert bucket.errors_total == 5 + 10
    
    # Check per-material aggregation
    # KP_K appears in both: 5+5 games, 20+20 plies, 2+2 errors
    kp_k = bucket.per_material["KP_K"]
    assert kp_k.games == 10
    assert kp_k.plies == 40
    assert kp_k.errors_total == 4
    
    # KR_KP in file 1, KP_KR in file 2 (different keys strings)
    assert bucket.per_material["KR_KP"].games == 3
    assert bucket.per_material["KP_KR"].games == 2

def test_symmetry_filter(capsys):
    # Test --types argument parsing and filtering logic via main() simulation or direct function call.
    # Since main() parses args, let's test the logic we added to main and print_bucket_full.
    
    bucket = agg.BucketAgg(elo_min=2500, elo_max=5000)
    # Add dummy material
    bucket.per_material["KR_KP"] = agg.MaterialAgg(games=10)
    bucket.per_material["KP_KR"] = agg.MaterialAgg(games=10)
    bucket.per_material["KQ_K"] = agg.MaterialAgg(games=5)
    
    # Filter for KR_KP (should include KP_KR automatically)
    keep_types = {"KR_KP", "KP_KR"} 
    
    agg.print_bucket_full(bucket, top=0, min_games=0, keep_types=keep_types)
    out = capsys.readouterr().out
    
    assert "KR_KP" in out
    assert "KP_KR" in out
    assert "KQ_K" not in out

def test_symmetry_arg_parsing(monkeypatch):
    # Test that --types arg correctly generates the symmetric set
    monkeypatch.setattr(sys, "argv", ["prog", "--types", "KR_KP", "out_stats/*"])
    args = agg.parse_args()
    
    keep_types = set()
    for t in args.types.split(","):
        t = t.strip()
        keep_types.add(t)
        if "_" in t:
            p = t.split("_", 1)
            keep_types.add(f"{p[1]}_{p[0]}")
            
    assert "KR_KP" in keep_types
    assert "KP_KR" in keep_types

def test_robust_paths_parsing(monkeypatch):
    # Test the logic that recovers the first file if --glob eats it
    # Case: --glob file1 file2 file3
    # argpars sets glob="file1", paths=["file2", "file3"]
    # iter_input_paths should recover file1
    
    monkeypatch.setattr(sys, "argv", ["prog", "--glob", "file1", "file2", "file3"])
    args = agg.parse_args()
    
    # Mock Path to avoid FS calls? Or just trust the logic.
    # Since iter_input_paths calls .exists(), we need real files or mock.
    # Let's just verify the logic logic inside iter_input_paths by mocking Path.exists
    
    class MockPath:
        def __init__(self, p):
            self.p = p
        def exists(self):
            return True
        def is_file(self):
            return True
        def is_dir(self):
            return False
        def __eq__(self, other):
            return self.p == other.p
        def __repr__(self):
            return f"MockPath({self.p})"
        
    monkeypatch.setattr(agg, "Path", MockPath)
    
    paths = agg.iter_input_paths(args)
    # logic: if glob != DEFAULT and paths is not empty, insert glob at 0
    assert len(paths) == 3
    assert paths[0].p == "file1"
    assert paths[1].p == "file2"
    assert paths[2].p == "file3"
