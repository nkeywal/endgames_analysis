#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <increment_mode> (all|yes|no)"
  exit 1
fi

export INCREMENT_MODE="$1"

mkdir -p out_stats

generate_tasks() {
  # 1700-2100 for specific months (First)
  echo "2025-02 1700 2100"
  echo "2025-10 1700 2100"

  # Mixed 2100-2500 and 2500-5000 for all months
  for m in 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12; do
    echo "$m 2100 2500"
    echo "$m 2500 5000"
  done
}

generate_tasks | xargs -n 3 -P 6 bash -c '
  m="$1"
  elo_min="$2"
  elo_max="$3"
  inc="$INCREMENT_MODE"
  
  pgn_file="lichess_raw_data/filtered_standard_rated_${m}_elo1400plus_blitz.pgn.zst"

  if [ ! -f "$pgn_file" ]; then
    echo "Error: File not found: $pgn_file" >&2
    exit 1
  fi

  echo "Starting: $m Elo $elo_min-$elo_max Inc=$inc using $pgn_file"
  
  zstd -dc "$pgn_file" \
    | .venv/bin/python3 endgame_stats.py \
      --pgn - \
      --month "$m" \
      --elo-min "$elo_min" \
      --elo-max "$elo_max" \
      --increment "$inc" \
      --out-dir out_stats

' _
