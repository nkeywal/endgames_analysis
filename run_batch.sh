#!/bin/bash

# Ensure output directory exists
mkdir -p out_stats

# Generator function to produce the task list
generate_tasks() {
  # (1) 1600-2100 for 2025-02 (First)
  echo "2025-02 1600 2100"

  # (2) Mixed 2100-2500 and 2500-5000 for all months
  # "mixé" ensures we don't do all high-elo then all low-elo, but interleave them.
  for m in 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12; do
    echo "$m 2100 2500"
    echo "$m 2500 5000"
  done

  # (1) 1600-2100 for 2025-10 (Last)
  echo "2025-10 1600 2100"
}

# Run with xargs -P 6 (6 parallel processes)
generate_tasks | xargs -n 3 -P 6 bash -c '
  m="$1"
  elo_min="$2"
  elo_max="$3"
  
  # Determine the correct input filename based on month
  # 2025-11 and 2025-12 have different naming convention in the listing
  if [[ "$m" > "2025-10" ]]; then
    pgn_file="lichess_raw_data/lichess_db_standard_rated_${m}.pgn.zst"
  else
    pgn_file="lichess_raw_data/filtered_standard_rated_${m}_elo1400plus.pgn.zst"
  fi

  if [ ! -f "$pgn_file" ]; then
    echo "Error: File not found: $pgn_file" >&2
    exit 1
  fi

  echo "Starting: $m Elo $elo_min-$elo_max using $pgn_file"
  
  # Use the virtual environment python explicitly
  zstd -dc "$pgn_file" \
    | .venv/bin/python3 endgame_stats.py \
      --pgn - \
      --month "$m" \
      --elo-min "$elo_min" \
      --elo-max "$elo_max" \
      --out-dir out_stats

' _
