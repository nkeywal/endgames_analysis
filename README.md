# Endgame Statistics Analyzer

This toolkit analyzes chess endgames from large PGN databases (specifically Lichess dumps). It filters games, analyzes endgame play accuracy using **Gaviota Tablebases**, and aggregates statistics by Elo buckets and material configurations.

## Methodology & Concepts

### 1. Evaluation Logic (WDL vs DTM)
The analysis is based strictly on **Tablebase WDL (Win/Draw/Loss)** data, not Distance-to-Mate (DTM).
* **Correct Move:** A move is considered "good" if it preserves the best theoretical result, without taking into account the distance to mate (DTM). For example, in a winning position, any move that keeps the position winning is considered correct, even if it delays checkmate by 50 moves compared to the fastest mate.
* **Error:** An error is recorded **only** if a move downgrades the theoretical result (e.g., `Win` $\to$ `Draw`, or `Draw` $\to$ `Loss`).

### 2. Material Key Format
Endgames are classified by a simplified string representing the material on the board, oriented by the **Side-to-Move**.
* **Format:** `[SideToMove]_[Opponent]`
* **Piece Order:** `K` (King), `Q` (Queen), `R` (Rook), `B` (Bishop), `N` (Knight), `P` (Pawn).
* **Example:** `KP_K` means it is the side with the King and Pawn's turn to move against a lone King.

### 3. Bishop Handling ("B" vs "D")
To distinguish between strategically distinct bishop endgames, the tool uses specific tokens:
* **`B`**: Represents a bishop in standard contexts (e.g., `KB_K`).
* **`D` (Different/Opposite Colors)**: If both sides have exactly one bishop and they are on **opposite colors**, the token `D` is used instead of `B`.
    * *Example:* `KD_KD` denotes an Opposite-Colored Bishop endgame.
    * *Example:* `KB_KB` denotes a Same-Colored Bishop endgame.

### 4. Analysis Scope
* **Piece Count:** Only positions with **3, 4, or 5 pieces** (total) are analyzed.
* **Exclusions:** Trivial wins (e.g., `KQ_K`, `KR_K`) and positions where one side has a bare King against 3+ units are excluded to focus on non-trivial conversions.

---

## Scripts Overview

The toolkit is composed of four main scripts:

* **`filter_pgn.py`**
    * **Purpose:** Efficiently streams and filters massive PGN dumps (e.g., 50GB+ compressed).
    * **Logic:** Keeps only **Standard**, **Rated**, **Blitz** games where **both players** meet a specified Elo threshold. It handles stream buffering to ensure no game data is lost during processing.

* **`endgame_stats.py`**
    * **Purpose:** The core analysis engine.
    * **Logic:** Reads filtered games and probes Gaviota Tablebases for every position with 3, 4, or 5 pieces. It tracks:
        * **Errors:** When a player blunders a Win to a Draw/Loss, or a Draw to a Loss.
        * **Time:** Games lost on time while in a tracked endgame.
    * **Output:** Generates a TSV file containing raw counts for each material key (e.g., `KP_K`, `KR_KP`).

* **`aggregate_endgame_stats.py`**
    * **Purpose:** Reporting and data aggregation.
    * **Logic:** Consumes multiple TSV files (e.g., from different months), sums the raw counts, and calculates derived metrics (Error Rate %, Time Loss %, etc.). It outputs a formatted table sorted by frequency or material.

* **`extract_by_key.py`**
    * **Purpose:** A utility tool for debugging and inspection.
    * **Logic:** Scans a compressed PGN file to find and extract the N-th game containing a specific material key (e.g., "Find the first game featuring `KD_KD`").

---

## Prerequisites

### 1. Python Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
