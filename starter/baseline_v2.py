"""Baseline v2: Contextual Heuristic, Spatial Hysteresis & Smart Tagging Fusion.
Implements the foundations from Phase 1 EDA and Borghesi et al. (2023).
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import load_smart_tagging as lst
from games import require, GAMES, GAME

# Constants
HOME, AWAY, DEAD, NOBODY = 0, 1, 2, -1
LABEL_MAP = {HOME: "H", AWAY: "A", DEAD: "D", NOBODY: "X"}

PITCH_LEN, PITCH_WID = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]

# Match structure constants
HALF_1_KICKOFF = 11185
HALF_1_END = 90651
HALF_2_START = 110808
HALF_2_KICKOFF = 113742
HALF_2_END = 190197

# Tuning parameters
MAX_GAP_ALIVE = 50        # Forward-fill gaps <= 2.0s during open play
STEAL_RADIUS = 2.5        # Max distance (m) to consider ball in controlled possession
STEAL_FRAMES = 8          # Consecutive frames rival must control ball to switch possession
MIN_SPELL_FRAMES = 50     # Debounce: min duration (2.0s) for a possession spell


def get_team_mapping(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if s == home_id else AWAY for s in slots], dtype=np.int8)


def load_set_pieces():
    """Load all SET PIECES from Smart Tagging XML."""
    events = lst.load()
    set_pieces = []
    for ev in events:
        if ev["code"].strip().upper() == "SET PIECES":
            team_str = ev["tags"].get("Team", "N/A")
            team = None
            if "barcelona" in team_str.lower():
                team = HOME
            elif "atletic" in team_str.lower() or "athletic" in team_str.lower():
                team = AWAY

            stype = ev["tags"].get("Type", ev["tags"].get("type", "N/A"))
            set_pieces.append({
                "start_frame": ev["start_frame"],
                "end_frame": ev["end_frame"],
                "team": team,
                "type": stype
            })
    return set_pieces


def build_predictions():
    print("=" * 70)
    print("Building Baseline v2 Predictions")
    print("=" * 70)

    paths = require(["atd_raw", "atd_meta"])
    raw_path, meta_path = paths[0], paths[1]

    team_of_track = get_team_mapping(meta_path)
    n_tracks = len(team_of_track)
    print(f"Loaded {n_tracks} track slots from {os.path.basename(meta_path)}.")

    set_pieces = load_set_pieces()
    print(f"Loaded {len(set_pieces)} SET PIECES from Smart Tagging.")

    # 1. Initialize frame predictions with NOBODY (-1 -> 'X')
    preds = np.full(N_FRAMES, NOBODY, dtype=np.int8)

    # 2. Known dead-ball stretches from match structure:
    # Between second half start and second half kickoff (graded frames that are all D)
    preds[HALF_2_START:HALF_2_KICKOFF] = DEAD

    # 3. Stream tracking data and extract nearest distances and ball presence
    print("Streaming ATD tracking data...")
    d_home = np.full(N_FRAMES, np.nan, dtype=np.float32)
    d_away = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_out = np.zeros(N_FRAMES, dtype=bool)
    ball_seen = np.zeros(N_FRAMES, dtype=bool)

    with open(raw_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame_idx = int(parts[0]) - 1
            if frame_idx >= N_FRAMES:
                continue

            ball = parts[2]
            if ball.startswith("N"):
                continue

            ball_seen[frame_idx] = True
            b_coords = ball.split(",")
            bx_norm = float(b_coords[0])
            by_norm = float(b_coords[1])

            # Check if physically outside pitch boundaries
            if bx_norm < -0.01 or bx_norm > 1.01 or by_norm < -0.01 or by_norm > 1.01:
                ball_out[frame_idx] = True

            bx = bx_norm * PITCH_LEN
            by = by_norm * PITCH_WID

            players = parts[1].split(";")
            h_dists = []
            a_dists = []

            for slot_idx, p_str in enumerate(players):
                if p_str.startswith("N") or slot_idx >= n_tracks:
                    continue
                p_coords = p_str.split(",")
                px = float(p_coords[0]) * PITCH_LEN
                py = float(p_coords[1]) * PITCH_WID
                dist = np.hypot(px - bx, py - by)
                if team_of_track[slot_idx] == HOME:
                    h_dists.append(dist)
                else:
                    a_dists.append(dist)

            if h_dists:
                d_home[frame_idx] = min(h_dists)
            if a_dists:
                d_away[frame_idx] = min(a_dists)

    print("Tracking stream processed.")

    # 4. State tracking with Hysteresis & Persistence for each half
    halves_ranges = [
        (HALF_1_KICKOFF, HALF_1_END),
        (HALF_2_KICKOFF, HALF_2_END)
    ]

    for start_f, end_f in halves_ranges:
        current_state = NOBODY
        steal_candidate = NOBODY
        steal_count = 0
        missing_count = 0

        for f in range(start_f, end_f + 1):
            if ball_out[f]:
                # Ball physically out of bounds -> DEAD
                current_state = DEAD
                steal_candidate = NOBODY
                steal_count = 0
                missing_count = 0
                preds[f] = DEAD
                continue

            if not ball_seen[f]:
                missing_count += 1
                if missing_count <= MAX_GAP_ALIVE and current_state in [HOME, AWAY]:
                    # Forward-fill: persist possessing team during short occlusions
                    preds[f] = current_state
                elif missing_count > 125:  # > 5s without ball
                    current_state = DEAD
                    preds[f] = DEAD
                else:
                    preds[f] = current_state if current_state != NOBODY else DEAD
                continue

            # Ball is seen inside pitch
            missing_count = 0
            dh = d_home[f]
            da = d_away[f]

            if np.isnan(dh) and np.isnan(da):
                preds[f] = current_state if current_state != NOBODY else DEAD
                continue

            # Determine nearest team
            instant_nearest = HOME if (np.isnan(da) or dh < da) else AWAY
            instant_min_dist = dh if instant_nearest == HOME else da

            if current_state == NOBODY or current_state == DEAD:
                # Coming from restart or stoppage -> take nearest player if in reasonable radius
                current_state = instant_nearest
                steal_candidate = NOBODY
                steal_count = 0
            elif current_state == instant_nearest:
                # Same team in possession -> reset steal tracker
                steal_candidate = NOBODY
                steal_count = 0
            else:
                # Rival is closer! Check hysteresis / controlled possession
                if instant_min_dist <= STEAL_RADIUS:
                    if steal_candidate == instant_nearest:
                        steal_count += 1
                    else:
                        steal_candidate = instant_nearest
                        steal_count = 1

                    if steal_count >= STEAL_FRAMES:
                        # Turnover confirmed!
                        current_state = steal_candidate
                        steal_candidate = NOBODY
                        steal_count = 0
                else:
                    # Rival is closer but far (> 2.5m) -> ball in flight / pass -> maintain possession
                    steal_candidate = NOBODY
                    steal_count = 0

            preds[f] = current_state

    # 5. Ingest SET PIECES to reinforce DEAD ball prior
    print("Applying SET PIECES dead-ball windows...")
    for sp in set_pieces:
        sp_start = sp["start_frame"]
        sp_end = sp["end_frame"]
        stype = sp["type"].lower()

        if "kick off (start)" in stype:
            continue

        # Look for stoppage window preceding or around the set piece
        # A stoppage typically lasts 150-350 frames before restart
        pre_window = 200
        w_start = max(0, sp_start - pre_window)
        w_end = min(N_FRAMES - 1, sp_start + 25)

        # Apply DEAD in window where ball was not in active controlled possession
        for f in range(w_start, w_end):
            if preds[f] in [HOME, AWAY]:
                # If ball is missing or far from players, mark DEAD
                if not ball_seen[f] or ball_out[f] or (d_home[f] > 3.0 and d_away[f] > 3.0):
                    preds[f] = DEAD

        # At the restart frame, assign possession to restarting team if known
        if sp["team"] in [HOME, AWAY]:
            restart_f = sp_start
            if restart_f < N_FRAMES and preds[restart_f] != NOBODY:
                preds[restart_f:min(N_FRAMES, restart_f + 25)] = sp["team"]

    # 6. Debounce / Post-processing: Remove isolated micro-spells < MIN_SPELL_FRAMES
    print("Applying Debounce / minimum possession duration...")
    labels_arr = preds.copy()

    # Find contiguous blocks
    change_idx = np.where(labels_arr[:-1] != labels_arr[1:])[0] + 1
    splits = np.concatenate(([0], change_idx, [len(labels_arr)]))

    for i in range(len(splits) - 1):
        s = splits[i]
        e = splits[i + 1]
        length = e - s
        lbl = labels_arr[s]

        # If a possession spell is < 1s (25 frames), absorb it into neighbors
        if lbl in [HOME, AWAY] and length < MIN_SPELL_FRAMES:
            prev_lbl = labels_arr[s - 1] if s > 0 else NOBODY
            next_lbl = labels_arr[e] if e < len(labels_arr) else NOBODY
            if prev_lbl == next_lbl and prev_lbl != NOBODY:
                labels_arr[s:e] = prev_lbl
            elif prev_lbl in [HOME, AWAY]:
                labels_arr[s:e] = prev_lbl
            elif next_lbl in [HOME, AWAY]:
                labels_arr[s:e] = next_lbl

    preds = labels_arr

    # 7. Write submission CSV
    out_labels = [LABEL_MAP.get(p, "X") for p in preds]
    out_path = os.path.join(ROOT, "evaluation", "results", "baseline_v2.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": out_labels})
    df_out.to_csv(out_path, index=False)
    print(f"\n[OK] Submission written to: {out_path}")
    print(f"Total rows: {len(df_out):,}")
    print("Label distribution in submission:")
    print(df_out["label"].value_counts().to_string())

    return out_path


if __name__ == "__main__":
    build_predictions()
