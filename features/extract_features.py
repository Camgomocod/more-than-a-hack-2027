import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import load_smart_tagging as lst

from games import GAME, GAMES, require

FEATURES_NPZ_PATH = os.path.join(HERE, "features.npz")

FPS = 25.0
DT = 1.0 / FPS
PITCH_LEN, PITCH_WID = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]

HOME, AWAY, NOBODY = 0, 1, -1

def get_team_mapping(meta_path):
    """Extract player slot to team affiliation mapping from ATD metadata.

    Args:
        meta_path (str): File path to the ATD tracking metadata XML.

    Returns:
        np.ndarray: 1D array mapping each player slot to HOME (0) or AWAY (1).
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if s == home_id else AWAY for s in slots], dtype=np.int8)

def extract_smart_tagging_features():
    """Extract tactical context priors and set-piece proximity features.

    Returns:
        dict: Dictionary of temporal distance arrays and phase indicator arrays.
    """
    print("Parsing Smart Tagging contextual priors...")
    events = lst.load()

    frames_to_next_setpiece = np.full(N_FRAMES, 500, dtype=np.int16)
    frames_since_last_setpiece = np.full(N_FRAMES, 500, dtype=np.int16)
    setpiece_team = np.zeros(N_FRAMES, dtype=np.int8)

    in_open_play_phase = np.zeros(N_FRAMES, dtype=np.int8)
    open_play_team = np.zeros(N_FRAMES, dtype=np.int8)

    set_pieces = []
    for ev in events:
        code = ev["code"].strip().upper()
        t_str = ev["tags"].get("Team", "N/A").lower()
        team_code = 0
        if "barcelona" in t_str:
            team_code = 1
        elif "atletic" in t_str or "athletic" in t_str:
            team_code = 2

        if code == "SET PIECES":
            set_pieces.append((ev["start_frame"], ev["end_frame"], team_code))
        elif code in ["BUILD UP", "ATTACKING TRANSITION", "PROGRESSION"]:
            sf = max(0, ev["start_frame"])
            ef = min(N_FRAMES, ev["end_frame"])
            in_open_play_phase[sf:ef] = 1
            if team_code > 0:
                open_play_team[sf:ef] = team_code

    set_pieces.sort(key=lambda item: item[0])

    sp_starts = np.array([sp[0] for sp in set_pieces])
    sp_teams = np.array([sp[2] for sp in set_pieces])

    if len(sp_starts) > 0:
        for f in range(N_FRAMES):
            future_idx = np.where(sp_starts >= f)[0]
            if len(future_idx) > 0:
                next_idx = future_idx[0]
                diff = sp_starts[next_idx] - f
                frames_to_next_setpiece[f] = min(500, diff)
                setpiece_team[f] = sp_teams[next_idx]

            past_idx = np.where(sp_starts <= f)[0]
            if len(past_idx) > 0:
                prev_idx = past_idx[-1]
                diff = f - sp_starts[prev_idx]
                frames_since_last_setpiece[f] = min(500, diff)

    return {
        "frames_to_next_setpiece": frames_to_next_setpiece,
        "frames_since_last_setpiece": frames_since_last_setpiece,
        "setpiece_team": setpiece_team,
        "in_open_play_phase": in_open_play_phase,
        "open_play_team": open_play_team,
    }

def run_feature_extraction():
    """Execute kinematic, spatial, and contextual feature extraction across all frames.

    Extracts 19 time-synchronized features and saves them to a compressed NPZ archive.
    """
    print("=" * 70)
    print("Fase 3: Feature Extraction Pipeline")
    print("=" * 70)

    paths = require(["atd_raw", "atd_meta"])
    raw_path, meta_path = paths[0], paths[1]

    team_of_track = get_team_mapping(meta_path)
    n_tracks = len(team_of_track)
    print(f"Loaded {n_tracks} track slots. (Home: {(team_of_track==HOME).sum()}, Away: {(team_of_track==AWAY).sum()})")

    st_feats = extract_smart_tagging_features()

    print("Allocating memory for 199,319 frames...")
    is_ball_missing = np.zeros(N_FRAMES, dtype=np.int8)
    missing_streak_frames = np.zeros(N_FRAMES, dtype=np.int16)
    ball_out_of_pitch = np.zeros(N_FRAMES, dtype=np.int8)
    dist_to_boundary_m = np.full(N_FRAMES, 50.0, dtype=np.float32)

    ball_speed_mps = np.zeros(N_FRAMES, dtype=np.float32)
    ball_accel_mps2 = np.zeros(N_FRAMES, dtype=np.float32)

    d_nearest_home_m = np.full(N_FRAMES, 50.0, dtype=np.float32)
    d_nearest_away_m = np.full(N_FRAMES, 50.0, dtype=np.float32)
    delta_d_m = np.zeros(N_FRAMES, dtype=np.float32)

    k5_home_mean_dist_m = np.full(N_FRAMES, 50.0, dtype=np.float32)
    k5_away_mean_dist_m = np.full(N_FRAMES, 50.0, dtype=np.float32)
    k5_home_ratio = np.full(N_FRAMES, 0.5, dtype=np.float32)

    centroid_5nn_x_m = np.full(N_FRAMES, PITCH_LEN / 2.0, dtype=np.float32)
    centroid_5nn_y_m = np.full(N_FRAMES, PITCH_WID / 2.0, dtype=np.float32)

    print(f"Streaming ATD raw data from {os.path.basename(raw_path)} ...")
    last_valid_bx = None
    last_valid_by = None
    last_speed = 0.0
    streak = 0

    with open(raw_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            f_idx = int(parts[0]) - 1
            if f_idx >= N_FRAMES:
                continue

            ball = parts[2]
            if ball.startswith("N"):
                is_ball_missing[f_idx] = 1
                streak += 1
                missing_streak_frames[f_idx] = min(32767, streak)

                if f_idx > 0:
                    d_nearest_home_m[f_idx] = d_nearest_home_m[f_idx - 1]
                    d_nearest_away_m[f_idx] = d_nearest_away_m[f_idx - 1]
                    delta_d_m[f_idx] = delta_d_m[f_idx - 1]
                    k5_home_ratio[f_idx] = k5_home_ratio[f_idx - 1]
                continue

            streak = 0
            is_ball_missing[f_idx] = 0
            missing_streak_frames[f_idx] = 0

            b_coords = ball.split(",")
            bx_norm = float(b_coords[0])
            by_norm = float(b_coords[1])

            bx = bx_norm * PITCH_LEN
            by = by_norm * PITCH_WID

            if bx_norm < 0.0 or bx_norm > 1.0 or by_norm < 0.0 or by_norm > 1.0:
                ball_out_of_pitch[f_idx] = 1

            d_x = min(max(0.0, bx), max(0.0, PITCH_LEN - bx))
            d_y = min(max(0.0, by), max(0.0, PITCH_WID - by))
            dist_to_boundary_m[f_idx] = min(d_x, d_y)

            curr_speed = 0.0
            if len(b_coords) > 2 and not b_coords[2].startswith("N"):
                curr_speed = float(b_coords[2])
            elif last_valid_bx is not None:
                dx = bx - last_valid_bx
                dy = by - last_valid_by
                curr_speed = np.hypot(dx, dy) / DT

            ball_speed_mps[f_idx] = curr_speed
            ball_accel_mps2[f_idx] = (curr_speed - last_speed) / DT

            last_speed = curr_speed
            last_valid_bx = bx
            last_valid_by = by

            players = parts[1].split(";")
            h_dists = []
            a_dists = []
            all_players_pos = []

            for slot_idx, p_str in enumerate(players):
                if p_str.startswith("N") or slot_idx >= n_tracks:
                    continue
                p_coords = p_str.split(",")
                px = float(p_coords[0]) * PITCH_LEN
                py = float(p_coords[1]) * PITCH_WID
                dist = np.hypot(px - bx, py - by)
                p_team = team_of_track[slot_idx]
                if p_team == HOME:
                    h_dists.append(dist)
                else:
                    a_dists.append(dist)
                all_players_pos.append((dist, p_team, px, py))

            if h_dists:
                h_sorted = sorted(h_dists)
                d_nearest_home_m[f_idx] = h_sorted[0]
                k5_home_mean_dist_m[f_idx] = np.mean(h_sorted[:min(5, len(h_sorted))])

            if a_dists:
                a_sorted = sorted(a_dists)
                d_nearest_away_m[f_idx] = a_sorted[0]
                k5_away_mean_dist_m[f_idx] = np.mean(a_sorted[:min(5, len(a_sorted))])

            delta_d_m[f_idx] = d_nearest_home_m[f_idx] - d_nearest_away_m[f_idx]

            if all_players_pos:
                all_players_pos.sort(key=lambda item: item[0])
                top5 = all_players_pos[:min(5, len(all_players_pos))]
                home_in_top5 = sum(1 for item in top5 if item[1] == HOME)
                k5_home_ratio[f_idx] = home_in_top5 / float(len(top5))
                centroid_5nn_x_m[f_idx] = np.mean([item[2] for item in top5])
                centroid_5nn_y_m[f_idx] = np.mean([item[3] for item in top5])

    print("Stream processing completed successfully.")

    print(f"Saving feature matrix to {FEATURES_NPZ_PATH} ...")
    feature_dict = {

        "is_ball_missing": is_ball_missing,
        "missing_streak_frames": missing_streak_frames,
        "ball_speed_mps": ball_speed_mps,
        "ball_accel_mps2": ball_accel_mps2,
        "ball_out_of_pitch": ball_out_of_pitch,
        "dist_to_boundary_m": dist_to_boundary_m,

        "d_nearest_home_m": d_nearest_home_m,
        "d_nearest_away_m": d_nearest_away_m,
        "delta_d_m": delta_d_m,
        "k5_home_mean_dist_m": k5_home_mean_dist_m,
        "k5_away_mean_dist_m": k5_away_mean_dist_m,
        "k5_home_ratio": k5_home_ratio,
        "centroid_5nn_x_m": centroid_5nn_x_m,
        "centroid_5nn_y_m": centroid_5nn_y_m,

        "frames_to_next_setpiece": st_feats["frames_to_next_setpiece"],
        "frames_since_last_setpiece": st_feats["frames_since_last_setpiece"],
        "setpiece_team": st_feats["setpiece_team"],
        "in_open_play_phase": st_feats["in_open_play_phase"],
        "open_play_team": st_feats["open_play_team"],
    }

    np.savez_compressed(FEATURES_NPZ_PATH, **feature_dict)
    file_size_mb = os.path.getsize(FEATURES_NPZ_PATH) / (1024 * 1024)
    print(f"[OK] Compressed feature bundle saved: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    run_feature_extraction()
