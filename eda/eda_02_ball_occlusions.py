import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, GAMES

ATD_RAW_PATH = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
ATD_META_PATH = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

FPS = 25.0
PITCH_LEN, PITCH_WID = GAMES[GAME]["pitch"]

def get_team_mapping():
    """Load tracking metadata XML and extract team affiliations.

    Returns:
        np.ndarray: 1D array with team mapping (0: Home, 1: Away) per player slot.
    """
    root = ET.parse(ATD_META_PATH).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]

    return np.array([0 if s == home_id else 1 for s in slots], dtype=np.int8)

def run_analysis():
    """Analyze ball occlusion dynamics, gap distributions, and player proximity."""
    print("=" * 70)
    print("EDA 02: Ball Occlusion Dynamics, 5-NN Analysis & Interpolation")
    print("=" * 70)

    team_of_track = get_team_mapping()
    n_tracks = len(team_of_track)
    print(f"Loaded {n_tracks} track slots. (Home: {(team_of_track==0).sum()}, Away: {(team_of_track==1).sum()})")

    gt_df = pd.read_csv(GT_PATH)
    gt_df.columns = [c.strip().lower() for c in gt_df.columns]
    gt_labels = gt_df["label"].values
    gt_frames = gt_df["frame"].values
    n_total_frames = len(gt_df)

    gt_min_frame = 11185
    gt_max_frame = 18684

    print("Streaming ATD raw tracking data...")

    ball_seen = np.zeros(n_total_frames, dtype=bool)
    ball_speed = np.full(n_total_frames, np.nan, dtype=np.float32)
    ball_x_m = np.full(n_total_frames, np.nan, dtype=np.float32)
    ball_y_m = np.full(n_total_frames, np.nan, dtype=np.float32)

    d_nearest_home = np.full(n_total_frames, np.nan, dtype=np.float32)
    d_nearest_away = np.full(n_total_frames, np.nan, dtype=np.float32)

    with open(ATD_RAW_PATH, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame_idx = int(parts[0]) - 1
            if frame_idx >= n_total_frames:
                continue

            ball = parts[2]
            if ball.startswith("N"):
                continue

            ball_seen[frame_idx] = True
            b_coords = ball.split(",")
            bx = float(b_coords[0]) * PITCH_LEN
            by = float(b_coords[1]) * PITCH_WID
            ball_x_m[frame_idx] = bx
            ball_y_m[frame_idx] = by
            if len(b_coords) > 2 and not b_coords[2].startswith("N"):
                ball_speed[frame_idx] = float(b_coords[2])

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
                if team_of_track[slot_idx] == 0:
                    h_dists.append(dist)
                else:
                    a_dists.append(dist)

            if h_dists:
                d_nearest_home[frame_idx] = min(h_dists)
            if a_dists:
                d_nearest_away[frame_idx] = min(a_dists)

    print("Tracking stream processed.")

    total_ball_seen = ball_seen.sum()
    pct_total_seen = total_ball_seen / n_total_frames * 100
    print(f"Full Match Ball Visibility: {total_ball_seen:,} / {n_total_frames:,} frames ({pct_total_seen:.2f}%)")

    gt_mask = (np.arange(n_total_frames) >= gt_min_frame) & (np.arange(n_total_frames) <= gt_max_frame)
    gt_ball_seen = ball_seen[gt_mask]
    n_gt_frames = gt_mask.sum()
    pct_gt_seen = gt_ball_seen.sum() / n_gt_frames * 100
    print(f"Public GT Window Ball Visibility: {gt_ball_seen.sum():,} / {n_gt_frames:,} frames ({pct_gt_seen:.2f}%)")
    print(f"Ball Missing in Public GT: {n_gt_frames - gt_ball_seen.sum():,} frames ({(100 - pct_gt_seen):.2f}%)")

    missing_runs = []
    in_gap = False
    gap_start = 0

    for i in range(n_total_frames):
        if not ball_seen[i] and not in_gap:
            in_gap = True
            gap_start = i
        elif ball_seen[i] and in_gap:
            in_gap = False
            missing_runs.append((gap_start, i - 1, i - gap_start))
    if in_gap:
        missing_runs.append((gap_start, n_total_frames - 1, n_total_frames - gap_start))

    gap_df = pd.DataFrame(missing_runs, columns=["start_frame", "end_frame", "length_frames"])
    gap_df["length_seconds"] = gap_df["length_frames"] / FPS

    micro_gaps = gap_df[gap_df["length_frames"] <= 12]
    med_gaps = gap_df[(gap_df["length_frames"] > 12) & (gap_df["length_frames"] <= 50)]
    long_gaps = gap_df[gap_df["length_frames"] > 50]

    print("\n--- Gap Length Distribution (Full Match) ---")
    print(f"Total Gaps: {len(gap_df):,}")
    print(f"  Micro-gaps (<= 0.5s / <= 12f):    {len(micro_gaps):6d} gaps ({len(micro_gaps)/len(gap_df)*100:5.1f}%) | {micro_gaps['length_frames'].sum():,d} frames")
    print(f"  Medium gaps (0.5s - 2s / 13-50f): {len(med_gaps):6d} gaps ({len(med_gaps)/len(gap_df)*100:5.1f}%) | {med_gaps['length_frames'].sum():,d} frames")
    print(f"  Long gaps (> 2.0s / > 50f):       {len(long_gaps):6d} gaps ({len(long_gaps)/len(gap_df)*100:5.1f}%) | {long_gaps['length_frames'].sum():,d} frames")

    gt_sub_labels = gt_labels[gt_mask]
    gt_sub_ball_seen = ball_seen[gt_mask]
    missing_mask_in_gt = ~gt_sub_ball_seen

    missing_true_labels = pd.Series(gt_sub_labels[missing_mask_in_gt]).value_counts()
    print("\n--- Ground Truth Reality when Ball is Missing in Public Window ---")
    print(f"Total Missing Frames in GT: {missing_mask_in_gt.sum()}")
    for lbl, cnt in missing_true_labels.items():
        print(f"  True Label '{lbl}': {cnt:5d} frames ({cnt / missing_mask_in_gt.sum()*100:5.2f}%)")

    alive_in_missing = missing_true_labels.get("H", 0) + missing_true_labels.get("A", 0)
    dead_in_missing = missing_true_labels.get("D", 0)
    print(f"  --> Actually ALIVE (H or A): {alive_in_missing} ({alive_in_missing / missing_mask_in_gt.sum()*100:.1f}%)")
    print(f"  --> Actually DEAD (D):       {dead_in_missing} ({dead_in_missing / missing_mask_in_gt.sum()*100:.1f}%)")

    print("\n--- Forward-Fill Recovery in Public GT ---")
    forward_fill_results = []

    base_team = np.full(n_gt_frames, -1, dtype=int)
    for idx_in_gt, f_idx in enumerate(range(gt_min_frame, gt_max_frame + 1)):
        if ball_seen[f_idx]:
            dh = d_nearest_home[f_idx]
            da = d_nearest_away[f_idx]
            if not np.isnan(dh) and not np.isnan(da):
                base_team[idx_in_gt] = 0 if dh <= da else 1

    for L in [0, 6, 12, 25, 50, 75, 100, 125]:
        filled = base_team.copy()
        if L > 0:
            last_val = -1
            gap_len = 0
            for i in range(len(filled)):
                if filled[i] != -1:
                    last_val = filled[i]
                    gap_len = 0
                else:
                    gap_len += 1
                    if gap_len <= L and last_val != -1:
                        filled[i] = last_val

        gt_alive_mask = (gt_sub_labels == "H") | (gt_sub_labels == "A")
        gt_alive_targets = np.where(gt_sub_labels[gt_alive_mask] == "H", 0, 1)
        pred_alive_calls = filled[gt_alive_mask]

        correct_alive = (pred_alive_calls == gt_alive_targets).sum()
        total_alive = gt_alive_mask.sum()
        acc_alive = correct_alive / total_alive * 100

        remaining_d = (filled == -1).sum()

        forward_fill_results.append({
            "L_frames": L,
            "L_seconds": L / FPS,
            "acc_alive_pct": acc_alive,
            "remaining_d_frames": remaining_d,
            "correct_alive_frames": correct_alive
        })
        print(f"  Window L={L:3d} ({L/FPS:4.2f}s): Acc on Alive: {acc_alive:5.2f}% | Correct: {correct_alive:5d}/{total_alive}")

    h_poss_mask = gt_mask & (gt_labels == "H") & ball_seen
    a_poss_mask = gt_mask & (gt_labels == "A") & ball_seen

    mean_dh_in_h = np.nanmean(d_nearest_home[h_poss_mask])
    mean_da_in_h = np.nanmean(d_nearest_away[h_poss_mask])

    mean_dh_in_a = np.nanmean(d_nearest_home[a_poss_mask])
    mean_da_in_a = np.nanmean(d_nearest_away[a_poss_mask])

    print("\n--- Proximity to Ball (Metres) during Ground Truth Possession ---")
    print(f"When HOME (Barcelona) has possession:")
    print(f"  Dist nearest Home player: {mean_dh_in_h:.2f} m | Dist nearest Away player: {mean_da_in_h:.2f} m (Delta: {mean_da_in_h - mean_dh_in_h:.2f} m)")
    print(f"When AWAY (Athletic) has possession:")
    print(f"  Dist nearest Away player: {mean_da_in_a:.2f} m | Dist nearest Home player: {mean_dh_in_a:.2f} m (Delta: {mean_dh_in_a - mean_da_in_a:.2f} m)")

if __name__ == "__main__":
    run_analysis()
