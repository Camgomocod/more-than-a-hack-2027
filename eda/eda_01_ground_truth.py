import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
META_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public_meta.json")

FPS = 25.0

def run_analysis():
    """Execute statistical ground-truth analysis on the public possession annotations.

    Computes class distributions, spell duration dynamics, and empirical state transitions.
    """
    print("=" * 70)
    print("EDA 01: Ground Truth Analysis & Hierarchical Decomposition")
    print("=" * 70)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    df = pd.read_csv(GT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    total_frames = len(df)
    graded_mask = df["label"].isin(["H", "A", "D"])
    graded_df = df[graded_mask].copy().reset_index(drop=True)
    n_graded = len(graded_df)

    print(f"Total Match Frames:   {total_frames:,}")
    print(f"Graded (Public GT):   {n_graded:,} frames ({n_graded / FPS:.1f} s / {n_graded / (FPS * 60):.2f} min)")

    counts = graded_df["label"].value_counts()
    h_count = counts.get("H", 0)
    a_count = counts.get("A", 0)
    d_count = counts.get("D", 0)
    alive_count = h_count + a_count

    p_alive = alive_count / n_graded
    p_dead = d_count / n_graded
    p_h_given_alive = h_count / alive_count if alive_count > 0 else 0
    p_a_given_alive = a_count / alive_count if alive_count > 0 else 0

    print("\n--- Hierarchical Breakdown (Borghesi et al. framework) ---")
    print(f"1. Dead vs. Alive (DA):")
    print(f"   Alive (H + A): {alive_count:5d} frames ({p_alive*100:5.2f}%)")
    print(f"   Dead  (D):     {d_count:5d} frames ({p_dead*100:5.2f}%)")
    print(f"2. Possession given Alive (POSS | Alive):")
    print(f"   Home (H):      {h_count:5d} frames ({p_h_given_alive*100:5.2f}%)")
    print(f"   Away (A):      {a_count:5d} frames ({p_a_given_alive*100:5.2f}%)")

    labels = graded_df["label"].values
    frames = graded_df["frame"].values

    change_indices = np.where(labels[:-1] != labels[1:])[0] + 1
    split_indices = np.concatenate(([0], change_indices, [len(labels)]))

    segments = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        seg_label = labels[start_idx]
        seg_start_frame = frames[start_idx]
        seg_end_frame = frames[end_idx - 1]
        dur_frames = end_idx - start_idx
        dur_seconds = dur_frames / FPS
        segments.append({
            "segment_id": i + 1,
            "label": seg_label,
            "start_frame": seg_start_frame,
            "end_frame": seg_end_frame,
            "dur_frames": dur_frames,
            "dur_seconds": dur_seconds,
            "is_possession": seg_label in ["H", "A"]
        })

    seg_df = pd.DataFrame(segments)

    poss_df = seg_df[seg_df["is_possession"]].copy().reset_index(drop=True)
    dead_df = seg_df[~seg_df["is_possession"]].copy().reset_index(drop=True)

    print(f"\nTotal Segments: {len(seg_df)}")
    print(f"Possessions:    {len(poss_df)} ({len(poss_df[poss_df['label']=='H'])} H, {len(poss_df[poss_df['label']=='A'])} A)")
    print(f"Dead Stoppages: {len(dead_df)}")

    poss_dur = poss_df["dur_seconds"]
    h_dur = poss_df[poss_df["label"] == "H"]["dur_seconds"]
    a_dur = poss_df[poss_df["label"] == "A"]["dur_seconds"]
    d_dur = dead_df["dur_seconds"]

    def stats_summary(s):
        """Compute descriptive summary statistics for a duration series.

        Args:
            s (pd.Series): Sequence of numerical duration values.

        Returns:
            dict: Summary metrics including count, mean, std, quantiles, and bounds.
        """
        return {
            "count": len(s),
            "mean": s.mean() if len(s) else 0,
            "std": s.std() if len(s) > 1 else 0,
            "min": s.min() if len(s) else 0,
            "p25": s.quantile(0.25) if len(s) else 0,
            "median": s.median() if len(s) else 0,
            "p75": s.quantile(0.75) if len(s) else 0,
            "p90": s.quantile(0.90) if len(s) else 0,
            "max": s.max() if len(s) else 0,
        }

    p_stats = stats_summary(poss_dur)
    d_stats = stats_summary(d_dur)

    print("\n--- Possession Duration (seconds) ---")
    print(f"  Mean: {p_stats['mean']:.2f}s | Median: {p_stats['median']:.2f}s | Min: {p_stats['min']:.2f}s | Max: {p_stats['max']:.2f}s | Std: {p_stats['std']:.2f}s")
    print(f"  H Mean: {h_dur.mean():.2f}s (n={len(h_dur)}) | A Mean: {a_dur.mean():.2f}s (n={len(a_dur)})")

    print("\n--- Dead Stoppage Duration (seconds) ---")
    print(f"  Mean: {d_stats['mean']:.2f}s | Median: {d_stats['median']:.2f}s | Min: {d_stats['min']:.2f}s | Max: {d_stats['max']:.2f}s")

    states = ["H", "A", "D"]
    state_to_idx = {s: i for i, s in enumerate(states)}
    trans_counts = np.zeros((3, 3), dtype=int)

    for i in range(len(labels) - 1):
        s_prev = labels[i]
        s_next = labels[i + 1]
        trans_counts[state_to_idx[s_prev], state_to_idx[s_next]] += 1

    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    spell_transitions = []
    for i in range(len(segments) - 1):
        from_label = segments[i]["label"]
        to_label = segments[i + 1]["label"]
        spell_transitions.append(f"{from_label} -> {to_label}")
    spell_trans_counts = pd.Series(spell_transitions).value_counts()

    print("\n--- Spell-Level Transitions ---")
    for trans, count in spell_trans_counts.items():
        print(f"  {trans:10s} : {count}")

if __name__ == "__main__":
    run_analysis()
