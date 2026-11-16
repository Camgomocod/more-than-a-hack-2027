import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import load_smart_tagging as lst

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

FPS = 25.0
GT_MIN = 11185
GT_MAX = 18684

def run_analysis():
    """Analyze tactical Smart Tagging events and their correlation with dead-ball phases."""
    print("=" * 70)
    print("EDA 03: Smart Tagging Mapping & Dead-Ball / Possession Priors")
    print("=" * 70)

    events = lst.load()
    print(f"Loaded {len(events)} Smart Tagging instances.")

    rows = []
    for ev in events:
        tags = ev["tags"]
        team = tags.get("Team", "N/A")
        sub_type = tags.get("Type", tags.get("type", "N/A"))
        half = tags.get("Half", "N/A")
        rows.append({
            "code": ev["code"].strip(),
            "start_frame": ev["start_frame"],
            "end_frame": ev["end_frame"],
            "start_s": ev["start_s"],
            "end_s": ev["end_s"],
            "dur_s": ev["end_s"] - ev["start_s"],
            "team": team,
            "type": sub_type,
            "half": half
        })
    df_events = pd.DataFrame(rows)

    code_counts = df_events["code"].value_counts()
    print("\n--- Event Distribution across Match ---")
    for code, cnt in code_counts.items():
        print(f"  {code:30s} : {cnt:3d}")

    set_pieces = df_events[df_events["code"].str.upper() == "SET PIECES"].copy()
    print(f"\nTotal SET PIECES: {len(set_pieces)}")
    sp_types = set_pieces["type"].value_counts()
    print("Set Piece Subtypes:")
    for stype, cnt in sp_types.items():
        print(f"  {stype:25s} : {cnt:2d}")

    gt_df = pd.read_csv(GT_PATH)
    gt_df.columns = [c.strip().lower() for c in gt_df.columns]
    gt_labels = gt_df["label"].values

    gt_events = df_events[
        (df_events["end_frame"] >= GT_MIN) & (df_events["start_frame"] <= GT_MAX)
    ].copy()
    print(f"\nEvents overlapping Public GT window: {len(gt_events)}")

    gt_dead_spans = [
        (12851, 12988, 5.52),
        (15889, 16399, 20.44),
        (17380, 17837, 18.32),
        (18027, 18388, 14.48)
    ]

    print("\n--- Correlation: GT Dead Ball Stoppages vs. Smart Tagging Events ---")
    dead_matches = []
    for d_start, d_end, dur in gt_dead_spans:
        nearby_events = df_events[
            (df_events["start_frame"] <= d_end + 75) & (df_events["end_frame"] >= d_start - 75)
        ]
        matched_desc = []
        for _, ev in nearby_events.iterrows():
            offset_start = ev["start_frame"] - d_start
            offset_end = ev["end_frame"] - d_end
            matched_desc.append(f"{ev['code']} ({ev['type']}) [start offset: {offset_start}f, end offset: {offset_end}f, team: {ev['team']}]")

        dead_matches.append({
            "gt_span": f"{d_start} - {d_end} ({dur:.2f}s)",
            "matched_events": matched_desc
        })
        print(f"Stoppage {d_start}-{d_end} ({dur:.2f}s):")
        for m in matched_desc:
            print(f"   -> {m}")

    event_priors = []
    for code in df_events["code"].unique():
        ev_subset = df_events[df_events["code"] == code]

        total_overlap_frames = 0
        h_frames = 0
        a_frames = 0
        d_frames = 0

        for _, ev in ev_subset.iterrows():
            ef_start = max(ev["start_frame"], GT_MIN)
            ef_end = min(ev["end_frame"], GT_MAX)
            if ef_start <= ef_end:
                span_len = ef_end - ef_start + 1
                total_overlap_frames += span_len
                span_labels = gt_labels[ef_start:ef_end + 1]
                h_frames += (span_labels == "H").sum()
                a_frames += (span_labels == "A").sum()
                d_frames += (span_labels == "D").sum()

        if total_overlap_frames > 0:
            event_priors.append({
                "code": code,
                "overlap_frames": total_overlap_frames,
                "dur_s": total_overlap_frames / FPS,
                "p_alive": (h_frames + a_frames) / total_overlap_frames,
                "p_dead": d_frames / total_overlap_frames,
                "p_h": h_frames / total_overlap_frames,
                "p_a": a_frames / total_overlap_frames
            })

    df_priors = pd.DataFrame(event_priors)

if __name__ == "__main__":
    run_analysis()
