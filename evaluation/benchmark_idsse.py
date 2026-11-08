"""Benchmark HMM/Viterbi Model on IDSSE Bundesliga Open Tracking & Event Data.
Matches: Sportec Solutions TRACAB optical tracking (25 fps) with official DFL events.
Evaluates out-of-domain generalization on 1. FC Köln vs FC Bayern München (Matchday 34).
Includes Phase 5.1 Kinematic Boundary Refinement comparison.
"""

import xml.etree.ElementTree as ET
import numpy as np
import time
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))

import possession_metrics as pm
from boundary_refinement import refine_transition_boundaries

HOME_ID = "DFL-CLU-000008"  # 1. FC Köln
AWAY_ID = "DFL-CLU-00000G"  # FC Bayern München

POS_FILE = os.path.join(ROOT, "data", "idsse", "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml")
EVENTS_FILE = os.path.join(ROOT, "data", "idsse", "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def evaluate_match_half(target_sec="firstHalf", min_spell=50):
    print(f"\n=======================================================")
    print(f"Loading and Evaluating {target_sec} from IDSSE...")
    print(f"=======================================================")
    t0 = time.time()
    ball_frames = {}
    home_player_frames = {}
    away_player_frames = {}

    context = ET.iterparse(POS_FILE, events=("start", "end"))

    for event, elem in context:
        if event == "end" and elem.tag == "FrameSet":
            sec = elem.get("GameSection")
            if sec == target_sec:
                team = elem.get("TeamId")
                pid = elem.get("PersonId")
                if team == "BALL":
                    for fr in elem:
                        n = int(fr.get("N"))
                        x = float(fr.get("X"))
                        y = float(fr.get("Y"))
                        s = float(fr.get("S")) if "S" in fr.attrib else 0.0
                        status = fr.get("BallStatus")
                        poss = fr.get("BallPossession")
                        ball_frames[n] = (x, y, s, status, poss)
                elif team == HOME_ID:
                    home_player_frames[pid] = {
                        int(fr.get("N")): (float(fr.get("X")), float(fr.get("Y"))) for fr in elem
                    }
                elif team == AWAY_ID:
                    away_player_frames[pid] = {
                        int(fr.get("N")): (float(fr.get("X")), float(fr.get("Y"))) for fr in elem
                    }
            elem.clear()

    frame_numbers = sorted(list(ball_frames.keys()))
    N = len(frame_numbers)

    # 1. Build Ground Truth from TRACAB
    gt_labels = []
    for fn in frame_numbers:
        bx, by, bs, status, poss = ball_frames[fn]
        if status == "0":
            gt_labels.append("D")
        elif poss == "1":
            gt_labels.append("H")
        elif poss == "2":
            gt_labels.append("A")
        else:
            gt_labels.append("X")
    gt_labels = np.array(gt_labels)

    # 2. Extract Features
    d_h = np.full(N, 50.0, dtype=np.float32)
    d_a = np.full(N, 50.0, dtype=np.float32)
    k5_ratio = np.full(N, 0.5, dtype=np.float32)
    ball_out = np.zeros(N, dtype=np.int8)
    ball_speed = np.zeros(N, dtype=np.float32)

    h_pids = list(home_player_frames.keys())
    a_pids = list(away_player_frames.keys())

    for idx, fn in enumerate(frame_numbers):
        bx, by, bs, _, _ = ball_frames[fn]
        ball_speed[idx] = bs
        if abs(bx) > 52.5 or abs(by) > 34.0:
            ball_out[idx] = 1

        h_dists = [
            np.hypot(home_player_frames[pid][fn][0] - bx, home_player_frames[pid][fn][1] - by)
            for pid in h_pids if fn in home_player_frames[pid]
        ]
        a_dists = [
            np.hypot(away_player_frames[pid][fn][0] - bx, away_player_frames[pid][fn][1] - by)
            for pid in a_pids if fn in away_player_frames[pid]
        ]

        if h_dists:
            d_h[idx] = min(h_dists)
        if a_dists:
            d_a[idx] = min(a_dists)

        all_dists = [(d, 0) for d in h_dists] + [(d, 1) for d in a_dists]
        if all_dists:
            all_dists.sort(key=lambda item: item[0])
            top5 = all_dists[:min(5, len(all_dists))]
            k5_ratio[idx] = sum(1 for d, t in top5 if t == 0) / float(len(top5))

    delta_d = d_h - d_a

    # 3. Model Emissions
    z_dead = (
        -2.5
        + 6.0 * ball_out
        + 1.5 * np.maximum(0, np.minimum(d_h, d_a) - 3.0)
    )
    p_d = sigmoid(z_dead)
    p_d = np.where(ball_out == 1, 0.999, p_d)
    p_alive = 1.0 - p_d

    z_poss = (
        - 1.5 * np.clip(delta_d, -8.0, 8.0)
        + 2.0 * (k5_ratio - 0.5)
    )
    p_h_alive = sigmoid(z_poss)
    p_a_alive = 1.0 - p_h_alive

    p_h = np.maximum(1e-6, p_alive * p_h_alive)
    p_a = np.maximum(1e-6, p_alive * p_a_alive)
    p_d = np.maximum(1e-6, p_d)

    total = p_h + p_a + p_d
    p_h /= total
    p_a /= total
    p_d /= total

    emissions = np.stack([np.log(p_h), np.log(p_a), np.log(p_d)], axis=1) * 0.80

    # 4. Markov Transition Matrix
    A = np.array([
        [0.9950, 0.0004, 0.0046],
        [0.0004, 0.9950, 0.0046],
        [0.0025, 0.0025, 0.9950]
    ])
    log_A = np.log(A)

    # 5. Viterbi Decoding
    viterbi = np.zeros((N, 3), dtype=np.float64)
    backpointer = np.zeros((N, 3), dtype=np.int32)
    viterbi[0] = emissions[0]

    for t in range(1, N):
        scores = viterbi[t - 1, :, None] + log_A
        backpointer[t] = np.argmax(scores, axis=0)
        viterbi[t] = np.max(scores, axis=0) + emissions[t]

    best_path = np.zeros(N, dtype=np.int8)
    best_path[-1] = np.argmax(viterbi[-1])
    for t in range(N - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    # 6. Debounce Filter
    labels = best_path.copy()
    if min_spell > 0:
        change_idx = np.where(labels[:-1] != labels[1:])[0] + 1
        splits = np.concatenate(([0], change_idx, [len(labels)]))
        for i in range(len(splits) - 1):
            s, e = splits[i], splits[i + 1]
            length = e - s
            lbl = labels[s]
            if length < min_spell:
                prev_lbl = labels[s - 1] if s > 0 else lbl
                next_lbl = labels[e] if e < len(labels) else lbl
                if prev_lbl == next_lbl:
                    labels[s:e] = prev_lbl
                elif prev_lbl in [0, 1]:
                    labels[s:e] = prev_lbl
                else:
                    labels[s:e] = next_lbl

    # 7. Apply Phase 5.1 Boundary Refinement
    labels_refined = refine_transition_boundaries(
        labels,
        d_h=d_h,
        d_a=d_a,
        ball_out=ball_out,
        ball_speed=ball_speed
    )

    gt_codes = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[x] for x in gt_labels], dtype=np.int8)
    res_base = pm.evaluate(gt_codes, labels, taus=(12, 25), bootstrap_n=0)
    res_ref = pm.evaluate(gt_codes, labels_refined, taus=(12, 25), bootstrap_n=0)
    elapsed = time.time() - t0

    print(f"Completed in {elapsed:.2f}s for {N:,} frames ({N/25.0/60.0:.1f} min).")
    print(f"  Macro F1:        Base={res_base['macro_f1']:.4f}  -->  Refined={res_ref['macro_f1']:.4f}")
    print(f"  Spell F1:        Base={res_base['spell_f1_mean']:.4f}  -->  Refined={res_ref['spell_f1_mean']:.4f}")
    print(f"  Accuracy:        Base={res_base['accuracy']*100:.2f}%  -->  Refined={res_ref['accuracy']*100:.2f}%")
    print(f"  Trans F1 (tau25):Base={res_base['transitions'][25]['f1']:.4f}  -->  Refined={res_ref['transitions'][25]['f1']:.4f}")
    print(f"  Mean Latency:    Base={res_base['transitions'][25]['mean_latency']:.2f}f  -->  Refined={res_ref['transitions'][25]['mean_latency']:.2f}f")
    print(f"  Share Error:     Base={res_base['share_err']:.4f}  -->  Refined={res_ref['share_err']:.4f}")
    return res_base, res_ref


def run_benchmark():
    r1_b, r1_r = evaluate_match_half("firstHalf")
    r2_b, r2_r = evaluate_match_half("secondHalf")

    print("\n" + "=" * 70)
    print("CONSOLIDATED FULL-MATCH BENCHMARK (145,967 FRAMES / 97.3 MIN)")
    print("=" * 70)

    avg_macro_b = (r1_b["macro_f1"] + r2_b["macro_f1"]) / 2.0
    avg_macro_r = (r1_r["macro_f1"] + r2_r["macro_f1"]) / 2.0

    avg_spell_b = (r1_b["spell_f1_mean"] + r2_b["spell_f1_mean"]) / 2.0
    avg_spell_r = (r1_r["spell_f1_mean"] + r2_r["spell_f1_mean"]) / 2.0

    avg_acc_b = (r1_b["accuracy"] + r2_b["accuracy"]) / 2.0
    avg_acc_r = (r1_r["accuracy"] + r2_r["accuracy"]) / 2.0

    avg_tf_b = (r1_b["transitions"][25]["f1"] + r2_b["transitions"][25]["f1"]) / 2.0
    avg_tf_r = (r1_r["transitions"][25]["f1"] + r2_r["transitions"][25]["f1"]) / 2.0

    avg_lat_b = (r1_b["transitions"][25]["mean_latency"] + r2_b["transitions"][25]["mean_latency"]) / 2.0
    avg_lat_r = (r1_r["transitions"][25]["mean_latency"] + r2_r["transitions"][25]["mean_latency"]) / 2.0

    print(f"{'Métrica':<25} {'Base (HMM Viterbi)':<22} {'Refinado (Paso 5.1)':<22} {'Diferencia':<10}")
    print("-" * 75)
    print(f"{'Spell F1 (Métrica Reina)':<25} {avg_spell_b:<22.4f} {avg_spell_r:<22.4f} {avg_spell_r - avg_spell_b:+.4f}")
    print(f"{'Macro F1':<25} {avg_macro_b:<22.4f} {avg_macro_r:<22.4f} {avg_macro_r - avg_macro_b:+.4f}")
    print(f"{'Accuracy (Exactitud)':<25} {avg_acc_b*100:<21.2f}% {avg_acc_r*100:<21.2f}% {(avg_acc_r - avg_acc_b)*100:+.2f}%")
    print(f"{'Transitions F1 (tau=25)':<25} {avg_tf_b:<22.4f} {avg_tf_r:<22.4f} {avg_tf_r - avg_tf_b:+.4f}")
    print(f"{'Latencia Media':<25} {avg_lat_b:<21.2f}f {avg_lat_r:<21.2f}f {avg_lat_r - avg_lat_b:+.2f}f")


if __name__ == "__main__":
    run_benchmark()
