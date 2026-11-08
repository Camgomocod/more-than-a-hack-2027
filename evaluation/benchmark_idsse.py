"""Benchmark HMM/Viterbi Model on IDSSE Bundesliga Open Tracking & Event Data.
Matches: Sportec Solutions TRACAB optical tracking (25 fps) with official DFL events.
Evaluates out-of-domain generalization on 1. FC Köln vs FC Bayern München (Matchday 34).
"""

import xml.etree.ElementTree as ET
import numpy as np
import time
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import possession_metrics as pm

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
                        status = fr.get("BallStatus")
                        poss = fr.get("BallPossession")
                        ball_frames[n] = (x, y, status, poss)
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
        bx, by, status, poss = ball_frames[fn]
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

    h_pids = list(home_player_frames.keys())
    a_pids = list(away_player_frames.keys())

    for idx, fn in enumerate(frame_numbers):
        bx, by, _, _ = ball_frames[fn]
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

    pred_labels = np.array([["H", "A", "D"][x] for x in labels])

    # 7. Evaluate
    gt_codes = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[x] for x in gt_labels], dtype=np.int8)
    pred_codes = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[x] for x in pred_labels], dtype=np.int8)

    res = pm.evaluate(gt_codes, pred_codes, taus=(12, 25), bootstrap_n=0)
    elapsed = time.time() - t0

    print(f"Completed in {elapsed:.2f}s for {N:,} frames ({N/25.0/60.0:.1f} min).")
    print(f"  macroF1:        {res['macro_f1']:.4f}")
    print(f"  spellF1:        {res['spell_f1_mean']:.4f}")
    print(f"  Accuracy:       {res['accuracy']:.4f}")
    print(f"  Cohen Kappa:    {res['kappa']:.4f}")
    print(f"  Alive/Dead F1:  {res['alive_dead_f1']:.4f}")
    print(f"  F1 H (Köln):    {res['per_class']['H']['f1']:.4f}")
    print(f"  F1 A (Bayern):  {res['per_class']['A']['f1']:.4f}")
    print(f"  F1 D:           {res['per_class']['D']['f1']:.4f}")
    print(f"  Share Error:    {res['share_err']:.4f} (GT: {res['gt_home_share']:.4f}, Pred: {res['pred_home_share']:.4f})")
    print(f"  Possessions:    GT {res['n_gt_spells']} vs Pred {res['n_pred_spells']}")
    return res


def run_benchmark():
    r1 = evaluate_match_half("firstHalf")
    r2 = evaluate_match_half("secondHalf")

    print("\n" + "=" * 60)
    print("CONSOLIDATED FULL-MATCH BENCHMARK (145,967 FRAMES / 97.3 MIN)")
    print("=" * 60)
    avg_macro = (r1["macro_f1"] + r2["macro_f1"]) / 2.0
    avg_spell = (r1["spell_f1_mean"] + r2["spell_f1_mean"]) / 2.0
    avg_acc = (r1["accuracy"] + r2["accuracy"]) / 2.0
    avg_kappa = (r1["kappa"] + r2["kappa"]) / 2.0
    avg_ad = (r1["alive_dead_f1"] + r2["alive_dead_f1"]) / 2.0
    avg_fd = (r1["per_class"]["D"]["f1"] + r2["per_class"]["D"]["f1"]) / 2.0
    avg_share_err = (r1["share_err"] + r2["share_err"]) / 2.0

    print(f"Macro F1 (Full Match Mean):     {avg_macro:.4f}")
    print(f"Spell F1 (Full Match Mean):     {avg_spell:.4f}")
    print(f"Accuracy (Full Match Mean):     {avg_acc:.4f}")
    print(f"Cohen Kappa (Full Match Mean):  {avg_kappa:.4f}")
    print(f"Alive/Dead F1:                  {avg_ad:.4f}")
    print(f"Dead Ball (D) F1:               {avg_fd:.4f}")
    print(f"Possession Share Error:         {avg_share_err:.4f} (~{avg_share_err*100:.2f}%)")


if __name__ == "__main__":
    run_benchmark()
