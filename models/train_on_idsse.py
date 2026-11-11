import xml.etree.ElementTree as ET
import numpy as np
import json
import time
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

HOME_ID = "DFL-CLU-000008"
AWAY_ID = "DFL-CLU-00000G"

POS_FILE = os.path.join(ROOT, "data", "idsse", "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml")
OUT_WEIGHTS = os.path.join(HERE, "bundesliga_weights.json")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def extract_match_dataset():
    print("=" * 70)
    print("1. Extracting 145,967 Frames of Tracking Data from IDSSE Bundesliga...")
    print("=" * 70)
    t0 = time.time()

    all_d_h = []
    all_d_a = []
    all_k5_ratio = []
    all_ball_out = []
    all_ball_speed = []
    all_gt_labels = []

    for section in ["firstHalf", "secondHalf"]:
        print(f"Streaming {section}...")
        ball_frames = {}
        home_player_frames = {}
        away_player_frames = {}

        context = ET.iterparse(POS_FILE, events=("start", "end"))
        for event, elem in context:
            if event == "end" and elem.tag == "FrameSet":
                sec = elem.get("GameSection")
                if sec == section:
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

        d_h = np.full(N, 50.0, dtype=np.float32)
        d_a = np.full(N, 50.0, dtype=np.float32)
        k5_ratio = np.full(N, 0.5, dtype=np.float32)
        ball_out = np.zeros(N, dtype=np.int8)
        ball_speed = np.zeros(N, dtype=np.float32)

        h_pids = list(home_player_frames.keys())
        a_pids = list(away_player_frames.keys())

        gt_labels = []
        for idx, fn in enumerate(frame_numbers):
            bx, by, bs, status, poss = ball_frames[fn]
            ball_speed[idx] = bs
            if abs(bx) > 52.5 or abs(by) > 34.0:
                ball_out[idx] = 1

            if status == "0":
                gt_labels.append(2)
            elif poss == "1":
                gt_labels.append(0)
            elif poss == "2":
                gt_labels.append(1)
            else:
                gt_labels.append(3)

            h_dists = [
                np.hypot(home_player_frames[pid][fn][0] - bx, home_player_frames[pid][fn][1] - by)
                for pid in h_pids if fn in home_player_frames[pid]
            ]
            a_dists = [
                np.hypot(away_player_frames[pid][fn][0] - bx, away_player_frames[pid][fn][1] - by)
                for pid in a_pids if fn in away_player_frames[pid]
            ]

            if h_dists: d_h[idx] = min(h_dists)
            if a_dists: d_a[idx] = min(a_dists)

            all_dists = [(d, 0) for d in h_dists] + [(d, 1) for d in a_dists]
            if all_dists:
                all_dists.sort(key=lambda item: item[0])
                top5 = all_dists[:min(5, len(all_dists))]
                k5_ratio[idx] = sum(1 for d, t in top5 if t == 0) / float(len(top5))

        all_d_h.append(d_h)
        all_d_a.append(d_a)
        all_k5_ratio.append(k5_ratio)
        all_ball_out.append(ball_out)
        all_ball_speed.append(ball_speed)
        all_gt_labels.append(np.array(gt_labels, dtype=np.int8))

    d_h_full = np.concatenate(all_d_h)
    d_a_full = np.concatenate(all_d_a)
    k5_ratio_full = np.concatenate(all_k5_ratio)
    ball_out_full = np.concatenate(all_ball_out)
    ball_speed_full = np.concatenate(all_ball_speed)
    gt_full = np.concatenate(all_gt_labels)

    elapsed = time.time() - t0
    print(f"Extraction completed in {elapsed:.2f}s. Total frames: {len(gt_full):,}")
    print(f"Label breakdown: Home(H): {(gt_full==0).sum():,}, Away(A): {(gt_full==1).sum():,}, Dead(D): {(gt_full==2).sum():,}")
    return d_h_full, d_a_full, k5_ratio_full, ball_out_full, ball_speed_full, gt_full

def train_logistic_regression(X, y, l2_reg=1e-4, lr=0.05, max_iter=200):
    N, D = X.shape
    weights = np.zeros(D, dtype=np.float64)
    bias = 0.0
    v_w = np.zeros(D, dtype=np.float64)
    v_b = 0.0
    momentum = 0.9

    for it in range(max_iter):
        z = np.dot(X, weights) + bias
        preds = sigmoid(z)
        error = preds - y

        grad_w = (np.dot(X.T, error) / N) + (l2_reg * weights)
        grad_b = np.mean(error)

        v_w = momentum * v_w + lr * grad_w
        v_b = momentum * v_b + lr * grad_b

        weights -= v_w
        bias -= v_b

        if it % 50 == 0 or it == max_iter - 1:
            loss = -np.mean(y * np.log(np.maximum(1e-12, preds)) + (1 - y) * np.log(np.maximum(1e-12, 1 - preds)))
            acc = np.mean((preds >= 0.5) == y)
            print(f"  Iteration {it:3d}: Log-Loss={loss:.4f}, Accuracy={acc*100:.2f}%")

    return weights, bias

def run_training():
    d_h, d_a, k5_ratio, ball_out, ball_speed, gt = extract_match_dataset()

    print("\n" + "=" * 70)
    print("2. Training Stage 1 Model: DA (Dead vs Alive Game)")
    print("=" * 70)

    y_da = (gt == 2).astype(np.float64)

    min_dist = np.minimum(d_h, d_a)
    dist_excess = np.maximum(0.0, min_dist - 3.0)

    X_da = np.stack([
        ball_out.astype(np.float64),
        dist_excess.astype(np.float64),
    ], axis=1)

    feature_names_da = ["ball_out_of_pitch", "min_dist_excess_3m"]
    w_da, b_da = train_logistic_regression(X_da, y_da, l2_reg=1e-3, lr=0.1, max_iter=150)
    print(f"DA Model Learned Parameters:")
    print(f"  Bias (b_da): {b_da:.4f}")
    for name, w in zip(feature_names_da, w_da):
        print(f"  Weight [{name}]: {w:.4f}")

    print("\n" + "=" * 70)
    print("3. Training Stage 2 Model: POSS (Home vs Away | Alive)")
    print("=" * 70)

    alive_mask = (gt == 0) | (gt == 1)
    d_h_alive = d_h[alive_mask]
    d_a_alive = d_a[alive_mask]
    k5_alive = k5_ratio[alive_mask]
    y_poss_orig = (gt[alive_mask] == 0).astype(np.float64)

    delta_d_alive = np.clip(d_h_alive - d_a_alive, -8.0, 8.0)
    k5_centered = k5_alive - 0.5

    X_poss_orig = np.stack([delta_d_alive, k5_centered], axis=1)
    X_poss_swap = np.stack([-delta_d_alive, -k5_centered], axis=1)
    y_poss_swap = 1.0 - y_poss_orig

    X_poss_aug = np.concatenate([X_poss_orig, X_poss_swap], axis=0)
    y_poss_aug = np.concatenate([y_poss_orig, y_poss_swap], axis=0)
    print(f"Symmetric Augmentation: {len(X_poss_orig):,} -> {len(X_poss_aug):,} balanced training samples.")

    feature_names_poss = ["delta_d_m", "k5_home_ratio_centered"]
    w_poss, b_poss = train_logistic_regression(X_poss_aug, y_poss_aug, l2_reg=1e-3, lr=0.1, max_iter=150)
    print(f"POSS Model Learned Parameters:")
    print(f"  Bias (b_poss): {b_poss:.4f} (Near 0 by symmetry)")
    for name, w in zip(feature_names_poss, w_poss):
        print(f"  Weight [{name}]: {w:.4f}")

    weights_dict = {
        "dataset": "IDSSE Bundesliga (DFL-MAT-J03WMX)",
        "total_training_frames": len(gt),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_da": {
            "bias": float(b_da),
            "weights": {name: float(w) for name, w in zip(feature_names_da, w_da)},
        },
        "model_poss": {
            "bias": float(b_poss),
            "weights": {name: float(w) for name, w in zip(feature_names_poss, w_poss)},
        },
    }

    with open(OUT_WEIGHTS, "w") as f:
        json.dump(weights_dict, f, indent=2)

    print(f"\n[OK] Trained weights successfully saved to: {OUT_WEIGHTS}")
    return weights_dict

if __name__ == "__main__":
    run_training()
