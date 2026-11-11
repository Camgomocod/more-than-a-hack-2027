import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAMES, GAME

HOME, AWAY, DEAD, NOBODY = 0, 1, 2, 3
LABEL_MAP = {HOME: "H", AWAY: "A", DEAD: "D", NOBODY: "X"}
CODE_MAP = {"H": HOME, "A": AWAY, "D": DEAD, "X": NOBODY}

N_FRAMES = GAMES[GAME]["n_video_frames"]

HALF_1_KICKOFF = 11185
HALF_1_END = 90651
HALF_2_START = 110808
HALF_2_KICKOFF = 113742
HALF_2_END = 190197

P_STAY = 0.9950
P_SWITCH = 0.0020
EMIT_SCALE = 0.80
MIN_SPELL_FRAMES = 50

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def compute_emission_log_probabilities(feats):
    print("Computing hierarchical emission distributions...")

    delta_d = feats["delta_d_m"]
    d_h = feats["d_nearest_home_m"]
    d_a = feats["d_nearest_away_m"]
    k5_ratio = feats["k5_home_ratio"]
    out_pitch = feats["ball_out_of_pitch"]
    f_to_sp = feats["frames_to_next_setpiece"]
    in_open = feats["in_open_play_phase"]
    open_team = feats["open_play_team"]
    streak = feats["missing_streak_frames"]

    z_dead = (
        -2.8
        + 6.0 * out_pitch
        + 3.5 * np.maximum(0, (180 - f_to_sp) / 100.0)
        - 3.0 * in_open
        + 1.2 * np.minimum(streak / 50.0, 5.0) * (1 - in_open)
        + 1.2 * np.maximum(0, np.minimum(d_h, d_a) - 3.0)
    )
    p_dead = sigmoid(z_dead)
    p_dead = np.where(out_pitch == 1, 0.999, p_dead)
    p_alive = 1.0 - p_dead

    z_poss = (
        - 1.5 * np.clip(delta_d, -8.0, 8.0)
        + 2.0 * (k5_ratio - 0.5)
        + 0.8 * np.where(open_team == 1, 1.0, np.where(open_team == 2, -1.0, 0.0))
    )
    p_h_alive = sigmoid(z_poss)
    p_a_alive = 1.0 - p_h_alive

    p_h = np.maximum(1e-6, p_alive * p_h_alive)
    p_a = np.maximum(1e-6, p_alive * p_a_alive)
    p_d = np.maximum(1e-6, p_dead)

    total = p_h + p_a + p_d
    p_h /= total
    p_a /= total
    p_d /= total

    log_emissions = np.stack([np.log(p_h), np.log(p_a), np.log(p_d)], axis=1) * EMIT_SCALE
    return log_emissions

def build_transition_matrix():
    A = np.zeros((3, 3), dtype=np.float64)

    A[0, 0] = P_STAY
    A[0, 1] = P_SWITCH * 0.2
    A[0, 2] = 1.0 - A[0, 0] - A[0, 1]

    A[1, 1] = P_STAY
    A[1, 0] = P_SWITCH * 0.2
    A[1, 2] = 1.0 - A[1, 0] - A[1, 1]

    A[2, 2] = P_STAY * 0.995
    rem = (1.0 - A[2, 2]) / 2.0
    A[2, 0] = rem
    A[2, 1] = rem

    return np.log(np.maximum(1e-12, A))

def viterbi_decode(log_emissions, log_A):
    T, K = log_emissions.shape
    viterbi = np.zeros((T, K), dtype=np.float64)
    backpointer = np.zeros((T, K), dtype=np.int32)

    viterbi[0] = log_emissions[0]

    for t in range(1, T):
        scores = viterbi[t - 1, :, None] + log_A
        backpointer[t] = np.argmax(scores, axis=0)
        viterbi[t] = np.max(scores, axis=0) + log_emissions[t]

    best_path = np.zeros(T, dtype=np.int8)
    best_path[-1] = np.argmax(viterbi[-1])
    for t in range(T - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return best_path

def debounce_spells(labels, min_length=MIN_SPELL_FRAMES):
    labels_clean = labels.copy()
    change_idx = np.where(labels_clean[:-1] != labels_clean[1:])[0] + 1
    splits = np.concatenate(([0], change_idx, [len(labels_clean)]))

    for i in range(len(splits) - 1):
        s, e = splits[i], splits[i + 1]
        length = e - s
        lbl = labels_clean[s]
        if length < min_length:
            prev_lbl = labels_clean[s - 1] if s > 0 else lbl
            next_lbl = labels_clean[e] if e < len(labels_clean) else lbl
            if prev_lbl == next_lbl:
                labels_clean[s:e] = prev_lbl
            elif prev_lbl in [HOME, AWAY]:
                labels_clean[s:e] = prev_lbl
            else:
                labels_clean[s:e] = next_lbl

    return labels_clean

def run_hmm_viterbi_pipeline():
    print("=" * 70)
    print("Running Hierarchical HMM Viterbi Pipeline...")
    print("=" * 70)

    feats_path = os.path.join(ROOT, "features", "features.npz")
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Features file not found at {feats_path}. Run features/extract_features.py first.")

    print(f"Loading features from {os.path.basename(feats_path)}...")
    feats = np.load(feats_path)

    log_emissions = compute_emission_log_probabilities(feats)
    log_A = build_transition_matrix()

    final_preds = np.full(N_FRAMES, NOBODY, dtype=np.int8)

    print(f"\nDecoding First Half: frames {HALF_1_KICKOFF:,} to {HALF_1_END:,} ({HALF_1_END - HALF_1_KICKOFF:,} frames)...")
    h1_emits = log_emissions[HALF_1_KICKOFF:HALF_1_END]
    h1_path = viterbi_decode(h1_emits, log_A)
    h1_clean = debounce_spells(h1_path, MIN_SPELL_FRAMES)
    final_preds[HALF_1_KICKOFF:HALF_1_END] = h1_clean

    print(f"Assigning Second Half pre-kickoff: frames {HALF_2_START:,} to {HALF_2_KICKOFF:,} -> DEAD")
    final_preds[HALF_2_START:HALF_2_KICKOFF] = DEAD

    print(f"Decoding Second Half: frames {HALF_2_KICKOFF:,} to {HALF_2_END:,} ({HALF_2_END - HALF_2_KICKOFF:,} frames)...")
    h2_emits = log_emissions[HALF_2_KICKOFF:HALF_2_END]
    h2_path = viterbi_decode(h2_emits, log_A)
    h2_clean = debounce_spells(h2_path, MIN_SPELL_FRAMES)
    final_preds[HALF_2_KICKOFF:HALF_2_END] = h2_clean

    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_viterbi.csv")

    out_labels = [LABEL_MAP[p] for p in final_preds]
    df_out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": out_labels})
    df_out.to_csv(out_path, index=False)

    print(f"\n[OK] Submission file saved to: {out_path}")
    print(f"Total rows: {len(df_out):,}")
    print("Label distribution in submission:")
    print(df_out["label"].value_counts().to_string())

    return out_path

if __name__ == "__main__":
    run_hmm_viterbi_pipeline()
