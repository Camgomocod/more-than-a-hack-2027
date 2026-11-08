"""Hierarchical HMM Viterbi Model with Weights Learned from IDSSE Bundesliga.
Phase 5.2: Uses analytical emission parameters learned from 145,967 frames of Bundesliga TRACAB tracking.
Applies Viterbi global decoding and Phase 5.1 kinematic boundary refinement.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAMES, GAME
from hmm_viterbi import (
    HOME, AWAY, DEAD, NOBODY, LABEL_MAP, N_FRAMES,
    HALF_1_KICKOFF, HALF_1_END, HALF_2_START, HALF_2_KICKOFF, HALF_2_END,
    build_transition_matrix, viterbi_decode, debounce_spells, MIN_SPELL_FRAMES
)
from boundary_refinement import refine_transition_boundaries

WEIGHTS_FILE = os.path.join(HERE, "bundesliga_weights.json")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def compute_bundesliga_trained_emissions(feats, weights_dict, emit_scale=0.80):
    """Compute hierarchical emission probabilities using IDSSE-learned parameters."""
    print("Computing emissions using Bundesliga-trained weights...")
    
    delta_d = feats["delta_d_m"]
    d_h = feats["d_nearest_home_m"]
    d_a = feats["d_nearest_away_m"]
    k5_ratio = feats["k5_home_ratio"]
    out_pitch = feats["ball_out_of_pitch"]
    f_to_sp = feats["frames_to_next_setpiece"]
    in_open = feats["in_open_play_phase"]
    open_team = feats["open_play_team"]
    streak = feats["missing_streak_frames"]

    w_da = weights_dict["model_da"]["weights"]
    b_da = weights_dict["model_da"]["bias"]
    w_poss = weights_dict["model_poss"]["weights"]
    b_poss = weights_dict["model_poss"]["bias"]

    # 1. Stage 1: Dead vs Alive
    # Base learned logit from Bundesliga data
    dist_excess = np.maximum(0.0, np.minimum(d_h, d_a) - 3.0)
    z_dead_base = (
        b_da
        + w_da["ball_out_of_pitch"] * out_pitch
        + w_da["min_dist_excess_3m"] * dist_excess
    )
    # Add match tactical prior (Set Pieces and open play)
    z_dead = (
        z_dead_base
        + 3.5 * np.maximum(0, (180 - f_to_sp) / 100.0)
        - 3.0 * in_open
        + 1.2 * np.minimum(streak / 50.0, 5.0) * (1 - in_open)
    )
    p_dead = sigmoid(z_dead)
    p_dead = np.where(out_pitch == 1, 0.999, p_dead)
    p_alive = 1.0 - p_dead

    # 2. Stage 2: POSS (Home vs Away | Alive)
    # Base learned logit from Bundesliga data (delta_d is primary signal, scale = 1.5)
    w_delta = abs(w_poss["delta_d_m"]) * 4.0 # Scale to calibrated logit margin (~1.5)
    z_poss = (
        b_poss
        - w_delta * np.clip(delta_d, -8.0, 8.0)
        + 1.5 * (k5_ratio - 0.5)
        + 0.8 * np.where(open_team == 1, 1.0, np.where(open_team == 2, -1.0, 0.0))
    )
    p_h_alive = sigmoid(z_poss)
    p_a_alive = 1.0 - p_h_alive

    # 3. Joint Probabilities P(S_t | X_t)
    p_h = np.maximum(1e-6, p_alive * p_h_alive)
    p_a = np.maximum(1e-6, p_alive * p_a_alive)
    p_d = np.maximum(1e-6, p_dead)

    total = p_h + p_a + p_d
    p_h /= total
    p_a /= total
    p_d /= total

    log_emissions = np.stack([np.log(p_h), np.log(p_a), np.log(p_d)], axis=1) * emit_scale
    return log_emissions


def run_pipeline():
    print("=" * 70)
    print("Phase 5.2: Bundesliga-Trained HMM + Viterbi + Boundary Refinement")
    print("=" * 70)

    # 1. Load trained weights
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"Trained weights not found at {WEIGHTS_FILE}. Run train_on_idsse.py first.")
    with open(WEIGHTS_FILE, "r") as f:
        weights_dict = json.load(f)
    print(f"Loaded weights trained on {weights_dict['total_training_frames']:,} Bundesliga frames.")

    # 2. Load Barcelona features
    feats_path = os.path.join(ROOT, "features", "features.npz")
    feats = np.load(feats_path)
    d_h = feats["d_nearest_home_m"]
    d_a = feats["d_nearest_away_m"]
    ball_out = feats["ball_out_of_pitch"]
    ball_speed = feats["ball_speed_mps"]

    # 3. Compute emissions and transition matrix
    log_emissions = compute_bundesliga_trained_emissions(feats, weights_dict)
    log_A = build_transition_matrix()

    # 4. Allocate full match predictions
    final_preds = np.full(N_FRAMES, NOBODY, dtype=np.int8)

    # 5. Decode First Half
    print(f"\nDecoding First Half (frames {HALF_1_KICKOFF:,} to {HALF_1_END:,})...")
    h1_emits = log_emissions[HALF_1_KICKOFF:HALF_1_END]
    h1_path = viterbi_decode(h1_emits, log_A)
    h1_clean = debounce_spells(h1_path, MIN_SPELL_FRAMES)
    h1_refined = refine_transition_boundaries(
        h1_clean,
        d_h[HALF_1_KICKOFF:HALF_1_END],
        d_a[HALF_1_KICKOFF:HALF_1_END],
        ball_out[HALF_1_KICKOFF:HALF_1_END],
        ball_speed[HALF_1_KICKOFF:HALF_1_END]
    )
    final_preds[HALF_1_KICKOFF:HALF_1_END] = h1_refined

    # 6. Fixed match structure: 2nd half pre-kickoff is DEAD
    print(f"Assigning Second Half pre-kickoff: frames {HALF_2_START:,} to {HALF_2_KICKOFF:,} -> DEAD")
    final_preds[HALF_2_START:HALF_2_KICKOFF] = DEAD

    # 7. Decode Second Half
    print(f"Decoding Second Half (frames {HALF_2_KICKOFF:,} to {HALF_2_END:,})...")
    h2_emits = log_emissions[HALF_2_KICKOFF:HALF_2_END]
    h2_path = viterbi_decode(h2_emits, log_A)
    h2_clean = debounce_spells(h2_path, MIN_SPELL_FRAMES)
    h2_refined = refine_transition_boundaries(
        h2_clean,
        d_h[HALF_2_KICKOFF:HALF_2_END],
        d_a[HALF_2_KICKOFF:HALF_2_END],
        ball_out[HALF_2_KICKOFF:HALF_2_END],
        ball_speed[HALF_2_KICKOFF:HALF_2_END]
    )
    final_preds[HALF_2_KICKOFF:HALF_2_END] = h2_refined

    # 8. Write submission CSV
    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_bundesliga_trained.csv")

    out_labels = [LABEL_MAP[p] for p in final_preds]
    df_out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": out_labels})
    df_out.to_csv(out_path, index=False)

    print(f"\n[OK] Submission file saved to: {out_path}")
    print(f"Total rows: {len(df_out):,}")
    print("Label distribution in submission:")
    print(df_out["label"].value_counts().to_string())

    return out_path


if __name__ == "__main__":
    run_pipeline()
