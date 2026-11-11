import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from boundary_refinement import refine_transition_boundaries
from hmm_viterbi import (
    DEAD,
    HALF_1_END,
    HALF_1_KICKOFF,
    HALF_2_END,
    HALF_2_KICKOFF,
    HALF_2_START,
    LABEL_MAP,
    MIN_SPELL_FRAMES,
    N_FRAMES,
    NOBODY,
    compute_emission_log_probabilities,
    debounce_spells,
    viterbi_decode,
)
from hmm_viterbi_bundesliga import compute_bundesliga_trained_emissions

WEIGHTS_FILE = os.path.join(HERE, "bundesliga_weights.json")
ALPHA = 0.35
P_STAY = 0.9950

def build_ensemble_transition_matrix(p_stay=P_STAY):
    """Build the logarithmic state transition matrix for the ensemble model.

    Args:
        p_stay (float): Probability of self-transition (remaining in the current state).

    Returns:
        np.ndarray: Log-transition probability matrix of shape (3, 3).
    """
    A = np.zeros((3, 3), dtype=np.float64)
    A[0, 0] = p_stay
    A[0, 1] = 0.0004
    A[0, 2] = 1.0 - A[0, 0] - A[0, 1]

    A[1, 1] = p_stay
    A[1, 0] = 0.0004
    A[1, 2] = 1.0 - A[1, 0] - A[1, 1]

    A[2, 2] = p_stay * 0.995
    rem = (1.0 - A[2, 2]) / 2.0
    A[2, 0] = rem
    A[2, 1] = rem

    return np.log(np.maximum(1e-12, A))

def run_ensemble_pipeline():
    """Execute the unified Bayesian ensemble inference pipeline.

    Returns:
        str: File path to the generated submission CSV.
    """
    print("=" * 70)
    print("Running Ensemble Pipeline...")
    print("=" * 70)

    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"Trained weights not found at {WEIGHTS_FILE}.")
    with open(WEIGHTS_FILE, "r") as f:
        weights_dict = json.load(f)

    feats_path = os.path.join(ROOT, "features", "features.npz")
    feats = np.load(feats_path)

    d_h = feats["d_nearest_home_m"]
    d_a = feats["d_nearest_away_m"]
    ball_out = feats["ball_out_of_pitch"]
    ball_speed = feats["ball_speed_mps"]

    print(f"Computing refined physical emissions...")
    emits_refined = compute_emission_log_probabilities(feats)

    print(f"Computing Bundesliga-trained emissions...")
    emits_bundesliga = compute_bundesliga_trained_emissions(feats, weights_dict)

    print(f"Blending emission log-distributions (alpha={ALPHA:.2f})...")
    log_emissions = ALPHA * emits_refined + (1.0 - ALPHA) * emits_bundesliga
    log_A = build_ensemble_transition_matrix()

    final_preds = np.full(N_FRAMES, NOBODY, dtype=np.int8)

    print(f"\nDecoding First Half: frames {HALF_1_KICKOFF:,} to {HALF_1_END:,}...")
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

    print(f"Assigning Second Half pre-kickoff: frames {HALF_2_START:,} to {HALF_2_KICKOFF:,} -> DEAD")
    final_preds[HALF_2_START:HALF_2_KICKOFF] = DEAD

    print(f"Decoding Second Half: frames {HALF_2_KICKOFF:,} to {HALF_2_END:,}...")
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

    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_ensemble.csv")

    out_labels = [LABEL_MAP[p] for p in final_preds]
    df_out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": out_labels})
    df_out.to_csv(out_path, index=False)

    print(f"\n[OK] Unified ensemble submission saved to: {out_path}")
    print(f"Total rows: {len(df_out):,}")
    print("Label distribution in submission:")
    print(df_out["label"].value_counts().to_string())

    return out_path

if __name__ == "__main__":
    run_ensemble_pipeline()
