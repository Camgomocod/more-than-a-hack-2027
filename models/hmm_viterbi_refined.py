"""Hierarchical HMM + Viterbi Decoding + Kinematic Boundary Refinement (Phase 5.1).
Integrates macroeconomic Markov decoding with microsecond kinematic snapping.
"""

import os
import sys
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
    compute_emission_log_probabilities, build_transition_matrix,
    viterbi_decode, debounce_spells, MIN_SPELL_FRAMES
)
from boundary_refinement import refine_transition_boundaries


def run_refined_pipeline():
    print("=" * 70)
    print("Phase 5.1: Hierarchical HMM + Viterbi + Boundary Refinement Pipeline")
    print("=" * 70)

    # 1. Load features
    feats_path = os.path.join(ROOT, "features", "features.npz")
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Features file not found at {feats_path}. Run features/extract_features.py first.")

    print(f"Loading features from {os.path.basename(feats_path)}...")
    feats = np.load(feats_path)

    d_h = feats["d_nearest_home_m"]
    d_a = feats["d_nearest_away_m"]
    ball_out = feats["ball_out_of_pitch"]
    ball_speed = feats["ball_speed_mps"]

    # 2. Compute emissions & transition matrix
    log_emissions = compute_emission_log_probabilities(feats)
    log_A = build_transition_matrix()

    # 3. Allocate full match predictions
    final_preds = np.full(N_FRAMES, NOBODY, dtype=np.int8)

    # 4. Decode First Half
    print(f"\nDecoding First Half: frames {HALF_1_KICKOFF:,} to {HALF_1_END:,}...")
    h1_emits = log_emissions[HALF_1_KICKOFF:HALF_1_END]
    h1_path = viterbi_decode(h1_emits, log_A)
    h1_clean = debounce_spells(h1_path, MIN_SPELL_FRAMES)
    # Apply kinematic refinement
    h1_refined = refine_transition_boundaries(
        h1_clean,
        d_h[HALF_1_KICKOFF:HALF_1_END],
        d_a[HALF_1_KICKOFF:HALF_1_END],
        ball_out[HALF_1_KICKOFF:HALF_1_END],
        ball_speed[HALF_1_KICKOFF:HALF_1_END]
    )
    final_preds[HALF_1_KICKOFF:HALF_1_END] = h1_refined

    # 5. Fixed match structure: 2nd half pre-kickoff is DEAD
    print(f"Assigning Second Half pre-kickoff: frames {HALF_2_START:,} to {HALF_2_KICKOFF:,} -> DEAD")
    final_preds[HALF_2_START:HALF_2_KICKOFF] = DEAD

    # 6. Decode Second Half
    print(f"Decoding Second Half: frames {HALF_2_KICKOFF:,} to {HALF_2_END:,}...")
    h2_emits = log_emissions[HALF_2_KICKOFF:HALF_2_END]
    h2_path = viterbi_decode(h2_emits, log_A)
    h2_clean = debounce_spells(h2_path, MIN_SPELL_FRAMES)
    # Apply kinematic refinement
    h2_refined = refine_transition_boundaries(
        h2_clean,
        d_h[HALF_2_KICKOFF:HALF_2_END],
        d_a[HALF_2_KICKOFF:HALF_2_END],
        ball_out[HALF_2_KICKOFF:HALF_2_END],
        ball_speed[HALF_2_KICKOFF:HALF_2_END]
    )
    final_preds[HALF_2_KICKOFF:HALF_2_END] = h2_refined

    # 7. Write submission CSV
    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_viterbi_refined.csv")

    out_labels = [LABEL_MAP[p] for p in final_preds]
    df_out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": out_labels})
    df_out.to_csv(out_path, index=False)

    print(f"\n[OK] Refined submission file saved to: {out_path}")
    print(f"Total rows: {len(df_out):,}")
    print("Label distribution in submission:")
    print(df_out["label"].value_counts().to_string())

    return out_path


if __name__ == "__main__":
    run_refined_pipeline()
