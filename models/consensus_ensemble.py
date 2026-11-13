"""Consensus Ensemble for Ball Possession Detection.

Combines causal temporal sequence modeling from the TCN with Bayesian HMM
dead-ball kinetic rest filtering and dual boundary snapping.
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

from boundary_refinement import refine_transition_boundaries
from hmm_ensemble import run_ensemble_pipeline
from hmm_viterbi import (
    DEAD,
    HALF_1_END,
    HALF_1_KICKOFF,
    HALF_2_END,
    HALF_2_KICKOFF,
    LABEL_MAP,
    N_FRAMES,
    debounce_spells,
)

TCN_RESULTS_PATH = os.path.join(ROOT, "evaluation", "results", "model_tcn_clean.csv")
ENSEMBLE_RESULTS_PATH = os.path.join(ROOT, "evaluation", "results", "model_ensemble.csv")
OUTPUT_PATH = os.path.join(ROOT, "evaluation", "results", "model_consensus.csv")

DEBOUNCE_MIN_FRAMES = 25


WINDOW_LIVE = 20


def read_predictions(file_path):
    """Read a submission CSV file and return the integer label array.

    Args:
        file_path (str): Path to the submission CSV file.

    Returns:
        np.ndarray: 1D array of categorical label codes.
    """
    reverse_map = {v: k for k, v in LABEL_MAP.items()}
    df = pd.read_csv(file_path)
    return np.array([reverse_map[x] for x in df["label"].values], dtype=np.int8)


def build_consensus_predictions():
    """Build consensus predictions between TCN and Bayesian HMM.

    Returns:
        np.ndarray: Refined 1D prediction array of length N_FRAMES.
    """
    if not os.path.exists(TCN_RESULTS_PATH):
        raise FileNotFoundError(f"TCN predictions not found at {TCN_RESULTS_PATH}.")

    if not os.path.exists(ENSEMBLE_RESULTS_PATH):
        run_ensemble_pipeline()

    tcn_labels = read_predictions(TCN_RESULTS_PATH)
    ens_labels = read_predictions(ENSEMBLE_RESULTS_PATH)

    feats = np.load(os.path.join(ROOT, "features", "features.npz"))
    d_h = feats["d_nearest_home_m"]
    d_a = feats["d_nearest_away_m"]
    ball_out = feats["ball_out_of_pitch"]
    ball_speed = feats["ball_speed_mps"]

    consensus = tcn_labels.copy()
    dead_mask = (ens_labels == DEAD) & ((tcn_labels == 0) | (tcn_labels == 1))
    consensus[dead_mask] = DEAD

    for start, end in [(HALF_1_KICKOFF, HALF_1_END), (HALF_2_KICKOFF, HALF_2_END)]:
        slice_d_h = d_h[start:end]
        slice_d_a = d_a[start:end]
        slice_out = ball_out[start:end]
        slice_spd = ball_speed[start:end]

        stage1 = refine_transition_boundaries(
            consensus[start:end],
            slice_d_h,
            slice_d_a,
            slice_out,
            slice_spd,
            window_live=WINDOW_LIVE,
        )
        stage2 = debounce_spells(stage1, DEBOUNCE_MIN_FRAMES)
        stage3 = refine_transition_boundaries(
            stage2,
            slice_d_h,
            slice_d_a,
            slice_out,
            slice_spd,
            window_live=WINDOW_LIVE,
        )
        consensus[start:end] = stage3

    return consensus


def run_consensus_pipeline():
    """Execute full consensus inference and write submission CSV.

    Returns:
        str: Output file path.
    """
    print("=" * 70)
    print("Running Consensus Ensemble Pipeline...")
    print("=" * 70)

    consensus = build_consensus_predictions()
    out_labels = [LABEL_MAP[val] for val in consensus]
    df_out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": out_labels})
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[OK] Consensus submission saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df_out):,}")
    print("Label distribution:")
    print(df_out["label"].value_counts().to_string())

    return OUTPUT_PATH


if __name__ == "__main__":
    run_consensus_pipeline()
