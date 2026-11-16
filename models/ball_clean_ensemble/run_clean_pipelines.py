"""Build the TCN pseudo-labels from the cleaned ball track.

Decodes the HMM ensemble (same emissions and transition matrix as
``models/hmm_ensemble.py``, but without the dead-ball kinetic boost, which is
specific to the shipped ensemble submission) on ``features_clean_long_lastseen.npz``
and writes ``results/clean_ensemble.csv``. Those are the pseudo-labels the TCN
is trained on.

No ground truth is read anywhere in this module.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MODELS_DIR)

sys.path.insert(0, ROOT)
sys.path.insert(0, MODELS_DIR)

from hmm_viterbi import (  # noqa: E402
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
from hmm_viterbi_bundesliga import (  # noqa: E402
    compute_bundesliga_trained_emissions,
)
from hmm_ensemble import ALPHA, build_ensemble_transition_matrix  # noqa: E402
from boundary_refinement import refine_transition_boundaries  # noqa: E402

WEIGHTS_FILE = os.path.join(MODELS_DIR, "bundesliga_weights.json")
DEFAULT_CLEAN_FEATURES = os.path.join(
    HERE, "features_clean_long_lastseen.npz")
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")


def _load_weights():
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            "No se encontró {}. Ejecuta models/train_on_idsse.py.".format(
                WEIGHTS_FILE))
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _decode_half(feats, emissions, log_transition, start, end):
    path = viterbi_decode(emissions[start:end], log_transition)
    cleaned = debounce_spells(path, MIN_SPELL_FRAMES)

    d_h = feats["d_nearest_home_m"][start:end]
    d_a = feats["d_nearest_away_m"][start:end]
    ball_out = feats["ball_out_of_pitch"][start:end]
    ball_speed = feats["ball_speed_mps"][start:end]
    speed_safe = np.where(np.isfinite(ball_speed), ball_speed, np.inf)
    return refine_transition_boundaries(
        cleaned, d_h, d_a, ball_out, speed_safe)


def build_pseudo_labels(features_path, out_path, weights):
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            "No se encontró {}. Ejecuta build_features_clean.py.".format(
                features_path))

    with np.load(features_path) as feats:
        refined = compute_emission_log_probabilities(feats)
        trained = compute_bundesliga_trained_emissions(feats, weights)
        emissions = ALPHA * refined + (1.0 - ALPHA) * trained
        log_transition = build_ensemble_transition_matrix()

        final_preds = np.full(N_FRAMES, NOBODY, dtype=np.int8)
        final_preds[HALF_1_KICKOFF:HALF_1_END] = _decode_half(
            feats, emissions, log_transition,
            HALF_1_KICKOFF, HALF_1_END)

        final_preds[HALF_2_START:HALF_2_KICKOFF] = DEAD

        final_preds[HALF_2_KICKOFF:HALF_2_END] = _decode_half(
            feats, emissions, log_transition,
            HALF_2_KICKOFF, HALF_2_END)

    labels = [LABEL_MAP[int(value)] for value in final_preds]
    frame = pd.DataFrame({
        "frame": np.arange(N_FRAMES, dtype=np.int64),
        "label": labels,
    })
    frame.to_csv(out_path, index=False)

    counts = frame["label"].value_counts().reindex(
        ["H", "A", "D", "X"], fill_value=0)
    print("Features: {}".format(features_path))
    print("Salida:   {}".format(out_path))
    print("Etiquetas:")
    print(counts.to_string())
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Genera las pseudo-etiquetas del TCN con el ensemble HMM.")
    parser.add_argument("--clean-features", default=DEFAULT_CLEAN_FEATURES)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    weights = _load_weights()
    build_pseudo_labels(
        args.clean_features,
        os.path.join(args.results_dir, "clean_ensemble.csv"),
        weights,
    )
    print("\nListo.")


if __name__ == "__main__":
    main()
