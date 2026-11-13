"""Coordinate search over the clean HMM/ensemble post-processing knobs.

This is a diagnostic tool. The public window is only 17 possessions, so a gain
smaller than the bootstrap band is not a result.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from boundary_refinement import refine_transition_boundaries  # noqa: E402
from hmm_ensemble import build_ensemble_transition_matrix  # noqa: E402
from hmm_viterbi import (  # noqa: E402
    DEAD,
    HALF_1_END,
    HALF_1_KICKOFF,
    HALF_2_END,
    HALF_2_KICKOFF,
    HALF_2_START,
    LABEL_MAP,
    N_FRAMES,
    NOBODY,
    compute_emission_log_probabilities,
    debounce_spells,
    viterbi_decode,
)
from hmm_viterbi_bundesliga import (  # noqa: E402
    compute_bundesliga_trained_emissions,
)

FEATURES = os.path.join(HERE, "features_clean_long_lastseen.npz")
WEIGHTS = os.path.join(MODELS_DIR, "bundesliga_weights.json")
GT_PATH = os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_MIN_SPELL = 50


def _load_emissions():
    with open(WEIGHTS, "r", encoding="utf-8") as handle:
        weights = json.load(handle)
    with np.load(FEATURES) as feats:
        hand = compute_emission_log_probabilities(feats) / 0.80
        trained = compute_bundesliga_trained_emissions(feats, weights) / 0.80
        arrays = {
            "d_h": feats["d_nearest_home_m"].copy(),
            "d_a": feats["d_nearest_away_m"].copy(),
            "ball_out": feats["ball_out_of_pitch"].copy(),
            "ball_speed": feats["ball_speed_mps"].copy(),
        }
    return hand, trained, arrays


def _decode(hand, trained, arrays, alpha, emit_scale, p_stay,
            min_spell, refine):
    emissions = (alpha * hand + (1.0 - alpha) * trained) * emit_scale
    log_transition = build_ensemble_transition_matrix(p_stay)
    final = np.full(N_FRAMES, NOBODY, dtype=np.int8)

    for start, end, is_dead in (
        (HALF_1_KICKOFF, HALF_1_END, False),
        (HALF_2_START, HALF_2_KICKOFF, True),
        (HALF_2_KICKOFF, HALF_2_END, False),
    ):
        if is_dead:
            final[start:end] = DEAD
            continue
        path = viterbi_decode(emissions[start:end], log_transition)
        path = debounce_spells(path, min_spell)
        if refine:
            speed = arrays["ball_speed"][start:end]
            speed = np.where(np.isfinite(speed), speed, np.inf)
            path = refine_transition_boundaries(
                path,
                arrays["d_h"][start:end],
                arrays["d_a"][start:end],
                arrays["ball_out"][start:end],
                speed,
            )
        final[start:end] = path
    return final


def _score(final, gt):
    pred = final.astype(np.int64)
    result = pm.evaluate(gt, pred)
    return result


def _print_row(label, result):
    print("{:52s} spell={:.4f} macro={:.4f} trans25={:.4f} "
          "poss={:2d} dur={:5.1f} frag={:.2f}".format(
              label,
              result["spell_f1_mean"],
              result["macro_f1"],
              result["transitions"][25]["f1"],
              result["n_pred_spells"],
              result["mean_spell_duration_s"],
              result["fragmentation"],
          ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="min_spell")
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    hand, trained, arrays = _load_emissions()

    base = {
        "alpha": 0.35,
        "emit_scale": 0.80,
        "p_stay": 0.9950,
        "min_spell": DEFAULT_MIN_SPELL,
        "refine": True,
    }
    grids = {
        "min_spell": [12, 18, 25, 37, 50, 62, 75, 100],
        "p_stay": [0.985, 0.990, 0.993, 0.995, 0.997, 0.998, 0.999],
        "alpha": [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0],
        "emit_scale": [0.5, 0.65, 0.8, 1.0, 1.2, 1.5],
        "refine": [True, False],
    }

    _print_row("base", _score(_decode(hand, trained, arrays, **base), gt))
    for key, values in grids.items():
        print("\n--- {} ---".format(key))
        for value in values:
            config = dict(base)
            config[key] = value
            result = _score(
                _decode(hand, trained, arrays, **config), gt)
            _print_row("{}={}".format(key, value), result)


if __name__ == "__main__":
    main()
