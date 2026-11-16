"""Decode the averaged TCN logits with the frozen Viterbi settings.

The settings (scale 0.8, minimum spell 62 frames, no boundary refinement) were
fixed before the final submission and are not re-tuned here. The script writes
``evaluation/results/model_tcn_clean.csv``, one of the two inputs of the
consensus stage. The public-window score printed at the end is a diagnostic
only; it is not a selection criterion.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from hmm_ensemble import build_ensemble_transition_matrix  # noqa: E402
from hmm_viterbi import (  # noqa: E402
    DEAD,
    HALF_1_END,
    HALF_1_KICKOFF,
    HALF_2_END,
    HALF_2_KICKOFF,
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
LOGITS = os.path.join(HERE, "tcn_clean_logits_ensemble.npz")
WEIGHTS = os.path.join(MODELS_DIR, "bundesliga_weights.json")
GT_PATH = os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUTPUT = os.path.join(ROOT, "evaluation", "results", "model_tcn_clean.csv")
HALVES = ((HALF_1_KICKOFF, HALF_1_END), (HALF_2_KICKOFF, HALF_2_END))

TCN_WEIGHT = 1.0
MIN_SPELL = 62
SCALE = 0.8
P_STAY = 0.9950


def _load(logits_path, features_path):
    with open(WEIGHTS, "r", encoding="utf-8") as handle:
        weights = json.load(handle)
    with np.load(features_path) as feats:
        hand = compute_emission_log_probabilities(feats) / 0.80
        trained = compute_bundesliga_trained_emissions(feats, weights) / 0.80
        hmm = 0.35 * hand + 0.65 * trained
    with np.load(logits_path) as bundle:
        logits = bundle["logits"].astype(np.float64)
    tcn = torch.log_softmax(torch.from_numpy(logits), dim=1).numpy()
    return hmm, tcn


def _decode(emissions):
    final = np.full(N_FRAMES, NOBODY, dtype=np.int8)
    log_transition = build_ensemble_transition_matrix(P_STAY)
    for start, stop in HALVES:
        path = viterbi_decode(emissions[start:stop], log_transition)
        final[start:stop] = debounce_spells(path, MIN_SPELL)
    final[HALF_1_END:HALF_2_KICKOFF] = DEAD
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logits", default=LOGITS)
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--out", default=OUTPUT)
    args = parser.parse_args()

    hmm, tcn = _load(args.logits, args.features)
    emissions = ((1.0 - TCN_WEIGHT) * hmm + TCN_WEIGHT * tcn) * SCALE
    labels = _decode(emissions)

    frame = pd.DataFrame({
        "frame": np.arange(N_FRAMES, dtype=np.int64),
        "label": [LABEL_MAP[int(value)] for value in labels],
    })
    frame.to_csv(args.out, index=False)
    print("Wrote {}".format(args.out))

    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    result = pm.evaluate(gt, labels.astype(np.int64))
    print("Diagnostic public window: spell={:.4f} macro={:.4f} trans25={:.4f} "
          "poss={:2d}".format(
              result["spell_f1_mean"],
              result["macro_f1"],
              result["transitions"][25]["f1"],
              result["n_pred_spells"]))


if __name__ == "__main__":
    main()
