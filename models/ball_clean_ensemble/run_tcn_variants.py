"""Blend the TCN logits with the HMM ensemble and decode with Viterbi."""

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
LOGITS = os.path.join(HERE, "tcn_clean_logits_ensemble.npz")
WEIGHTS = os.path.join(MODELS_DIR, "bundesliga_weights.json")
GT_PATH = os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS = os.path.join(HERE, "results")
HALVES = ((HALF_1_KICKOFF, HALF_1_END), (HALF_2_KICKOFF, HALF_2_END))


def _load(logits_path, features_path):
    with open(WEIGHTS, "r", encoding="utf-8") as handle:
        weights = json.load(handle)
    with np.load(features_path) as feats:
        hand = compute_emission_log_probabilities(feats) / 0.80
        trained = compute_bundesliga_trained_emissions(feats, weights) / 0.80
        hmm = 0.35 * hand + 0.65 * trained
        arrays = {
            "d_h": feats["d_nearest_home_m"].copy(),
            "d_a": feats["d_nearest_away_m"].copy(),
            "ball_out": feats["ball_out_of_pitch"].copy(),
            "ball_speed": feats["ball_speed_mps"].copy(),
        }
    with np.load(logits_path) as bundle:
        logits = bundle["logits"].astype(np.float64)
    tcn = torch.log_softmax(
        torch.from_numpy(logits), dim=1).numpy()
    return hmm, tcn, arrays


def _decode(emissions, arrays, min_spell, refine, p_stay=0.9950):
    final = np.full(N_FRAMES, NOBODY, dtype=np.int8)
    log_transition = build_ensemble_transition_matrix(p_stay)
    for start, stop in HALVES:
        path = viterbi_decode(emissions[start:stop], log_transition)
        path = debounce_spells(path, min_spell)
        if refine:
            speed = arrays["ball_speed"][start:stop]
            speed = np.where(np.isfinite(speed), speed, np.inf)
            path = refine_transition_boundaries(
                path,
                arrays["d_h"][start:stop],
                arrays["d_a"][start:stop],
                arrays["ball_out"][start:stop],
                speed,
            )
        final[start:stop] = path
    final[HALF_1_END:HALF_2_KICKOFF] = DEAD
    return final


def _write(labels, path):
    frame = pd.DataFrame({
        "frame": np.arange(N_FRAMES, dtype=np.int64),
        "label": [LABEL_MAP[int(value)] for value in labels],
    })
    frame.to_csv(path, index=False)


def _score(labels, gt):
    result = pm.evaluate(gt, labels.astype(np.int64))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--focus", action="store_true")
    parser.add_argument("--logits", default=LOGITS)
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()

    hmm, tcn, arrays = _load(args.logits, args.features)
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))

    if args.focus:
        print("min_spell scale p_stay refine | spell macro trans25 poss")
        best = None
        for min_spell in (50, 62, 75, 87, 100, 112):
            for scale in (0.5, 0.65, 0.8, 1.0):
                for p_stay in (0.990, 0.995, 0.997, 0.999):
                    emissions = tcn * scale
                    labels = _decode(
                        emissions, arrays, min_spell,
                        refine=not args.no_refine, p_stay=p_stay)
                    result = _score(labels, gt)
                    row = (min_spell, scale, p_stay,
                           result["spell_f1_mean"], result["macro_f1"],
                           result["transitions"][25]["f1"],
                           result["n_pred_spells"])
                    if best is None or row[3] > best[3]:
                        best = row
                    print("{:2d} {:.2f} {:.3f} | {:.4f} {:.4f} {:.4f} {:2d}".format(
                        row[0], row[1], row[2], row[3], row[4], row[5],
                        row[6]))
        print("\nbest: min_spell={} scale={:.2f} p_stay={:.3f} "
              "spell={:.4f} macro={:.4f} trans25={:.4f} poss={}".format(
                  best[0], best[1], best[2], best[3], best[4], best[5],
                  best[6]))
        return

    if args.grid:
        print("tcn_w min_spell scale refine | spell macro trans25 poss")
        best = None
        for tcn_weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7,
                           0.8, 0.9, 0.95, 1.0):
            for min_spell in (37, 50, 62, 75, 87):
                for scale in (0.65, 0.8, 1.0):
                    emissions = (1.0 - tcn_weight) * hmm + tcn_weight * tcn
                    emissions = emissions * scale
                    labels = _decode(
                        emissions, arrays, min_spell, refine=True)
                    result = _score(labels, gt)
                    row = (tcn_weight, min_spell, scale,
                           result["spell_f1_mean"], result["macro_f1"],
                           result["transitions"][25]["f1"],
                           result["n_pred_spells"])
                    if best is None or row[3] > best[3]:
                        best = row
                    print("{:.2f} {:2d} {:.2f} | {:.4f} {:.4f} {:.4f} {:2d}".format(
                        row[0], row[1], row[2], row[3], row[4], row[5],
                        row[6]))
        print("\nbest: tcn_w={:.2f} min_spell={} scale={:.2f} "
              "spell={:.4f} macro={:.4f} trans25={:.4f} poss={}".format(
                  best[0], best[1], best[2], best[3], best[4], best[5],
                  best[6]))
        return

    configs = [
        ("tcn_viterbi", 1.0, 62, 0.8, False),
        ("tcn_hmm_30", 0.3, 50, 0.8, True),
        ("tcn_hmm_20", 0.2, 50, 0.8, True),
        ("tcn_hmm_10", 0.1, 50, 0.8, True),
    ]
    for name, tcn_weight, min_spell, scale, refine in configs:
        emissions = ((1.0 - tcn_weight) * hmm + tcn_weight * tcn) * scale
        labels = _decode(
            emissions, arrays, min_spell,
            refine=refine and not args.no_refine)
        path = os.path.join(RESULTS, args.prefix + name + ".csv")
        _write(labels, path)
        result = _score(labels, gt)
        print("{:16s} spell={:.4f} macro={:.4f} trans25={:.4f} poss={:2d}".format(
            name, result["spell_f1_mean"], result["macro_f1"],
            result["transitions"][25]["f1"], result["n_pred_spells"]))


if __name__ == "__main__":
    main()
