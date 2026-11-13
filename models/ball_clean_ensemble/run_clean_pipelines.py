"""Run the existing HMM ensemble variants on cleaned features.

The colleague's files are imported, not modified. This module only reproduces
their orchestration with configurable feature and output paths.
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
    build_transition_matrix,
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
DEFAULT_BASELINE_FEATURES = os.path.join(ROOT, "features", "features.npz")
DEFAULT_CLEAN_FEATURES = os.path.join(
    HERE, "features_clean_long_lastseen.npz")
DEFAULT_RAWMISSING_FEATURES = os.path.join(
    HERE, "features_clean_rawmissing.npz")
DEFAULT_CENTROIDREF_FEATURES = os.path.join(
    HERE, "features_clean_centroidref.npz")
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")


def _section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _load_weights():
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            "No se encontró {}. Ejecuta models/train_on_idsse.py.".format(
                WEIGHTS_FILE))
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _emissions(feats, variant, weights):
    if variant == "viterbi":
        return (
            compute_emission_log_probabilities(feats),
            build_transition_matrix(),
            False,
        )
    if variant == "bundesliga":
        return (
            compute_bundesliga_trained_emissions(feats, weights),
            build_transition_matrix(),
            True,
        )
    if variant == "ensemble":
        refined = compute_emission_log_probabilities(feats)
        trained = compute_bundesliga_trained_emissions(feats, weights)
        return (
            ALPHA * refined + (1.0 - ALPHA) * trained,
            build_ensemble_transition_matrix(),
            True,
        )
    raise ValueError("variante desconocida: {}".format(variant))


def _decode_half(feats, emissions, log_transition, start, end, refine):
    path = viterbi_decode(emissions[start:end], log_transition)
    cleaned = debounce_spells(path, MIN_SPELL_FRAMES)
    if not refine:
        return cleaned

    d_h = feats["d_nearest_home_m"][start:end]
    d_a = feats["d_nearest_away_m"][start:end]
    ball_out = feats["ball_out_of_pitch"][start:end]
    ball_speed = feats["ball_speed_mps"][start:end]
    speed_safe = np.where(np.isfinite(ball_speed), ball_speed, np.inf)
    return refine_transition_boundaries(
        cleaned, d_h, d_a, ball_out, speed_safe)


def _run_variant(features_path, variant, out_path, weights):
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            "No se encontró {}. Ejecuta build_features_clean.py.".format(
                features_path))

    with np.load(features_path) as feats:
        emissions, log_transition, refine = _emissions(
            feats, variant, weights)

        final_preds = np.full(N_FRAMES, NOBODY, dtype=np.int8)
        final_preds[HALF_1_KICKOFF:HALF_1_END] = _decode_half(
            feats, emissions, log_transition,
            HALF_1_KICKOFF, HALF_1_END, refine)

        final_preds[HALF_2_START:HALF_2_KICKOFF] = DEAD

        final_preds[HALF_2_KICKOFF:HALF_2_END] = _decode_half(
            feats, emissions, log_transition,
            HALF_2_KICKOFF, HALF_2_END, refine)

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


def _check_baseline_reproduction(out_path):
    reference = os.path.join(
        ROOT, "evaluation", "results", "model_ensemble.csv")
    if not os.path.exists(reference):
        print("Referencia baseline no encontrada; se omite la comparación.")
        return

    current = pd.read_csv(out_path)
    expected = pd.read_csv(reference)
    same_shape = len(current) == len(expected)
    same_labels = same_shape and bool(
        (current["label"].values == expected["label"].values).all())
    same_frames = same_shape and bool(
        (current["frame"].values == expected["frame"].values).all())
    if same_labels and same_frames:
        print("Reproducción del baseline: OK (idéntico a model_ensemble.csv)")
    else:
        print("Reproducción del baseline: DIFIERE de model_ensemble.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta las variantes HMM sobre features limpias.")
    parser.add_argument(
        "--baseline-features", default=DEFAULT_BASELINE_FEATURES)
    parser.add_argument(
        "--clean-features", default=DEFAULT_CLEAN_FEATURES)
    parser.add_argument(
        "--rawmissing-features", default=DEFAULT_RAWMISSING_FEATURES)
    parser.add_argument(
        "--centroidref-features", default=DEFAULT_CENTROIDREF_FEATURES)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    weights = _load_weights()

    if not args.skip_baseline:
        _section("Sección 0 — Baseline reproducible (features antiguas)")
        baseline_path = os.path.join(
            args.results_dir, "baseline_ensemble.csv")
        _run_variant(
            args.baseline_features, "ensemble", baseline_path, weights)
        _check_baseline_reproduction(baseline_path)

    _section("Sección 1 — Clean Viterbi (long + last_seen)")
    _run_variant(
        args.clean_features, "viterbi",
        os.path.join(args.results_dir, "clean_viterbi.csv"), weights)

    _section("Sección 2 — Clean Bundesliga (long + last_seen)")
    _run_variant(
        args.clean_features, "bundesliga",
        os.path.join(args.results_dir, "clean_bundesliga.csv"), weights)

    _section("Sección 3 — Clean Ensemble (long + last_seen)")
    _run_variant(
        args.clean_features, "ensemble",
        os.path.join(args.results_dir, "clean_ensemble.csv"), weights)

    _section("Sección 4 — Ablación missing-mode raw (ensemble)")
    _run_variant(
        args.rawmissing_features, "ensemble",
        os.path.join(args.results_dir, "clean_ensemble_rawmissing.csv"),
        weights)

    _section("Sección 5 — Ablación reference-mode centroid (ensemble)")
    _run_variant(
        args.centroidref_features, "ensemble",
        os.path.join(args.results_dir, "clean_ensemble_centroidref.csv"),
        weights)

    print("\nTodas las variantes terminaron.")


if __name__ == "__main__":
    main()
