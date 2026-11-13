"""Build a no-cleaning control feature bundle for the TCN ablation.

The control uses the original feature pipeline (``features/features.npz``) and
the raw ball positions. Cleaning-derived fields are neutralized: no Akima
micro-interpolation, no gap features, no last_seen reference and raw missing
semantics. The schema is the same 37-channel schema used by the TCN, so the
control can be trained and decoded with the exact same code.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402
DEFAULT_BASELINE = os.path.join(ROOT, "features", "features.npz")
DEFAULT_BALL_CLEAN = os.path.join(
    ROOT, "models", "akima_with_data_prep", "ball_clean.npz")
OUT_PATH = os.path.join(HERE, "features_noclean.npz")

PITCH_LEN, PITCH_WID = GAMES[GAME]["pitch"]


def build_control(baseline_path, ball_clean_path):
    with np.load(baseline_path) as handle:
        feats = {key: handle[key].copy() for key in handle.files}
    with np.load(ball_clean_path) as handle:
        ball = {key: handle[key].copy() for key in handle.files}

    n = len(ball["video_frame"])
    imputed = ball["ball_is_imputed"].astype(bool)
    observed = ball["ball_observed"].astype(np.uint8)
    missing = (observed == 0).astype(np.uint8)

    ball_x = ball["ball_x"].astype(np.float32).copy()
    ball_y = ball["ball_y"].astype(np.float32).copy()
    ball_x[imputed] = np.nan
    ball_y[imputed] = np.nan

    feats.update({
        "video_frame": np.arange(n, dtype=np.int64),
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_observed": observed,
        "ball_is_imputed": np.zeros(n, dtype=np.uint8),
        "ball_is_missing": missing,
        "is_ball_missing": missing.astype(np.int8),
        "ball_gap_len": np.zeros(n, dtype=np.uint16),
        "ball_gap_age": np.zeros(n, dtype=np.uint16),
        "ball_gap_remaining": np.zeros(n, dtype=np.uint16),
        "ball_was_raw_missing": missing,
        "last_seen_ball_x_m": np.full(
            n, PITCH_LEN / 2.0, dtype=np.float32),
        "last_seen_ball_y_m": np.full(
            n, PITCH_WID / 2.0, dtype=np.float32),
        "n_players_visible": np.zeros(n, dtype=np.uint16),
        "centroid_5nn_speed_mps": np.zeros(n, dtype=np.float32),
        "reference_source": np.full(n, 2, dtype=np.uint8),
    })
    return feats


def main():
    parser = argparse.ArgumentParser(
        description="Genera un control sin limpieza para el TCN.")
    parser.add_argument("--baseline-features", default=DEFAULT_BASELINE)
    parser.add_argument("--ball-clean", default=DEFAULT_BALL_CLEAN)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    feats = build_control(args.baseline_features, args.ball_clean)
    np.savez_compressed(args.out, **feats)
    print("Control sin limpieza: {}".format(os.path.abspath(args.out)))
    print("Frames: {} | observed: {} | missing: {}".format(
        len(feats["video_frame"]),
        int(feats["ball_observed"].sum()),
        int(feats["ball_is_missing"].sum()),
    ))


if __name__ == "__main__":
    main()
