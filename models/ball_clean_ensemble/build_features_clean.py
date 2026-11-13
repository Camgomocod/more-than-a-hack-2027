"""Build feature bundles that consume the cleaned ball track.

This module does not modify any of the existing model files. It imports the
team mapping and Smart Tagging helpers from features/extract_features.py and
writes only inside models/ball_clean_ensemble/.

The same streaming pass produces three feature bundles:

    features_clean_long_lastseen.npz   long missing + last_seen reference
    features_clean_rawmissing.npz      raw missing + last_seen reference
    features_clean_centroidref.npz     long missing + centroid reference

The difference between last_seen and centroid is only applied while the ball
is missing: last_seen uses the last observed ball position as the reference,
while centroid first selects the five players closest to that position, takes
their centroid, and uses that centroid as the reference for distances.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MODELS_DIR)

sys.path.insert(0, ROOT)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, os.path.join(ROOT, "features"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, require  # noqa: E402
import extract_features as ef  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = 25.0
DT = 1.0 / FPS
PITCH_LEN, PITCH_WID = GAMES[GAME]["pitch"]
HOME = 0
NO_TEAM = -1

MODES = ("long_lastseen", "rawmissing", "centroidref")
EXTRA_MODES = ("noclean",)
DEFAULT_BALL_CLEAN = os.path.join(
    ROOT, "models", "akima_with_data_prep", "ball_clean.npz")
DEFAULT_OUT_DIR = HERE


def _as_f32(values):
    return np.asarray(values, dtype=np.float32)


def _speed_from_positions(ball_x, ball_y, finite):
    """Fallback speed in m/s from cleaned positions when the ATD speed is NaN.

    The old feature extractor did the same for observed frames and left speed
    at zero on missing frames. Keeping that convention makes the HMM
    comparison fair.
    """
    n = len(ball_x)
    speed = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return speed

    index = np.arange(n)
    left = np.clip(index - 1, 0, n - 1)
    right = np.clip(index + 1, 0, n - 1)
    has_left = (index > 0) & finite[left]
    has_right = (index < n - 1) & finite[right]

    both = has_left & has_right
    dx = (ball_x[right].astype(np.float64) - ball_x[left]) * PITCH_LEN
    dy = (ball_y[right].astype(np.float64) - ball_y[left]) * PITCH_WID
    speed[both] = np.hypot(dx[both], dy[both]) / (2.0 * DT)

    only_right = has_right & ~has_left
    dx = (ball_x[right].astype(np.float64) - ball_x[index]) * PITCH_LEN
    dy = (ball_y[right].astype(np.float64) - ball_y[index]) * PITCH_WID
    speed[only_right] = np.hypot(dx[only_right], dy[only_right]) / DT

    only_left = has_left & ~has_right
    dx = (ball_x[index].astype(np.float64) - ball_x[left]) * PITCH_LEN
    dy = (ball_y[index].astype(np.float64) - ball_y[left]) * PITCH_WID
    speed[only_left] = np.hypot(dx[only_left], dy[only_left]) / DT

    return speed


def _ball_derived(ball):
    """Speed, acceleration, out-of-pitch and boundary distance arrays."""
    ball_x = _as_f32(ball["ball_x"])
    ball_y = _as_f32(ball["ball_y"])
    raw_speed = _as_f32(ball["ball_speed_mps"])
    finite = np.isfinite(ball_x) & np.isfinite(ball_y)

    calculated = _speed_from_positions(ball_x, ball_y, finite)
    speed = np.where(np.isfinite(raw_speed), raw_speed, calculated)
    speed = np.where(finite, speed, 0.0)
    speed = np.where(np.isfinite(speed), speed, 0.0).astype(np.float32)

    n = len(ball_x)
    accel = np.full(n, np.nan, dtype=np.float32)
    if n >= 3:
        both = np.isfinite(speed[:-2]) & np.isfinite(speed[2:])
        accel[1:-1] = np.where(
            both, (speed[2:] - speed[:-2]) / (2.0 * DT), np.nan)
    out_of_pitch = np.zeros(n, dtype=np.int8)
    dist_to_boundary = np.full(n, np.nan, dtype=np.float32)

    if finite.any():
        outside = finite & (
            (ball_x < 0.0) | (ball_x > 1.0) | (ball_y < 0.0) | (ball_y > 1.0))
        out_of_pitch[outside] = 1

        bx_m = ball_x.astype(np.float64) * PITCH_LEN
        by_m = ball_y.astype(np.float64) * PITCH_WID
        dx = np.minimum(np.maximum(0.0, bx_m),
                        np.maximum(0.0, PITCH_LEN - bx_m))
        dy = np.minimum(np.maximum(0.0, by_m),
                        np.maximum(0.0, PITCH_WID - by_m))
        dist_to_boundary[finite] = np.minimum(dx, dy)[finite].astype(np.float32)

    return speed, accel, out_of_pitch, dist_to_boundary, finite


def _empty_arrays(n):
    """Feature arrays shared by every mode."""
    return {
        "d_h_L": np.full(n, 50.0, dtype=np.float32),
        "d_a_L": np.full(n, 50.0, dtype=np.float32),
        "k5h_L": np.full(n, 50.0, dtype=np.float32),
        "k5a_L": np.full(n, 50.0, dtype=np.float32),
        "d_h_C": np.full(n, 50.0, dtype=np.float32),
        "d_a_C": np.full(n, 50.0, dtype=np.float32),
        "k5h_C": np.full(n, 50.0, dtype=np.float32),
        "k5a_C": np.full(n, 50.0, dtype=np.float32),
        "k5_ratio": np.full(n, 0.5, dtype=np.float32),
        "centroid_x": np.full(n, PITCH_LEN / 2.0, dtype=np.float32),
        "centroid_y": np.full(n, PITCH_WID / 2.0, dtype=np.float32),
        "centroid_speed": np.full(n, np.nan, dtype=np.float32),
        "reference_source": np.zeros(n, dtype=np.uint8),
        "n_players": np.zeros(n, dtype=np.uint16),
    }


def _reference_points(ball, finite_ball, n):
    """Reference point in metres for every frame, plus last_seen in metres."""
    ball_x = _as_f32(ball["ball_x"]).astype(np.float64)
    ball_y = _as_f32(ball["ball_y"]).astype(np.float64)
    last_x = _as_f32(ball["last_seen_ball_x"]).astype(np.float64)
    last_y = _as_f32(ball["last_seen_ball_y"]).astype(np.float64)

    ref_x = np.full(n, PITCH_LEN / 2.0, dtype=np.float64)
    ref_y = np.full(n, PITCH_WID / 2.0, dtype=np.float64)
    source = np.full(n, 2, dtype=np.uint8)

    finite_last = np.isfinite(last_x) & np.isfinite(last_y)
    ref_x[finite_last] = last_x[finite_last] * PITCH_LEN
    ref_y[finite_last] = last_y[finite_last] * PITCH_WID
    source[finite_last] = 1

    ref_x[finite_ball] = ball_x[finite_ball] * PITCH_LEN
    ref_y[finite_ball] = ball_y[finite_ball] * PITCH_WID
    source[finite_ball] = 0

    last_seen_x_m = np.where(finite_last, last_x * PITCH_LEN, np.nan)
    last_seen_y_m = np.where(finite_last, last_y * PITCH_WID, np.nan)
    return ref_x, ref_y, source, last_seen_x_m.astype(np.float32), \
        last_seen_y_m.astype(np.float32)


def _nearest_stats(distances, teams, mask):
    """Nearest distance and mean of the five nearest for one team mask."""
    selected = distances[mask]
    if selected.size == 0:
        return 50.0, 50.0
    ordered = np.sort(selected)
    return float(ordered[0]), float(np.mean(ordered[:min(5, ordered.size)]))


def build_features(ball_clean_path, raw_path, meta_path, out_dir, modes):
    started = time.time()

    with np.load(ball_clean_path) as handle:
        ball = {key: handle[key] for key in handle.files}

    n = len(ball["ball_x"])
    if n != N_FRAMES:
        raise ValueError(
            "ball_clean tiene {} frames, se esperaban {}".format(n, N_FRAMES))

    print("Cargando mapeo de equipos y Smart Tagging...")
    team_of_track = ef.get_team_mapping(meta_path)
    smart = ef.extract_smart_tagging_features()

    speed, accel, out_of_pitch, dist_to_boundary, finite_ball = \
        _ball_derived(ball)
    ref_x, ref_y, ref_source, last_seen_x_m, last_seen_y_m = \
        _reference_points(ball, finite_ball, n)
    arrays = _empty_arrays(n)
    arrays["reference_source"][:] = ref_source

    print("Procesando jugadores frame a frame...")
    with open(raw_path, "r") as handle:
        for line in handle:
            first_sep = line.find(":")
            last_sep = line.rfind(":")
            if first_sep < 0 or last_sep <= first_sep:
                continue

            frame_id = int(line[:first_sep])
            index = frame_id - 1
            if index < 0 or index >= n:
                continue

            players = line[first_sep + 1:last_sep].split(";")
            pxs = []
            pys = []
            speeds = []
            teams = []

            for slot, player in enumerate(players):
                if slot >= len(team_of_track):
                    continue
                if not player or player.startswith("NaN"):
                    continue
                parts = player.split(",")
                if len(parts) < 2:
                    continue
                try:
                    px = float(parts[0])
                    py = float(parts[1])
                except ValueError:
                    continue
                if not (np.isfinite(px) and np.isfinite(py)):
                    continue

                pxs.append(px * PITCH_LEN)
                pys.append(py * PITCH_WID)

                player_speed = np.nan
                if len(parts) > 2:
                    try:
                        player_speed = float(parts[2])
                    except ValueError:
                        player_speed = np.nan
                speeds.append(player_speed)
                teams.append(int(team_of_track[slot]))

            if not pxs:
                continue

            px_arr = np.asarray(pxs, dtype=np.float64)
            py_arr = np.asarray(pys, dtype=np.float64)
            sp_arr = np.asarray(speeds, dtype=np.float64)
            team_arr = np.asarray(teams, dtype=np.int8)

            arrays["n_players"][index] = px_arr.size
            lx = ref_x[index]
            ly = ref_y[index]

            dist_l = np.hypot(px_arr - lx, py_arr - ly)
            order = np.argsort(dist_l)
            top = order[:min(5, order.size)]

            arrays["k5_ratio"][index] = float(
                np.mean(team_arr[top] == HOME))
            arrays["centroid_x"][index] = float(np.mean(px_arr[top]))
            arrays["centroid_y"][index] = float(np.mean(py_arr[top]))
            top_speeds = sp_arr[top]
            if np.isfinite(top_speeds).any():
                arrays["centroid_speed"][index] = float(
                    np.nanmean(top_speeds))

            home_mask = team_arr == HOME
            away_mask = ~home_mask

            d_h, k5h = _nearest_stats(dist_l, team_arr, home_mask)
            d_a, k5a = _nearest_stats(dist_l, team_arr, away_mask)
            arrays["d_h_L"][index] = d_h
            arrays["d_a_L"][index] = d_a
            arrays["k5h_L"][index] = k5h
            arrays["k5a_L"][index] = k5a

            if finite_ball[index]:
                arrays["d_h_C"][index] = d_h
                arrays["d_a_C"][index] = d_a
                arrays["k5h_C"][index] = k5h
                arrays["k5a_C"][index] = k5a
            else:
                cx = arrays["centroid_x"][index]
                cy = arrays["centroid_y"][index]
                dist_c = np.hypot(px_arr - cx, py_arr - cy)
                d_h_c, k5h_c = _nearest_stats(dist_c, team_arr, home_mask)
                d_a_c, k5a_c = _nearest_stats(dist_c, team_arr, away_mask)
                arrays["d_h_C"][index] = d_h_c
                arrays["d_a_C"][index] = d_a_c
                arrays["k5h_C"][index] = k5h_c
                arrays["k5a_C"][index] = k5a_c

    delta_L = arrays["d_h_L"] - arrays["d_a_L"]
    delta_C = arrays["d_h_C"] - arrays["d_a_C"]

    missing_long = ball["ball_is_missing"].astype(np.uint8)
    missing_raw = (ball["ball_observed"].astype(np.uint8) == 0).astype(np.uint8)
    gap_age = ball["ball_gap_age"].astype(np.int32)
    streak_long = np.where(missing_long == 1, gap_age, 0)
    streak_raw = gap_age

    shared = {
        "video_frame": np.arange(n, dtype=np.int64),
        "ball_x": ball["ball_x"],
        "ball_y": ball["ball_y"],
        "ball_speed_mps": speed,
        "ball_accel_mps2": accel,
        "ball_out_of_pitch": out_of_pitch,
        "dist_to_boundary_m": dist_to_boundary,
        "ball_observed": ball["ball_observed"].astype(np.uint8),
        "ball_is_imputed": ball["ball_is_imputed"].astype(np.uint8),
        "ball_gap_len": ball["ball_gap_len"].astype(np.uint16),
        "ball_gap_age": ball["ball_gap_age"].astype(np.uint16),
        "ball_gap_remaining": ball["ball_gap_remaining"].astype(np.uint16),
        "ball_was_raw_missing": missing_raw,
        "reference_source": arrays["reference_source"],
        "n_players_visible": arrays["n_players"],
        "last_seen_ball_x_m": last_seen_x_m,
        "last_seen_ball_y_m": last_seen_y_m,
        "k5_home_ratio": arrays["k5_ratio"],
        "centroid_5nn_x_m": arrays["centroid_x"],
        "centroid_5nn_y_m": arrays["centroid_y"],
        "centroid_5nn_speed_mps": arrays["centroid_speed"],
        "frames_to_next_setpiece": smart["frames_to_next_setpiece"],
        "frames_since_last_setpiece": smart["frames_since_last_setpiece"],
        "setpiece_team": smart["setpiece_team"],
        "in_open_play_phase": smart["in_open_play_phase"],
        "open_play_team": smart["open_play_team"],
    }

    mode_paths = {}
    for mode in modes:
        if mode == "long_lastseen":
            d_h = arrays["d_h_L"]
            d_a = arrays["d_a_L"]
            k5h = arrays["k5h_L"]
            k5a = arrays["k5a_L"]
            delta = delta_L
            missing = missing_long
            streak = streak_long
        elif mode == "rawmissing":
            d_h = arrays["d_h_L"]
            d_a = arrays["d_a_L"]
            k5h = arrays["k5h_L"]
            k5a = arrays["k5a_L"]
            delta = delta_L
            missing = missing_raw
            streak = streak_raw
        elif mode == "centroidref":
            d_h = arrays["d_h_C"]
            d_a = arrays["d_a_C"]
            k5h = arrays["k5h_C"]
            k5a = arrays["k5a_C"]
            delta = delta_C
            missing = missing_long
            streak = streak_long
        elif mode == "noclean":
            d_h = arrays["d_h_L"]
            d_a = arrays["d_a_L"]
            k5h = arrays["k5h_L"]
            k5a = arrays["k5a_L"]
            delta = delta_L
            missing = missing_raw
            streak = streak_raw
        else:
            raise ValueError("modo desconocido: {}".format(mode))

        bundle = dict(shared)
        bundle.update({
            "d_nearest_home_m": d_h,
            "d_nearest_away_m": d_a,
            "delta_d_m": delta,
            "k5_home_mean_dist_m": k5h,
            "k5_away_mean_dist_m": k5a,
            "is_ball_missing": missing.astype(np.int8),
            "ball_is_missing": missing.astype(np.uint8),
            "missing_streak_frames": np.minimum(
                streak, 32767).astype(np.int16),
        })

        out_path = os.path.join(out_dir, "features_clean_{}.npz".format(mode))
        np.savez_compressed(out_path, **bundle)
        mode_paths[mode] = out_path
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print("  [OK] {} ({:.2f} MB)".format(out_path, size_mb))

    meta = {
        "ball_clean_path": os.path.abspath(ball_clean_path),
        "raw_atd_path": os.path.abspath(raw_path),
        "atd_meta_path": os.path.abspath(meta_path),
        "n_frames": n,
        "modes": {
            mode: os.path.abspath(path) for mode, path in mode_paths.items()
        },
        "counts": {
            "observed": int(ball["ball_observed"].sum()),
            "imputed": int(ball["ball_is_imputed"].sum()),
            "missing_long": int(missing_long.sum()),
            "missing_raw": int(missing_raw.sum()),
        },
        "reference_modes": {
            "long_lastseen": "last_seen",
            "rawmissing": "last_seen",
            "centroidref": "centroid_of_5nn_around_last_seen",
        },
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - started, 3),
    }
    meta_path_out = os.path.join(out_dir, "features_clean_meta.json")
    with open(meta_path_out, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print("Meta: {}".format(meta_path_out))
    print("Conteos: {}".format(meta["counts"]))
    return mode_paths, meta


def main():
    parser = argparse.ArgumentParser(
        description="Genera features limpias para el ensemble HMM.")
    parser.add_argument("--ball-clean", default=DEFAULT_BALL_CLEAN)
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--meta-path", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--mode", action="append", choices=MODES + EXTRA_MODES, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    paths = require(["atd_raw", "atd_meta"])
    raw_path = args.raw_path or paths[0]
    meta_path = args.meta_path or paths[1]

    modes = tuple(args.mode) if args.mode else MODES
    if args.all:
        modes = MODES

    os.makedirs(args.out_dir, exist_ok=True)
    build_features(
        ball_clean_path=args.ball_clean,
        raw_path=raw_path,
        meta_path=meta_path,
        out_dir=args.out_dir,
        modes=modes,
    )


if __name__ == "__main__":
    main()
