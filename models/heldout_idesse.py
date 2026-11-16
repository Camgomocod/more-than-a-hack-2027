"""Held-out IDSSE benchmark for the possession models.

The public BAR-ATH ground truth is not held out: it was used to select and tune
the ensemble. This module downloads unseen IDSSE matches, parses DFL tracking
XML without extra dependencies, builds the model feature dictionary, runs the
two model families and scores everything with the official metrics.

Panel A is the fair IDSSE-only comparison: nearest-player baseline, physical
HMM, Bundesliga-trained HMM and the HMM ensemble.
Panel B transfers the BAR-ATH candidates to IDSSE: TCN and consensus. IDSSE
has no Smart Tagging, so the set-piece features are neutralised for Panel B.
"""

import argparse
import json
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
TCN_DIR = MODELS_DIR / "ball_clean_ensemble"

for path in (ROOT / "evaluation", MODELS_DIR, TCN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import possession_metrics as pm  # noqa: E402
from boundary_refinement import refine_transition_boundaries  # noqa: E402
from hmm_ensemble import (  # noqa: E402
    ALPHA,
    DEAD_BOOST,
    DEAD_SPEED_THRESHOLD,
    build_ensemble_transition_matrix,
)
from hmm_viterbi import (  # noqa: E402
    DEAD,
    HOME,
    AWAY,
    NOBODY,
    build_transition_matrix,
    compute_emission_log_probabilities,
    debounce_spells,
    viterbi_decode,
)
from hmm_viterbi_bundesliga import (  # noqa: E402
    compute_bundesliga_trained_emissions,
)
from tcn_model import PossessionTCN  # noqa: E402
from train_tcn_clean import (  # noqa: E402
    PITCH_LEN,
    PITCH_WID,
    apply_normalizer,
    build_design_matrix,
)

MATCH_COMPETITION = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}

DEFAULT_MATCHES = ("J03WN1", "J03WPY")
DEFAULT_TRAINING_MATCH = "J03WMX"
DEFAULT_TCN_CHECKPOINTS = (
    "tcn_clean.pt",
    "tcn_clean_seed1.pt",
    "tcn_clean_seed2.pt",
    "tcn_clean_seed11.pt",
    "tcn_clean_seed23.pt",
    "tcn_clean_seed42.pt",
)

MODEL_PANEL = {
    "baseline_nearest": "A",
    "hmm_physical": "A",
    "hmm_bundesliga": "A",
    "hmm_ensemble": "A",
    "tcn_clean": "B",
    "consensus": "B",
}
MODEL_ORDER = list(MODEL_PANEL)
PAIR_ORDER = (
    ("consensus", "tcn_clean"),
    ("consensus", "hmm_ensemble"),
    ("tcn_clean", "hmm_ensemble"),
    ("hmm_bundesliga", "hmm_physical"),
)

SEGMENT_ORDER = ("firstHalf", "secondHalf")
GAP_FRAMES = 1
NEUTRAL_SETPIECE_FRAMES = 500
NEUTRAL_SETPIECE_TEAM = -1
NEUTRAL_OPEN_PLAY_TEAM = -1
NEUTRAL_IN_OPEN_PLAY = 0


def _match_urls(match_id):
    if match_id not in MATCH_COMPETITION:
        raise KeyError(
            "unknown match {}; known matches: {}".format(
                match_id, ", ".join(sorted(MATCH_COMPETITION))))
    competition = MATCH_COMPETITION[match_id]
    stem = "DFL-MAT-{}".format(match_id)
    base = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
    return {
        "match_info": "{}/DFL_02_01_matchinformation_{}_{}.xml".format(
            base, competition, stem),
        "positions": "{}/DFL_04_03_positions_raw_observed_{}_{}.xml".format(
            base, competition, stem),
    }


def _download(url, path, force=False):
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "heldout-idesse/1.0"})
    print("downloading {}".format(url))
    last_report = time.time()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with open(tmp, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    if time.time() - last_report > 10:
                        print("  {} MB".format(
                            handle.tell() // (1024 * 1024)), flush=True)
                        last_report = time.time()
        if tmp.stat().st_size < 1024:
            raise RuntimeError("downloaded file is too small: {}".format(url))
        shutil.move(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def download_match(match_id, data_dir, force=False):
    """Download the two XML files for one IDSSE match and return their paths."""
    urls = _match_urls(match_id)
    out_dir = Path(data_dir) / match_id
    return {
        kind: _download(url, out_dir / url.rsplit("/", 1)[-1], force=force)
        for kind, url in urls.items()
    }


def parse_match_info(path):
    """Read team ids and pitch dimensions from an IDSSE match-information file."""
    root = ET.parse(path).getroot()
    info = root.find("MatchInformation")
    if info is None:
        raise ValueError("MatchInformation missing in {}".format(path))
    general = info.find("General")
    environment = info.find("Environment")
    if general is None:
        raise ValueError("General block missing in {}".format(path))

    home = general.get("HomeTeamId") or general.get("HomeTeamID")
    away = (general.get("GuestTeamId") or general.get("AwayTeamId")
            or general.get("GuestTeamID"))
    if not home or not away:
        raise ValueError("team ids missing in {}".format(path))

    pitch_x = float(environment.get("PitchX", 105.0)) if environment is not None else 105.0
    pitch_y = float(environment.get("PitchY", 68.0)) if environment is not None else 68.0
    return {
        "home_id": home,
        "away_id": away,
        "pitch_length": pitch_x,
        "pitch_width": pitch_y,
    }


def _float_attr(frame, name, default=np.nan):
    value = frame.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_positions(path, home_id, away_id):
    """Stream DFL position XML into per-half ball and player arrays."""
    raw = {}
    context = ET.iterparse(path, events=("end",))
    for event, elem in context:
        if elem.tag != "FrameSet":
            continue
        section = elem.get("GameSection")
        if section not in SEGMENT_ORDER:
            elem.clear()
            continue
        team = elem.get("TeamId")
        block = raw.setdefault(
            section, {"ball": {}, "home": {}, "away": {}})
        if team == "BALL":
            for frame in elem:
                number = int(frame.get("N"))
                block["ball"][number] = (
                    _float_attr(frame, "X"),
                    _float_attr(frame, "Y"),
                    _float_attr(frame, "S"),
                    frame.get("BallStatus"),
                    frame.get("BallPossession"),
                )
        elif team in (home_id, away_id):
            side = "home" if team == home_id else "away"
            person = elem.get("PersonId")
            if person is None:
                elem.clear()
                continue
            player = block[side].setdefault(person, {})
            for frame in elem:
                player[int(frame.get("N"))] = (
                    _float_attr(frame, "X"),
                    _float_attr(frame, "Y"),
                    _float_attr(frame, "S"),
                )
        elem.clear()

    segments = []
    for section in SEGMENT_ORDER:
        if section not in raw or not raw[section]["ball"]:
            continue
        segments.append(_segment_arrays(raw[section], section))
    if not segments:
        raise ValueError("no position frames found in {}".format(path))
    return segments


def _segment_arrays(raw, section):
    numbers = sorted(raw["ball"])
    n = len(numbers)
    index = {number: i for i, number in enumerate(numbers)}
    ball = np.full((n, 5), np.nan, dtype=np.float32)
    for i, number in enumerate(numbers):
        ball[i, 0], ball[i, 1], ball[i, 2] = raw["ball"][number][:3]
        status = raw["ball"][number][3]
        possession = raw["ball"][number][4]
        ball[i, 3] = float(status) if status not in (None, "") else np.nan
        ball[i, 4] = float(possession) if possession not in (None, "") else np.nan

    players = {}
    for side in ("home", "away"):
        ids = list(raw[side])
        values = np.full((n, len(ids), 3), np.nan, dtype=np.float32)
        for j, person in enumerate(ids):
            for number, xyz in raw[side][person].items():
                i = index.get(number)
                if i is not None:
                    values[i, j] = xyz
        players[side] = values

    return {
        "section": section,
        "ball": ball,
        "home": players["home"],
        "away": players["away"],
    }


def _to_mps(values):
    finite = values[np.isfinite(values)]
    if finite.size and np.nanpercentile(finite, 99) > 55.0:
        return values / 3.6
    return values


def _track_speed(x, y, raw_speed):
    speed = _to_mps(raw_speed.astype(np.float32, copy=True))
    if x.shape[0] > 1:
        calc = np.hypot(np.gradient(x, axis=0), np.gradient(y, axis=0)) * 25.0
    else:
        calc = np.zeros_like(x)
    speed = np.where(np.isfinite(speed) & (speed > 0), speed, calc)
    speed[~np.isfinite(x) | ~np.isfinite(y)] = 0.0
    return np.maximum(speed, 0.0).astype(np.float32)


def _forward_fill(values):
    out = values.copy()
    current = np.nan
    for i, value in enumerate(out):
        if np.isfinite(value):
            current = value
        else:
            out[i] = current
    first = np.flatnonzero(np.isfinite(out))
    if first.size:
        out[:first[0]] = out[first[0]]
    else:
        out[:] = np.nan
    return out


def _gap_features(missing):
    n = len(missing)
    length = np.zeros(n, dtype=np.float32)
    age = np.zeros(n, dtype=np.float32)
    remaining = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        size = j - i
        steps = np.arange(size, dtype=np.float32)
        length[i:j] = size
        age[i:j] = steps
        remaining[i:j] = size - 1 - steps
        i = j
    return length, age, remaining


def _player_features(home, away, ball_x, ball_y, ball_observed, last_x, last_y):
    n = len(ball_x)
    hx, hy, hs = home[:, :, 0], home[:, :, 1], home[:, :, 2]
    ax, ay, as_ = away[:, :, 0], away[:, :, 1], away[:, :, 2]

    d_h = np.full(n, 50.0, dtype=np.float32)
    d_a = np.full(n, 50.0, dtype=np.float32)
    k5_ratio = np.full(n, 0.5, dtype=np.float32)
    k5_home_mean = np.full(n, 50.0, dtype=np.float32)
    k5_away_mean = np.full(n, 50.0, dtype=np.float32)
    centroid_x = np.zeros(n, dtype=np.float32)
    centroid_y = np.zeros(n, dtype=np.float32)
    centroid_speed = np.zeros(n, dtype=np.float32)
    n_visible = np.zeros(n, dtype=np.int16)

    for i in range(n):
        ref_x = ball_x[i] if ball_observed[i] else last_x[i]
        ref_y = ball_y[i] if ball_observed[i] else last_y[i]
        h_finite = np.isfinite(hx[i]) & np.isfinite(hy[i])
        a_finite = np.isfinite(ax[i]) & np.isfinite(ay[i])
        h_pos = np.column_stack((hx[i][h_finite], hy[i][h_finite]))
        a_pos = np.column_stack((ax[i][a_finite], ay[i][a_finite]))
        h_speed = hs[i][h_finite]
        a_speed = as_[i][a_finite]
        n_visible[i] = len(h_pos) + len(a_pos)

        if len(h_pos):
            h_dist = np.hypot(h_pos[:, 0] - ref_x, h_pos[:, 1] - ref_y)
            d_h[i] = float(np.min(h_dist))
            k5_home_mean[i] = float(np.mean(np.sort(h_dist)[:5]))
        if len(a_pos):
            a_dist = np.hypot(a_pos[:, 0] - ref_x, a_pos[:, 1] - ref_y)
            d_a[i] = float(np.min(a_dist))
            k5_away_mean[i] = float(np.mean(np.sort(a_dist)[:5]))

        positions = np.vstack((h_pos, a_pos)) if len(h_pos) + len(a_pos) else None
        if positions is None or not len(positions):
            centroid_x[i] = ref_x
            centroid_y[i] = ref_y
            continue

        distances = np.hypot(
            positions[:, 0] - ref_x, positions[:, 1] - ref_y)
        top = np.argsort(distances)[:5]
        k5_ratio[i] = float(np.mean(top < len(h_pos)))
        centroid_x[i] = float(np.mean(positions[top, 0]))
        centroid_y[i] = float(np.mean(positions[top, 1]))
        speeds = np.concatenate((h_speed, a_speed))[top]
        speeds = speeds[np.isfinite(speeds)]
        if len(speeds):
            centroid_speed[i] = float(np.mean(speeds))

    return {
        "d_nearest_home_m": d_h,
        "d_nearest_away_m": d_a,
        "delta_d_m": (d_h - d_a).astype(np.float32),
        "k5_home_ratio": k5_ratio,
        "k5_home_mean_dist_m": k5_home_mean,
        "k5_away_mean_dist_m": k5_away_mean,
        "centroid_5nn_x_m": centroid_x + PITCH_LEN / 2.0,
        "centroid_5nn_y_m": centroid_y + PITCH_WID / 2.0,
        "centroid_5nn_speed_mps": centroid_speed,
        "n_players_visible": n_visible,
    }


def segment_features(segment):
    """Build the 37-feature model dictionary for one half."""
    ball = segment["ball"]
    bx, by = ball[:, 0], ball[:, 1]
    observed = np.isfinite(bx) & np.isfinite(by)
    n = len(bx)

    ball_x = (bx + PITCH_LEN / 2.0) / PITCH_LEN
    ball_y = (by + PITCH_WID / 2.0) / PITCH_WID

    last_x = _forward_fill(bx)
    last_y = _forward_fill(by)
    has_last = np.isfinite(last_x) & np.isfinite(last_y)
    last_x = np.where(has_last, last_x, 0.0)
    last_y = np.where(has_last, last_y, 0.0)
    reference = np.where(observed, 0, np.where(has_last, 1, 2)).astype(np.uint8)

    speed = _track_speed(
        bx.reshape(-1, 1), by.reshape(-1, 1), ball[:, 2].reshape(-1, 1))[:, 0]
    speed[~observed] = 0.0
    if n > 1:
        accel = np.gradient(speed) * 25.0
    else:
        accel = np.zeros(n, dtype=np.float32)
    accel[~observed] = 0.0

    gap_len, gap_age, gap_remaining = _gap_features(~observed)
    out = np.zeros(n, dtype=np.int8)
    out[observed & ((np.abs(bx) > PITCH_LEN / 2.0) | (np.abs(by) > PITCH_WID / 2.0))] = 1
    boundary = np.minimum(PITCH_LEN / 2.0 - np.abs(bx), PITCH_WID / 2.0 - np.abs(by))
    boundary = np.where(observed, np.maximum(boundary, 0.0), np.nan)

    players = _player_features(
        segment["home"], segment["away"], bx, by, observed, last_x, last_y)

    feats = {
        "video_frame": np.arange(n, dtype=np.int32),
        "ball_x": ball_x.astype(np.float32),
        "ball_y": ball_y.astype(np.float32),
        "ball_speed_mps": speed,
        "ball_accel_mps2": accel.astype(np.float32),
        "ball_out_of_pitch": out,
        "dist_to_boundary_m": boundary.astype(np.float32),
        "ball_observed": observed.astype(np.uint8),
        "ball_is_missing": (~observed).astype(np.uint8),
        "is_ball_missing": (~observed).astype(np.uint8),
        "ball_is_imputed": np.zeros(n, dtype=np.uint8),
        "ball_was_raw_missing": (~observed).astype(np.uint8),
        "ball_gap_len": gap_len,
        "ball_gap_age": gap_age,
        "ball_gap_remaining": gap_remaining,
        "missing_streak_frames": gap_age,
        "reference_source": reference,
        "last_seen_ball_x_m": (last_x + PITCH_LEN / 2.0).astype(np.float32),
        "last_seen_ball_y_m": (last_y + PITCH_WID / 2.0).astype(np.float32),
        "frames_to_next_setpiece": np.full(n, NEUTRAL_SETPIECE_FRAMES, dtype=np.int16),
        "frames_since_last_setpiece": np.full(n, NEUTRAL_SETPIECE_FRAMES, dtype=np.int16),
        "setpiece_team": np.full(n, NEUTRAL_SETPIECE_TEAM, dtype=np.int8),
        "in_open_play_phase": np.full(n, NEUTRAL_IN_OPEN_PLAY, dtype=np.int8),
        "open_play_team": np.full(n, NEUTRAL_OPEN_PLAY_TEAM, dtype=np.int8),
    }
    feats.update(players)
    return feats


def segment_labels(segment):
    """Map DFL BallStatus/BallPossession to H/A/D/X."""
    ball = segment["ball"]
    status = ball[:, 3]
    possession = ball[:, 4]
    labels = np.full(len(ball), NOBODY, dtype=np.int64)
    labels[status == 0] = DEAD
    alive = status == 1
    labels[alive & (possession == 1)] = HOME
    labels[alive & (possession == 2)] = AWAY
    return labels


def _gap_value(key):
    if key in ("ball_x", "ball_y", "dist_to_boundary_m"):
        return np.nan
    if key in ("last_seen_ball_x_m", "last_seen_ball_y_m"):
        return PITCH_LEN / 2.0 if "x" in key else PITCH_WID / 2.0
    if key in ("centroid_5nn_x_m", "centroid_5nn_y_m"):
        return PITCH_LEN / 2.0 if "x" in key else PITCH_WID / 2.0
    if key == "k5_home_ratio":
        return 0.5
    if key in ("frames_to_next_setpiece", "frames_since_last_setpiece"):
        return NEUTRAL_SETPIECE_FRAMES
    if key in ("setpiece_team", "open_play_team"):
        return -1
    if key in ("d_nearest_home_m", "d_nearest_away_m",
               "k5_home_mean_dist_m", "k5_away_mean_dist_m"):
        return 50.0
    if key == "reference_source":
        return 2
    if key == "in_open_play_phase":
        return 0
    if key in ("ball_is_missing", "is_ball_missing", "ball_was_raw_missing"):
        return 1
    if key == "ball_gap_len":
        return GAP_FRAMES
    return 0


def _concat_features(items):
    if not items:
        raise ValueError("no feature segments to concatenate")
    out = {}
    for key in items[0]:
        pieces = []
        for i, item in enumerate(items):
            if i:
                pieces.append(np.full(
                    GAP_FRAMES, _gap_value(key), dtype=item[key].dtype))
            pieces.append(item[key])
        out[key] = np.concatenate(pieces)
    out["video_frame"] = np.arange(len(out["video_frame"]), dtype=np.int32)
    return out


def build_match_features(segments):
    """Concatenate halves with one X frame between them."""
    feats = _concat_features([segment_features(item) for item in segments])
    gt_parts = []
    for i, segment in enumerate(segments):
        if i:
            gt_parts.append(np.full(GAP_FRAMES, NOBODY, dtype=np.int64))
        gt_parts.append(segment_labels(segment))
    gt = np.concatenate(gt_parts)
    return feats, gt, pm.valid_runs(gt)


def load_weights(path=None):
    path = Path(path) if path is not None else MODELS_DIR / "bundesliga_weights.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tcn_models(paths=None):
    """Load the six TCN checkpoints, or a caller-supplied list."""
    import torch

    if paths is None:
        paths = [TCN_DIR / name for name in DEFAULT_TCN_CHECKPOINTS]
    loaded = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print("skipping missing TCN checkpoint {}".format(path))
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        model = PossessionTCN(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loaded.append({
            "model": model,
            "median": checkpoint["median"],
            "scale": checkpoint["scale"],
            "feature_names": list(checkpoint.get("feature_names", [])),
        })
    if not loaded:
        raise RuntimeError("no TCN checkpoints loaded")
    return loaded


def _tcn_probabilities(models, matrix, feature_names):
    import torch

    total = None
    for item in models:
        if item["feature_names"] and item["feature_names"] != feature_names:
            raise ValueError("TCN feature order does not match build_design_matrix")
        normalized = apply_normalizer(matrix, item["median"], item["scale"])
        with torch.no_grad():
            logits = item["model"](torch.from_numpy(normalized.T[None]))[0].T.numpy()
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        total = probabilities if total is None else total + probabilities
    return total / float(len(models))


def _decode_runs(emissions, runs, transition, min_spell, refine, feats):
    labels = np.full(len(emissions), NOBODY, dtype=np.int8)
    speed = np.where(
        np.isfinite(feats["ball_speed_mps"]), feats["ball_speed_mps"], np.inf)
    for start, stop in runs:
        path = viterbi_decode(emissions[start:stop], transition)
        path = debounce_spells(path, min_spell)
        if refine:
            path = refine_transition_boundaries(
                path,
                feats["d_nearest_home_m"][start:stop],
                feats["d_nearest_away_m"][start:stop],
                feats["ball_out_of_pitch"][start:stop],
                speed[start:stop],
            )
        labels[start:stop] = path
    return labels


def _ensemble_emissions(feats, weights):
    physical = compute_emission_log_probabilities(feats)
    trained = compute_bundesliga_trained_emissions(feats, weights)
    emissions = ALPHA * physical + (1.0 - ALPHA) * trained
    speed = np.where(
        np.isfinite(feats["ball_speed_mps"]), feats["ball_speed_mps"],
        DEAD_SPEED_THRESHOLD)
    extra_dead = DEAD_BOOST * (1 - feats["in_open_play_phase"]) * np.maximum(
        0.0, (DEAD_SPEED_THRESHOLD - speed) / DEAD_SPEED_THRESHOLD)
    emissions[:, DEAD] += extra_dead
    maximum = np.max(emissions, axis=1, keepdims=True)
    exponentials = np.exp(emissions - maximum)
    return np.log(np.maximum(
        1e-12, exponentials / np.sum(exponentials, axis=1, keepdims=True)))


def _baseline_predictions(feats, runs):
    labels = np.full(len(feats["ball_x"]), NOBODY, dtype=np.int8)
    observed = feats["ball_observed"] == 1
    for start, stop in runs:
        previous = DEAD
        for i in range(start, stop):
            if feats["ball_out_of_pitch"][i]:
                label = DEAD
            elif not observed[i]:
                label = previous
            elif feats["d_nearest_home_m"][i] < feats["d_nearest_away_m"][i]:
                label = HOME
            elif feats["d_nearest_away_m"][i] < feats["d_nearest_home_m"][i]:
                label = AWAY
            else:
                label = previous
            labels[i] = label
            previous = label
    return labels


def _consensus_predictions(tcn, ensemble, feats, runs):
    labels = tcn.copy()
    dead_mask = (ensemble == DEAD) & ((tcn == HOME) | (tcn == AWAY))
    labels[dead_mask] = DEAD
    speed = np.where(
        np.isfinite(feats["ball_speed_mps"]), feats["ball_speed_mps"], np.inf)
    for start, stop in runs:
        stage = refine_transition_boundaries(
            labels[start:stop],
            feats["d_nearest_home_m"][start:stop],
            feats["d_nearest_away_m"][start:stop],
            feats["ball_out_of_pitch"][start:stop],
            speed[start:stop],
            window_live=20,
        )
        stage = debounce_spells(stage, 25)
        stage = refine_transition_boundaries(
            stage,
            feats["d_nearest_home_m"][start:stop],
            feats["d_nearest_away_m"][start:stop],
            feats["ball_out_of_pitch"][start:stop],
            speed[start:stop],
            window_live=20,
        )
        labels[start:stop] = stage
    return labels


def predict_models(feats, runs, weights, tcn_models=None):
    """Run both panels. Panel B is omitted if no TCN checkpoints are available."""
    predictions = {
        "baseline_nearest": _baseline_predictions(feats, runs),
        "hmm_physical": _decode_runs(
            compute_emission_log_probabilities(feats), runs,
            build_transition_matrix(), 50, False, feats),
        "hmm_bundesliga": _decode_runs(
            compute_bundesliga_trained_emissions(feats, weights), runs,
            build_transition_matrix(), 50, True, feats),
        "hmm_ensemble": _decode_runs(
            _ensemble_emissions(feats, weights), runs,
            build_ensemble_transition_matrix(), 50, True, feats),
    }

    if tcn_models:
        matrix, names = build_design_matrix(feats)
        probabilities = _tcn_probabilities(tcn_models, matrix, names)
        tcn_log_probabilities = np.log(np.maximum(1e-12, probabilities))
        tcn = _decode_runs(
            tcn_log_probabilities * 0.8, runs,
            build_ensemble_transition_matrix(), 62, False, feats)
        predictions["tcn_clean"] = tcn
        predictions["consensus"] = _consensus_predictions(
            tcn, predictions["hmm_ensemble"], feats, runs)
    return predictions


def _metric_row(match, model, result):
    return {
        "match": match,
        "panel": MODEL_PANEL[model],
        "model": model,
        "spell_f1": result["spell_f1_mean"],
        "spell_lo": result["spell_bootstrap_low"],
        "spell_hi": result["spell_bootstrap_high"],
        "macro_f1": result["macro_f1"],
        "alive_dead_f1": result["alive_dead_f1"],
        "n_gt_spells": result["n_gt_spells"],
        "n_pred_spells": result["n_pred_spells"],
        "n_graded": result["n_graded"],
        "transitions_25": result["transitions"][25]["f1"],
    }


def _spell_flags(gt, pred, runs):
    filled = pm.fill_abstentions(pred, runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = {
        threshold: pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
        for threshold in pm.SPELL_IOUS
    }
    return flags, len(gt_spells), len(pred_spells)


def _score_from_flags(flags, n_gt, n_pred, indices):
    scores = []
    for threshold in pm.SPELL_IOUS:
        true_positive = int(flags[threshold][indices].sum())
        false_positive = n_pred - true_positive
        false_negative = n_gt - true_positive
        precision = true_positive / float(true_positive + false_positive) \
            if true_positive + false_positive else 0.0
        recall = true_positive / float(true_positive + false_negative) \
            if true_positive + false_negative else 0.0
        scores.append(2.0 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(scores))


def paired_spell_difference(gt, pred_a, pred_b, runs, n_draws, seed=0):
    """Paired bootstrap of mean spell-F1 for A minus B, resampling GT spells."""
    flags_a, n_gt, n_pred_a = _spell_flags(gt, pred_a, runs)
    flags_b, _, n_pred_b = _spell_flags(gt, pred_b, runs)
    if n_gt == 0:
        return {"difference": np.nan, "low": np.nan, "high": np.nan,
                "fraction_a_better": np.nan}
    all_indices = np.arange(n_gt)
    observed_difference = (
        _score_from_flags(flags_a, n_gt, n_pred_a, all_indices)
        - _score_from_flags(flags_b, n_gt, n_pred_b, all_indices))
    rng = np.random.default_rng(seed)
    differences = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        indices = rng.integers(0, n_gt, n_gt)
        differences[draw] = (
            _score_from_flags(flags_a, n_gt, n_pred_a, indices)
            - _score_from_flags(flags_b, n_gt, n_pred_b, indices))
    low, high = np.percentile(differences, (5, 95))
    return {
        "difference": float(observed_difference),
        "low": float(low),
        "high": float(high),
        "fraction_a_better": float(np.mean(differences > 0)),
    }


def _join_with_gap(items, gap_value):
    pieces = []
    for i, item in enumerate(items):
        if i:
            pieces.append(np.full(GAP_FRAMES, gap_value, dtype=item.dtype))
        pieces.append(item)
    return np.concatenate(pieces)


def _sort_table(table):
    if table.empty:
        return table
    order = {model: i for i, model in enumerate(MODEL_ORDER)}
    return (table.assign(_order=table["model"].map(order))
                 .sort_values(["match", "_order"])
                 .drop(columns="_order")
                 .reset_index(drop=True))


def _sort_paired(table, matches):
    if table.empty:
        return table
    match_order = {match: i for i, match in enumerate(list(matches) + ["POOLED"])}
    pair_order = {pair: i for i, pair in enumerate(PAIR_ORDER)}
    return (table.assign(
                _match=table["match"].map(match_order),
                _pair=[pair_order[(a, b)] for a, b in zip(
                    table["model_a"], table["model_b"])])
                 .sort_values(["_match", "_pair"])
                 .drop(columns=["_match", "_pair"])
                 .reset_index(drop=True))


def _write_summary(path, per_match, pooled, paired, config):
    lines = [
        "# Held-out IDSSE benchmark",
        "",
        "Matches: {}".format(", ".join(config["matches"])),
        "Bootstrap draws: {}".format(config["bootstrap"]),
        "",
        "Panel A is the fair IDSSE-only comparison. Panel B transfers BAR-ATH "
        "candidates with Smart Tagging neutralised; treat B as indicative.",
        "",
        "## Per match",
        "",
        per_match.to_string(index=False),
        "",
        "## Pooled",
        "",
        pooled.to_string(index=False),
        "",
        "## Paired spell-F1 differences",
        "",
        paired.to_string(index=False),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(matches=DEFAULT_MATCHES, data_dir=None, out_dir=None,
                  bootstrap=1000, download=True, force=False,
                  checkpoint_paths=None, seed=0):
    """Download, parse, predict and score the requested held-out matches."""
    matches = tuple(matches)
    if DEFAULT_TRAINING_MATCH in matches:
        print("warning: {} was used to train the Bundesliga weights and is not "
              "a fair held-out match".format(DEFAULT_TRAINING_MATCH))
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data" / "idsse"
    out_dir = Path(out_dir) if out_dir is not None else ROOT / "evaluation" / "heldout_idesse"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = load_weights()
    tcn_models = load_tcn_models(checkpoint_paths)

    rows = []
    pooled_gt = []
    pooled_predictions = {}

    for match in matches:
        print("\n=== {} ===".format(match))
        if download:
            paths = download_match(match, data_dir, force=force)
        else:
            paths = {
                "match_info": data_dir / match / "DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml".format(
                    MATCH_COMPETITION[match], match),
                "positions": data_dir / match / "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml".format(
                    MATCH_COMPETITION[match], match),
            }
        info = parse_match_info(paths["match_info"])
        segments = parse_positions(
            paths["positions"], info["home_id"], info["away_id"])
        feats, gt, runs = build_match_features(segments)
        predictions = predict_models(feats, runs, weights, tcn_models)

        for model in MODEL_ORDER:
            if model not in predictions:
                continue
            result = pm.evaluate(gt, predictions[model], bootstrap_n=bootstrap)
            rows.append(_metric_row(match, model, result))

        for model_a, model_b in PAIR_ORDER:
            if model_a not in predictions or model_b not in predictions:
                continue
            stats = paired_spell_difference(
                gt, predictions[model_a], predictions[model_b], runs,
                bootstrap, seed=seed)
            rows.append({"_paired": {
                "match": match,
                "model_a": model_a,
                "model_b": model_b,
                "difference": stats["difference"],
                "low": stats["low"],
                "high": stats["high"],
                "fraction_a_better": stats["fraction_a_better"],
            }})

        pooled_gt.append(gt)
        for model, prediction in predictions.items():
            pooled_predictions.setdefault(model, []).append(prediction)

    paired_rows = [row["_paired"] for row in rows if "_paired" in row]
    rows = [row for row in rows if "_paired" not in row]
    per_match = _sort_table(pd.DataFrame(rows))

    joined_gt = _join_with_gap(pooled_gt, NOBODY)
    pooled_runs = pm.valid_runs(joined_gt)
    pooled_rows = []
    for model in MODEL_ORDER:
        if model not in pooled_predictions:
            continue
        joined_prediction = _join_with_gap(pooled_predictions[model], NOBODY)
        result = pm.evaluate(joined_gt, joined_prediction, bootstrap_n=bootstrap)
        pooled_rows.append(_metric_row("POOLED", model, result))
        for model_a, model_b in PAIR_ORDER:
            if model_b != model or model_a not in pooled_predictions:
                continue
            stats = paired_spell_difference(
                joined_gt, _join_with_gap(pooled_predictions[model_a], NOBODY),
                joined_prediction, pooled_runs, bootstrap, seed=seed)
            paired_rows.append({
                "match": "POOLED",
                "model_a": model_a,
                "model_b": model_b,
                "difference": stats["difference"],
                "low": stats["low"],
                "high": stats["high"],
                "fraction_a_better": stats["fraction_a_better"],
            })
    pooled = _sort_table(pd.DataFrame(pooled_rows))
    paired = _sort_paired(pd.DataFrame(paired_rows), matches)

    per_match.to_csv(out_dir / "heldout_per_match.csv", index=False)
    pooled.to_csv(out_dir / "heldout_pooled.csv", index=False)
    paired.to_csv(out_dir / "heldout_paired.csv", index=False)
    payload = {
        "config": {
            "matches": list(matches),
            "bootstrap": bootstrap,
            "seed": seed,
            "neutral_setpiece_frames": NEUTRAL_SETPIECE_FRAMES,
            "neutral_setpiece_team": NEUTRAL_SETPIECE_TEAM,
            "neutral_open_play_team": NEUTRAL_OPEN_PLAY_TEAM,
            "neutral_in_open_play": NEUTRAL_IN_OPEN_PLAY,
        },
        "per_match": per_match.to_dict(orient="records"),
        "pooled": pooled.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
    }
    with open(out_dir / "heldout_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=lambda value: (
            value.item() if hasattr(value, "item") else str(value)))
    _write_summary(
        out_dir / "heldout_summary.md", per_match, pooled, paired, payload["config"])

    print("\nPer match")
    print(per_match.to_string(index=False))
    print("\nPooled")
    print(pooled.to_string(index=False))
    print("\nPaired")
    print(paired.to_string(index=False))
    print("\nWrote {}".format(out_dir))
    return {
        "per_match": per_match,
        "pooled": pooled,
        "paired": paired,
        "out_dir": out_dir,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", nargs="+", default=list(DEFAULT_MATCHES))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    checkpoints = [Path(item) for item in args.checkpoints] if args.checkpoints else None
    run_benchmark(
        matches=args.matches,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        bootstrap=args.bootstrap,
        download=not args.no_download,
        force=args.force_download,
        checkpoint_paths=checkpoints,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
