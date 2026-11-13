"""Train a causal TCN on cleaned ball features and HMM pseudo-labels.

The TCN never sees the public ground truth. Its targets are algorithm-produced
pseudo-labels (the clean HMM ensemble), which is allowed by the competition
rules. The public GT is only used after training for diagnosis.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)

import possession_metrics as pm  # noqa: E402
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
)
from hmm_viterbi_bundesliga import (  # noqa: E402
    compute_bundesliga_trained_emissions,
)
from tcn_model import PossessionTCN  # noqa: E402

FEATURES = os.path.join(HERE, "features_clean_long_lastseen.npz")
LABELS = os.path.join(HERE, "results", "clean_ensemble.csv")
WEIGHTS = os.path.join(MODELS_DIR, "bundesliga_weights.json")
GT_PATH = os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
MODEL_PATH = os.path.join(HERE, "tcn_clean.pt")
LOGITS_PATH = os.path.join(HERE, "tcn_clean_logits.npz")


def _tagged_path(path, tag):
    if not tag:
        return path
    stem, ext = os.path.splitext(path)
    return stem + "_" + tag + ext


PITCH_LEN = 105.3
PITCH_WID = 68.0
HALVES = ((HALF_1_KICKOFF, HALF_1_END), (HALF_2_KICKOFF, HALF_2_END))
CODE_MAP = {"H": 0, "A": 1, "D": 2, "X": 3}
SEED = 7


def _cap(values, low, high):
    return np.clip(values, low, high)


def _safe_ratio(values, denom, fill):
    return np.where(np.isfinite(values), values / denom, fill)


def build_design_matrix(feats):
    """Build the TCN input matrix and feature names from cleaned features."""
    n = len(feats["video_frame"])
    ball_x = feats["ball_x"].astype(np.float32)
    ball_y = feats["ball_y"].astype(np.float32)
    finite_ball = np.isfinite(ball_x) & np.isfinite(ball_y)
    last_x = feats["last_seen_ball_x_m"].astype(np.float32)
    last_y = feats["last_seen_ball_y_m"].astype(np.float32)
    finite_last = np.isfinite(last_x) & np.isfinite(last_y)

    ball_x_filled = np.where(finite_ball, ball_x, _safe_ratio(
        last_x, PITCH_LEN, 0.5))
    ball_y_filled = np.where(finite_ball, ball_y, _safe_ratio(
        last_y, PITCH_WID, 0.5))
    last_x_norm = np.where(finite_last, last_x / PITCH_LEN, 0.5)
    last_y_norm = np.where(finite_last, last_y / PITCH_WID, 0.5)

    columns = []
    names = []

    def add(name, values):
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float32))

    add("ball_x", ball_x_filled)
    add("ball_y", ball_y_filled)
    add("ball_observed", feats["ball_observed"])
    add("ball_is_missing", feats["ball_is_missing"])
    add("ball_is_imputed", feats["ball_is_imputed"])
    add("ball_out_of_pitch", feats["ball_out_of_pitch"])
    add("ball_speed_mps", _cap(feats["ball_speed_mps"], 0.0, 45.0))
    add("ball_accel_mps2", _cap(feats["ball_accel_mps2"], -50.0, 50.0))
    add("dist_to_boundary_m", _cap(feats["dist_to_boundary_m"], 0.0, 80.0))
    add("ball_gap_len", _cap(feats["ball_gap_len"], 0.0, 500.0))
    add("ball_gap_age", _cap(feats["ball_gap_age"], 0.0, 500.0))
    add("ball_gap_remaining", _cap(feats["ball_gap_remaining"], 0.0, 500.0))
    add("n_players_visible", _cap(feats["n_players_visible"], 0.0, 30.0))
    add("last_seen_ball_x", last_x_norm)
    add("last_seen_ball_y", last_y_norm)
    add("k5_home_ratio", feats["k5_home_ratio"])
    add("centroid_5nn_x", feats["centroid_5nn_x_m"] / PITCH_LEN)
    add("centroid_5nn_y", feats["centroid_5nn_y_m"] / PITCH_WID)
    add("centroid_5nn_speed", _cap(
        feats["centroid_5nn_speed_mps"], 0.0, 45.0))
    add("frames_to_next_setpiece", _cap(
        feats["frames_to_next_setpiece"], 0.0, 500.0))
    add("frames_since_last_setpiece", _cap(
        feats["frames_since_last_setpiece"], 0.0, 500.0))
    add("in_open_play_phase", feats["in_open_play_phase"])
    add("d_nearest_home_m", _cap(feats["d_nearest_home_m"], 0.0, 120.0))
    add("d_nearest_away_m", _cap(feats["d_nearest_away_m"], 0.0, 120.0))
    add("delta_d_m", _cap(feats["delta_d_m"], -60.0, 60.0))
    add("k5_home_mean_dist_m", _cap(
        feats["k5_home_mean_dist_m"], 0.0, 120.0))
    add("k5_away_mean_dist_m", _cap(
        feats["k5_away_mean_dist_m"], 0.0, 120.0))
    add("missing_streak_frames", _cap(
        feats["missing_streak_frames"], 0.0, 500.0))

    reference = feats["reference_source"]
    for value in range(3):
        add("reference_source_{}".format(value), reference == value)
    setpiece_team = feats["setpiece_team"]
    for value in range(3):
        add("setpiece_team_{}".format(value), setpiece_team == value)
    open_play_team = feats["open_play_team"]
    for value in range(3):
        add("open_play_team_{}".format(value), open_play_team == value)

    matrix = np.stack(columns, axis=1).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, names


def fit_normalizer(matrix):
    median = np.median(matrix, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    scale = q75 - q25
    std = np.std(matrix, axis=0)
    scale = np.where(scale < 1e-6, np.where(std < 1e-6, 1.0, std), scale)
    normalized = np.clip((matrix - median) / scale, -10.0, 10.0)
    return normalized.astype(np.float32), median, scale


def apply_normalizer(matrix, median, scale):
    return np.clip(
        (matrix - median) / scale, -10.0, 10.0).astype(np.float32)


def load_labels(path):
    frame = pd.read_csv(path)
    labels = frame["label"].astype(str).str.strip().str.upper().values
    codes = np.array([CODE_MAP[label] for label in labels], dtype=np.int64)
    target = np.full(len(codes), -100, dtype=np.int64)
    for start, stop in HALVES:
        window = codes[start:stop]
        target[start:stop] = np.where(window == 3, -100, window)
    return target


def build_soft_targets(feats):
    with open(WEIGHTS, "r", encoding="utf-8") as handle:
        weights = json.load(handle)
    hand = compute_emission_log_probabilities(feats) / 0.80
    trained = compute_bundesliga_trained_emissions(feats, weights) / 0.80
    emissions = 0.35 * hand + 0.65 * trained
    emissions = emissions - emissions.max(axis=1, keepdims=True)
    probabilities = np.exp(emissions)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def transition_weights(target, radius=25, weight=5.0):
    weights = np.ones(len(target), dtype=np.float32)
    for start, stop in HALVES:
        window = target[start:stop]
        changes = np.flatnonzero(window[1:] != window[:-1]) + 1
        for change in changes:
            lo = max(0, change - radius)
            hi = min(len(window), change + radius)
            weights[start + lo:start + hi] = weight
    return weights


def sample_batch(matrix, target, soft, weights, batch_size, seq_len, rng):
    starts = []
    for _ in range(batch_size):
        start, stop = HALVES[rng.randint(len(HALVES))]
        max_start = stop - seq_len
        starts.append(rng.randint(start, max_start + 1))

    x = np.stack([matrix[s:s + seq_len] for s in starts], axis=0)
    y = np.stack([target[s:s + seq_len] for s in starts], axis=0)
    s = np.stack([soft[s:s + seq_len] for s in starts], axis=0)
    w = np.stack([weights[s:s + seq_len] for s in starts], axis=0)
    return x, y, s, w


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    print("Loading features: {}".format(args.features))
    with np.load(args.features) as feats:
        matrix, names = build_design_matrix(feats)
        soft = build_soft_targets(feats)

    print("Normalizing {} features...".format(len(names)))
    matrix, median, scale = fit_normalizer(matrix)
    print("Loading labels: {}".format(args.labels))
    target = load_labels(args.labels)
    weights = transition_weights(target)

    valid = target >= 0
    counts = np.bincount(target[valid], minlength=3).astype(np.float64)
    class_weights = counts.sum() / (3.0 * counts)
    class_weights = np.clip(class_weights, 0.5, 3.0)
    print("Pseudo-label counts H/A/D: {}".format(counts.astype(int).tolist()))
    print("Class weights: {}".format(np.round(class_weights, 3).tolist()))

    model = PossessionTCN(
        in_channels=len(names),
        n_classes=3,
        channels=args.channels,
        levels=args.levels,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps)
    class_weight_tensor = torch.tensor(
        class_weights, dtype=torch.float32)

    print("Training TCN: {} epochs x {} steps, seq_len={}, batch={}".format(
        args.epochs, args.steps, args.seq_len, args.batch_size))
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step in range(args.steps):
            x, y, s, w = sample_batch(
                matrix, target, soft, weights,
                args.batch_size, args.seq_len, rng)
            xb = torch.from_numpy(x.transpose(0, 2, 1))
            yb = torch.from_numpy(y)
            sb = torch.from_numpy(s)
            wb = torch.from_numpy(w)

            logits = model(xb)
            mask = yb.reshape(-1) >= 0
            logits_flat = logits.permute(0, 2, 1).reshape(-1, 3)
            w_flat = wb.reshape(-1)

            hard = F.cross_entropy(
                logits_flat[mask], yb.reshape(-1)[mask],
                weight=class_weight_tensor, reduction="none")
            hard = (hard * w_flat[mask]).mean()

            soft_flat = sb.reshape(-1, 3)[mask]
            logp = F.log_softmax(logits_flat, dim=1)[mask]
            soft_loss = -(soft_flat * logp).sum(dim=1)
            soft_loss = (soft_loss * w_flat[mask]).mean()

            tv = (logits[:, :, 1:] - logits[:, :, :-1]).abs().mean()
            loss = hard + args.soft_weight * soft_loss + args.tv_weight * tv

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())

            if step % 25 == 0 or step == args.steps - 1:
                print("  epoch {:2d} step {:3d} loss={:.4f} lr={:.2e}".format(
                    epoch + 1, step, running / (step + 1),
                    scheduler.get_last_lr()[0]))

    elapsed = time.time() - t0
    print("Training finished in {:.1f}s".format(elapsed))
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "in_channels": len(names),
            "n_classes": 3,
            "channels": args.channels,
            "levels": args.levels,
            "dropout": args.dropout,
        },
        "feature_names": names,
        "median": median,
        "scale": scale,
        "training": {
            "features": args.features,
            "labels": args.labels,
            "epochs": args.epochs,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "soft_weight": args.soft_weight,
            "tv_weight": args.tv_weight,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
        },
    }
    model_path = _tagged_path(MODEL_PATH, args.tag)
    torch.save(checkpoint, model_path)
    print("Saved model to {}".format(model_path))
    return model, matrix, median, scale


def predict(model, matrix, logits_path):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(matrix.T[None])
        logits = model(x)[0].T.numpy()
    probabilities = torch.softmax(
        torch.from_numpy(logits), dim=1).numpy()
    np.savez_compressed(
        logits_path,
        logits=logits.astype(np.float32),
        probabilities=probabilities.astype(np.float32),
    )
    print("Saved logits to {}".format(logits_path))
    return logits, probabilities


def _outside_fill(labels):
    labels = labels.copy()
    labels[:HALF_1_KICKOFF] = NOBODY
    labels[HALF_1_END:HALF_2_KICKOFF] = DEAD
    labels[HALF_2_END:] = NOBODY
    return labels


def write_submission(labels, path):
    frame = pd.DataFrame({
        "frame": np.arange(N_FRAMES, dtype=np.int64),
        "label": [LABEL_MAP[int(value)] for value in labels],
    })
    frame.to_csv(path, index=False)
    print("Wrote {}".format(path))


def write_variants(logits, tag=""):
    suffix = "_" + tag if tag else ""
    raw = np.argmax(logits, axis=1).astype(np.int8)
    raw = _outside_fill(raw)
    write_submission(
        raw,
        os.path.join(
            HERE, "results", "tcn_clean_raw{}.csv".format(suffix)))

    for min_spell in (25, 50):
        labels = np.full(N_FRAMES, NOBODY, dtype=np.int8)
        for start, stop in HALVES:
            labels[start:stop] = debounce_spells(
                raw[start:stop].astype(np.int8), min_spell)
        labels = _outside_fill(labels)
        write_submission(
            labels,
            os.path.join(
                HERE, "results",
                "tcn_clean_debounce{}{}.csv".format(min_spell, suffix)))


def evaluate(paths):
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    for path in paths:
        pred = pm.label_codes(pm.read_label_csv(path))
        result = pm.evaluate(gt, pred)
        print("{:32s} spell={:.4f} macro={:.4f} trans25={:.4f} "
              "poss={:2d}".format(
                  os.path.basename(path),
                  result["spell_f1_mean"],
                  result["macro_f1"],
                  result["transitions"][25]["f1"],
                  result["n_pred_spells"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--soft-weight", type=float, default=0.5)
    parser.add_argument("--tv-weight", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="")
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--labels", default=LABELS)
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

    model_path = _tagged_path(MODEL_PATH, args.tag)
    logits_path = _tagged_path(LOGITS_PATH, args.tag)

    if args.load and os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        config = checkpoint["config"]
        model = PossessionTCN(**config)
        model.load_state_dict(checkpoint["state_dict"])
        with np.load(args.features) as feats:
            matrix, names = build_design_matrix(feats)
        matrix = apply_normalizer(
            matrix, checkpoint["median"], checkpoint["scale"])
        print("Loaded {}".format(model_path))
    else:
        model, matrix, _, _ = train(args)

    logits, _ = predict(model, matrix, logits_path)
    if args.skip_variants:
        return

    suffix = "_" + args.tag if args.tag else ""
    write_variants(logits, args.tag)
    evaluate([
        os.path.join(HERE, "results", "tcn_clean_raw{}.csv".format(suffix)),
        os.path.join(
            HERE, "results", "tcn_clean_debounce25{}.csv".format(suffix)),
        os.path.join(
            HERE, "results", "tcn_clean_debounce50{}.csv".format(suffix)),
    ])


if __name__ == "__main__":
    main()
