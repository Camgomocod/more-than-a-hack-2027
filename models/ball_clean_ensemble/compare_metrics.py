"""Print a compact comparison of the ball-clean ensemble runs."""

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCOREBOARD = os.path.join(HERE, "evaluation", "scoreboard.csv")

COLUMNS = (
    "submission",
    "spell_f1_mean",
    "macro_f1",
    "spell_f1_at_headline",
    "n_pred_segments",
    "fragmentation",
    "mean_abs_latency",
    "n_abstain_on_graded",
    "spell_bootstrap_low",
    "spell_bootstrap_high",
)


def main():
    parser = argparse.ArgumentParser(
        description="Compara el scoreboard de las variantes limpias.")
    parser.add_argument("--scoreboard", default=DEFAULT_SCOREBOARD)
    args = parser.parse_args()

    if not os.path.exists(args.scoreboard):
        raise FileNotFoundError(
            "No se encontró {}. Ejecuta run_evaluation.py primero.".format(
                args.scoreboard))

    frame = pd.read_csv(args.scoreboard)
    columns = [column for column in COLUMNS if column in frame.columns]
    if "spell_f1_mean" in frame.columns:
        frame = frame.sort_values(
            "spell_f1_mean", ascending=False, na_position="last")

    print("\nComparación de variantes")
    print("=" * 100)
    print(frame[columns].to_string(index=False))

    baseline = frame[
        frame["submission"].astype(str).str.contains("baseline_ensemble")
    ]
    if baseline.empty or "spell_f1_mean" not in frame.columns:
        return

    base = baseline.iloc[0]
    metrics = [
        "spell_f1_mean",
        "macro_f1",
        "spell_f1_at_headline",
        "n_pred_segments",
        "fragmentation",
        "mean_abs_latency",
        "n_abstain_on_graded",
    ]
    metrics = [metric for metric in metrics if metric in frame.columns]

    deltas = frame[["submission"] + metrics].copy()
    for metric in metrics:
        deltas["delta_" + metric] = deltas[metric] - base[metric]

    print("\nDeltas respecto a baseline_ensemble")
    print("=" * 100)
    delta_columns = ["submission"] + [
        "delta_" + metric for metric in metrics]
    print(deltas[delta_columns].to_string(index=False))


if __name__ == "__main__":
    main()
