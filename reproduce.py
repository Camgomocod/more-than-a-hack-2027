"""One entry point for the shipped submission.

    python reproduce.py            score los_chuchos.csv against the public GT
    python reproduce.py --verify   rebuild los_chuchos.csv from the committed
                                   artefacts and check its SHA-256

The default path needs numpy and pandas only: no data bundle, no torch, no GPU.
It runs the official scorer, compares the result with the recorded public
score, and prints the held-out IDSSE benchmark.

``--verify`` re-runs the last stage of the pipeline (the consensus ensemble)
from the artefacts committed in this repository and checks that it reproduces
``los_chuchos.csv`` byte for byte.

The full rebuild from the gated data is documented in TECHNICAL_DOCUMENT.md.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402

SUBMISSION = os.path.join(ROOT, "los_chuchos.csv")
EXPECTED_SHA256 = "852AA5FBD8AAFB0881FE0EBA41655E301873453EF39F0A08A96F44655306DA53"
EXPECTED_PUBLIC = {
    "spell_f1_mean": 0.6400,
    "macro_f1": 0.8667,
}
TOLERANCE = 1e-4
HELDOUT_POOLED = os.path.join(ROOT, "evaluation", "heldout_idesse",
                              "heldout_pooled.csv")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _rule(title):
    sys.stdout.flush()
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    sys.stdout.flush()


def _check_format():
    table = pm.read_label_csv(SUBMISSION)
    expected_rows = GAMES[GAME]["n_video_frames"]
    reason = pm.validate_submission(table, expected_rows)
    if reason:
        raise SystemExit("FAIL  {} is not a valid submission: {}".format(
            os.path.basename(SUBMISSION), reason))
    codes = pm.label_codes(table)
    counts = {name: int(np.sum(codes == index))
              for index, name in enumerate(pm.LABELS)}
    print("Format OK: {} rows, frame 0..{}".format(len(codes), len(codes) - 1))
    print("Labels: " + ", ".join("{} {:.1f}%".format(
        name, 100.0 * count / len(codes)) for name, count in counts.items()))
    sys.stdout.flush()


def _score():
    report_path = os.path.join(ROOT, "evaluation", "reports",
                               "los_chuchos.json")
    command = [
        sys.executable,
        os.path.join("evaluation", "run_evaluation.py"),
        "--results", ".",
        "--gt", "public",
        "--bootstrap", "1000",
    ]
    sys.stdout.flush()
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit("FAIL  the official scorer returned {}".format(
            result.returncode))
    if not os.path.exists(report_path):
        raise SystemExit("FAIL  no report at {}".format(report_path))

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    print()
    print("Recorded public score:")
    worst = 0.0
    for key, expected in EXPECTED_PUBLIC.items():
        found = float(report[key])
        delta = abs(found - expected)
        worst = max(worst, delta)
        print("  {:16s} {:.4f}  (expected {:.4f}, delta {:.5f})".format(
            key, found, expected, delta))
    print("  {:16s} {:.4f}".format("accuracy", float(report["accuracy"])))
    print("  {:16s} {:.4f}".format(
        "trans_f1@12", float(report["transitions"]["12"]["f1"])))
    print("  {:16s} {:.4f}".format(
        "trans_f1@25", float(report["transitions"]["25"]["f1"])))

    if worst <= TOLERANCE:
        print("\nMatches the recorded public score: OK")
    else:
        print("\nDIFFERS from the recorded public score by up to {:.5f}. "
              "Check git status and the CSV hash.".format(worst))


def _heldout():
    if not os.path.exists(HELDOUT_POOLED):
        return
    table = pd.read_csv(HELDOUT_POOLED)
    _rule("Held-out IDSSE benchmark (two unseen matches, pooled)")
    columns = ["panel", "model", "spell_f1", "spell_lo", "spell_hi",
               "macro_f1", "n_gt_spells", "n_pred_spells"]
    print(table[columns].to_string(index=False,
                                   float_format=lambda value: "{:.4f}".format(value)))
    print("\nPanel A is the fair IDSSE-only comparison. Panel B transfers the "
          "BAR-ATH\nmodels with Smart Tagging neutralised; treat it as "
          "indicative.")
    print("The consensus keeps the public lead (0.6400) but loses to the TCN "
          "on these\nunseen matches. That in-sample risk is stated in "
          "TECHNICAL_DOCUMENT.md.")


def _next_steps():
    _rule("What you can run next")
    print("  python reproduce.py --verify   rebuild the submission from the "
          "committed artefacts (no data needed)")
    print("  python evaluation/run_evaluation.py --results . --gt public --bootstrap 1000")
    print("                                 the official scorer on its own")
    print("  jupyter notebook visual_validation/barca_atd_animation.ipynb")
    print("                                 frame-by-frame TCN vs consensus")
    print()
    print("The full rebuild from the gated data is in TECHNICAL_DOCUMENT.md "
          "(section 6).")


def _verify():
    _rule("Verifying that los_chuchos.csv is reproducible")
    inputs = [
        os.path.join(ROOT, "features", "features.npz"),
        os.path.join(ROOT, "evaluation", "results", "model_tcn_clean.csv"),
        os.path.join(ROOT, "evaluation", "results", "model_ensemble.csv"),
    ]
    missing = [path for path in inputs if not os.path.exists(path)]
    if missing:
        raise SystemExit("FAIL  missing committed input(s):\n  " +
                         "\n  ".join(missing))

    if not os.path.exists(SUBMISSION):
        raise SystemExit("FAIL  {} is missing".format(SUBMISSION))
    before = _sha256(SUBMISSION)
    backup_handle, backup_path = tempfile.mkstemp(suffix=".csv")
    os.close(backup_handle)
    shutil.copy2(SUBMISSION, backup_path)

    try:
        result = subprocess.run(
            [sys.executable, os.path.join("models", "consensus_ensemble.py")],
            cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit("FAIL  models/consensus_ensemble.py returned {}"
                             .format(result.returncode))

        after = _sha256(SUBMISSION)
        print()
        print("  rebuild : {}".format(after))
        print("  expected: {}".format(EXPECTED_SHA256))
        print("  shipped : {}".format(before))

        if after != EXPECTED_SHA256:
            shutil.copy2(backup_path, SUBMISSION)
            raise SystemExit(
                "FAIL  the rebuild does not match the expected SHA-256. "
                "The shipped file has been restored.")
        if after != before:
            shutil.copy2(backup_path, SUBMISSION)
            raise SystemExit(
                "FAIL  the rebuild matches the expected SHA-256 but not the "
                "shipped file. The shipped file has been restored.")
        print("\nSHA-256 verified: the committed artefacts regenerate "
              "los_chuchos.csv exactly.")
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="rebuild los_chuchos.csv and check its SHA-256")
    args = parser.parse_args()

    print("Python {} | numpy {} | pandas {}".format(
        sys.version.split()[0], np.__version__, pd.__version__))

    if args.verify:
        _verify()
        return

    _rule("Scoring the shipped submission")
    print("File: {}".format(os.path.relpath(SUBMISSION, ROOT)))
    print("SHA-256: {}".format(_sha256(SUBMISSION)))
    _check_format()
    _score()
    _heldout()
    _next_steps()


if __name__ == "__main__":
    main()
