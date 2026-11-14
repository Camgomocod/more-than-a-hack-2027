"""Diagnostic validation for the public scoreboard candidates.

This script does not tune anything. It scores frozen submissions, adds a paired
bootstrap over ground-truth possessions, and reports leave-one-possession-out
sensitivity. The public five-minute window is treated as an in-sample diagnostic,
not as a model-selection set.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "models"))

import possession_metrics as pm  # noqa: E402

DEFAULT_GT = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
CANDIDATES = {
    "model_tcn_clean": ROOT / "evaluation" / "results" / "model_tcn_clean.csv",
    "model_ensemble": ROOT / "evaluation" / "results" / "model_ensemble.csv",
    "model_consensus": ROOT / "evaluation" / "results" / "model_consensus.csv",
}
PROVENANCE = [
    ("5485e0d", "improve spellF1 to 0.5730 and macroF1 to 0.8309 via dead-ball kinetic rest filter"),
    ("086d429", "achieve rank 1 spellF1 0.6400 and macroF1 0.8508 via consensus ensemble"),
    ("64d16b3", "reach all-time macroF1 0.8667 and accuracy 87.15% (spellF1 0.6400) via expanded dead-ball kinetic persistence"),
]


def _float(value):
    return None if value is None else float(value)


def _codes(path, expected_length):
    table = pm.read_label_csv(str(path))
    reason = pm.validate_submission(table, expected_length)
    if reason:
        raise ValueError("{}: {}".format(path, reason))
    return pm.label_codes(table)


def _flags(gt_spells, filled, runs):
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    return {
        threshold: pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
        for threshold in pm.SPELL_IOUS
    }, len(pred_spells)


def _score_from_flags(flags, n_gt, n_pred, picks=None):
    f1s = []
    for threshold in pm.SPELL_IOUS:
        row = flags[threshold]
        if picks is not None:
            true_positive = int(row[picks].sum())
        else:
            true_positive = int(row.sum())
        false_negative = n_gt - true_positive
        false_positive = max(0, n_pred - true_positive)
        f1s.append(
            2.0 * true_positive
            / (2 * true_positive + false_positive + false_negative)
            if true_positive
            else 0.0
        )
    return float(np.mean(f1s))


def _paired_bootstrap(records, n_gt, draws, seed):
    rng = np.random.RandomState(seed)
    names = list(records)
    scores = {name: np.empty(draws, dtype=np.float64) for name in names}
    for draw in range(draws):
        picks = rng.randint(0, n_gt, n_gt)
        for name in names:
            record = records[name]
            scores[name][draw] = _score_from_flags(
                record["flags"], n_gt, record["n_pred"], picks
            )

    pairs = {}
    for left in names:
        for right in names:
            if left >= right:
                continue
            difference = scores[left] - scores[right]
            pairs["{} vs {}".format(left, right)] = {
                "mean_difference": float(difference.mean()),
                "p05": float(np.percentile(difference, 5)),
                "p95": float(np.percentile(difference, 95)),
                "fraction_left_better": float((difference > 0).mean()),
            }
    return pairs


def _leave_one_out(record, gt_spells):
    n_gt = record["n_gt"]
    scores = []
    for index in range(n_gt):
        keep = np.ones(n_gt, dtype=bool)
        keep[index] = False
        scores.append(
            _score_from_flags(record["flags"], n_gt - 1, record["n_pred"], keep)
        )
    scores = np.asarray(scores)
    worst = int(np.argmin(scores))
    best = int(np.argmax(scores))
    start, stop, code = gt_spells[worst]
    return {
        "full": record["spell_f1_mean"],
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "worst_index": worst,
        "worst_frame": int(start),
        "worst_stop": int(stop),
        "worst_minute": float(start) / pm.FPS / 60.0,
        "worst_label": pm.LABELS[code],
        "best_index": best,
        "best_frame": int(gt_spells[best][0]),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _markdown(report):
    lines = []
    lines.append("# Validación diagnóstica de `model_consensus`")
    lines.append("")
    lines.append("**Advertencia:** esta validación es *in-sample* sobre la ventana pública de "
                 "5 minutos / 17 posesiones. No es una estimación del partido completo y no "
                 "debe usarse para elegir hiperparámetros.")
    lines.append("")
    lines.append("## Auditoría de procedencia")
    lines.append("")
    lines.append("El historial de git muestra que el score público fue usado como objetivo "
                 "de selección:")
    lines.append("")
    lines.append("| commit | mensaje |")
    lines.append("|---|---|")
    for commit, message in PROVENANCE:
        lines.append("| `{}` | {} |".format(commit, message))
    lines.append("")
    lines.append("Por tanto, cualquier intervalo calculado aquí sigue siendo optimista: "
                 "el modelo fue elegido mirando estas mismas 17 posesiones.")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append("| submission | spellF1 | macroF1 | transF1@12 | transF1@25 | "
                 "poss | dur | frag | bootstrap spellF1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, record in report["candidates"].items():
        lines.append(
            "| `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {:.2f} s | {:.3f} | "
            "[{:.4f}, {:.4f}] |".format(
                name,
                record["spell_f1_mean"],
                record["macro_f1"],
                record["transitions"]["12"]["f1"],
                record["transitions"]["25"]["f1"],
                record["n_pred_spells"],
                record["mean_spell_duration_s"],
                record["fragmentation"],
                record["spell_bootstrap_low"],
                record["spell_bootstrap_high"],
            )
        )
    lines.append("")
    lines.append("## Bootstrap pareado")
    lines.append("")
    lines.append("Mismos resamples de posesiones del GT para todos los modelos. "
                 "Intervalo percentil 5–95 y fracción de resamples en los que el primero gana.")
    lines.append("")
    lines.append("| comparación | diferencia media | p05 | p95 | fracción A > B |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, pair in report["paired_bootstrap"].items():
        lines.append(
            "| {} | {:+.4f} | {:+.4f} | {:+.4f} | {:.3f} |".format(
                name,
                pair["mean_difference"],
                pair["p05"],
                pair["p95"],
                pair["fraction_left_better"],
            )
        )
    lines.append("")
    lines.append("## `spellF1` por IoU")
    lines.append("")
    lines.append("| submission | " + " | ".join("IoU {:.1f}".format(t) for t in pm.SPELL_IOUS) + " |")
    lines.append("|---|" + "---:|" * len(pm.SPELL_IOUS))
    for name, record in report["candidates"].items():
        values = [record["spell_f1"][str(threshold)]["f1"] for threshold in pm.SPELL_IOUS]
        lines.append("| `{}` | {} |".format(
            name, " | ".join("{:.4f}".format(value) for value in values)))
    lines.append("")
    lines.append("## Transiciones")
    lines.append("")
    lines.append("| submission | tol | F1 | precisión | recall | latencia media | "
                 "matched | missed | spurious |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, record in report["candidates"].items():
        for tol in ("12", "25"):
            transition = record["transitions"][tol]
            lines.append(
                "| `{}` | {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {} | {} | {} |".format(
                    name,
                    tol,
                    transition["f1"],
                    transition["precision"],
                    transition["recall"],
                    transition["mean_abs_latency"],
                    transition["matched"],
                    transition["missed"],
                    transition["spurious"],
                )
            )
    lines.append("")
    lines.append("## Sensibilidad a una posesión")
    lines.append("")
    lines.append("Jackknife: se elimina una posesión del GT y se recalcula `spellF1`. "
                 "Sirve para ver cuánto depende el resultado de una sola posesión.")
    lines.append("")
    lines.append("| submission | completo | mínimo | máximo | peor posesión |")
    lines.append("|---|---:|---:|---:|---|")
    for name, record in report["candidates"].items():
        loo = record["leave_one_out"]
        lines.append(
            "| `{}` | {:.4f} | {:.4f} | {:.4f} | #{} ({}, {:.2f} min) |".format(
                name,
                loo["full"],
                loo["min"],
                loo["max"],
                loo["worst_index"],
                loo["worst_label"],
                loo["worst_minute"],
            )
        )
    lines.append("")
    lines.append("## Lectura")
    lines.append("")
    lines.append("- El bootstrap pareado mide si la diferencia se mantiene al remuestrear "
                 "posesiones, pero **no corrige** el sesgo de haber elegido el modelo en esta misma ventana.")
    lines.append("- El jackknife muestra cuánto cambia el resultado al quitar una posesión; "
                 "si el rango es grande, el 0.64 no es estable.")
    lines.append("- La decisión correcta para la submission final no debería basarse solo en esta tabla.")
    lines.append("")

    consensus = report["candidates"]["model_consensus"]
    tcn = report["candidates"]["model_tcn_clean"]
    pair = report["paired_bootstrap"]["model_consensus vs model_tcn_clean"]
    lines.append("## Conclusión operativa")
    lines.append("")
    lines.append(
        "`model_consensus` es el mejor candidato en la ventana pública: "
        "`spellF1` {:.4f} frente a {:.4f} de `model_tcn_clean`, con una diferencia "
        "pareada media de {:+.4f} (p05 {:+.4f}, p95 {:+.4f}) y {:.1f}% de resamples "
        "a favor.".format(
            consensus["spell_f1_mean"],
            tcn["spell_f1_mean"],
            pair["mean_difference"],
            pair["p05"],
            pair["p95"],
            100.0 * pair["fraction_left_better"],
        )
    )
    lines.append("")
    lines.append(
        "Pero la ventana pública ya se usó para seleccionar y ajustar el modelo "
        "(ver auditoría de commits). El propio scoreboard avisa de que las diferencias "
        "pequeñas quedan dentro del ruido con 17 posesiones. El jackknife lo confirma: "
        "`model_consensus` baja a {:.4f} al quitar una sola posesión, mientras que "
        "`model_tcn_clean` llega a {:.4f} en su mejor caso.".format(
            consensus["leave_one_out"]["min"], tcn["leave_one_out"]["max"]
        )
    )
    lines.append("")
    lines.append(
        "**Recomendación:** no cambiar `los_chuchos.csv` a `model_consensus.csv` solo "
        "con esta evidencia. Mantener `model_tcn_clean.csv` como submission final y "
        "usar `model_consensus` como candidato experimental hasta tener una validación "
        "held-out o de partido completo. Si el equipo decide asumir el riesgo de "
        "sobreajuste para el scoreboard público, `model_consensus` es la mejor opción "
        "en esta ventana, pero debe etiquetarse como tal."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(HERE / "evaluation"))
    args = parser.parse_args()

    gt_codes = pm.label_codes(pm.read_label_csv(args.gt))
    runs = pm.valid_runs(gt_codes)
    gt_spells = pm.possession_spells(gt_codes, runs)
    n_gt = len(gt_spells)

    records = {}
    for name, path in CANDIDATES.items():
        pred_codes = _codes(path, len(gt_codes))
        filled = pm.fill_abstentions(pred_codes, runs)
        flags, n_pred = _flags(gt_spells, filled, runs)
        result = pm.evaluate(gt_codes, pred_codes, bootstrap_n=args.bootstrap)
        record = {
            "path": str(path.relative_to(ROOT)),
            "n_gt": n_gt,
            "n_pred": n_pred,
            "flags": flags,
            "spell_f1_mean": _float(result["spell_f1_mean"]),
            "macro_f1": _float(result["macro_f1"]),
            "accuracy": _float(result["accuracy"]),
            "spell_bootstrap_low": _float(result["spell_bootstrap_low"]),
            "spell_bootstrap_high": _float(result["spell_bootstrap_high"]),
            "spell_f1": result["spell_f1"],
            "n_pred_spells": int(result["n_pred_spells"]),
            "n_gt_spells": int(result["n_gt_spells"]),
            "mean_spell_duration_s": _float(result["mean_spell_duration_s"]),
            "gt_mean_spell_duration_s": _float(result["gt_mean_spell_duration_s"]),
            "fragmentation": _float(result["fragmentation"]),
            "transitions": {},
        }
        for tolerance in (12, 25):
            transition = result["transitions"][tolerance]
            record["transitions"][str(tolerance)] = {
                "f1": _float(transition["f1"]),
                "precision": _float(transition["precision"]),
                "recall": _float(transition["recall"]),
                "mean_abs_latency": _float(transition["mean_abs_latency"]),
                "matched": len(transition["pairs"]),
                "missed": len(transition["missed"]),
                "spurious": len(transition["spurious"]),
            }
        record["leave_one_out"] = _leave_one_out(record, gt_spells)
        records[name] = record

    paired = _paired_bootstrap(records, n_gt, args.bootstrap, args.seed)
    report = {
        "ground_truth": str(Path(args.gt).relative_to(ROOT)),
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
        "candidates": {
            name: {key: value for key, value in record.items() if key != "flags"}
            for name, record in records.items()
        },
        "paired_bootstrap": paired,
        "provenance": PROVENANCE,
    }
    report = _json_safe(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "consensus_validation.json"
    md_path = out_dir / "consensus_validation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")

    print("Wrote {}".format(json_path))
    print("Wrote {}".format(md_path))
    print(_markdown(report))


if __name__ == "__main__":
    main()
