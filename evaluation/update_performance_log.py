"""Generate and update structured performance logs (JSON & Markdown).
Tracks model evolution, scoreboard metrics, and external benchmarks.
"""

import json
import os
import sys
import pandas as pd
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCOREBOARD_CSV = os.path.join(HERE, "scoreboard.csv")
PERF_JSON = os.path.join(HERE, "performance_log.json")
PERF_MD = os.path.join(ROOT, "docs", "performance_log.md")

DESCRIPTIONS = {
    "model_viterbi_refined": {
        "phase": "Fase 5.1 (Modelo Refinado)",
        "summary": "HMM Jerárquico + Viterbi Global + Módulo de Refinamiento Cinemático de Fronteras (latencia cero, física pura).",
        "rank_official": 1,
    },
    "model_viterbi": {
        "phase": "Fase 4 (HMM Viterbi Base)",
        "summary": "HMM Jerárquico con Decodificación Global de Viterbi, matriz A con inercia Markoviana (0.995) y temperatura de emisión (0.80).",
        "rank_official": 2,
    },
    "baseline_smooth125": {
        "phase": "Baseline Oficial",
        "summary": "Suavizado temporal por ventana móvil de 125 frames (5.0 s). Alta inercia, bajo recall en D (7.9%).",
        "rank_official": 2,
    },
    "baseline_v2": {
        "phase": "Fase 2 (Heurístico Contextual)",
        "summary": "Histéresis espacial (radio 2.5m, 8 frames), burbuja 5-NN (+4.3m) y fusión con Smart Tagging (Set Pieces).",
        "rank_official": 3,
    },
    "baseline_smooth50": {
        "phase": "Baseline Oficial",
        "summary": "Suavizado temporal por ventana móvil de 50 frames (2.0 s).",
        "rank_official": 4,
    },
    "baseline_smooth25": {
        "phase": "Baseline Oficial",
        "summary": "Suavizado temporal por ventana móvil de 25 frames (1.0 s).",
        "rank_official": 5,
    },
    "baseline_raw": {
        "phase": "Baseline Oficial",
        "summary": "Asignación frame a frame al jugador más cercano sin memoria temporal (131 micro-posesiones de 2.0s).",
        "rank_official": 6,
    },
}

EXTERNAL_BENCHMARKS = [
    {
        "model": "model_viterbi_refined",
        "dataset": "IDSSE (Bundesliga)",
        "match": "1. FC Köln vs. FC Bayern München (DFL-MAT-J03WMX)",
        "provider": "Sportec Solutions TRACAB (25 fps)",
        "date": "2026-11-08",
        "total_frames": 145967,
        "duration_minutes": 97.3,
        "macro_f1": 0.8317,
        "spell_f1_mean": 0.4055,
        "accuracy": 0.8419,
        "kappa": 0.7554,
        "alive_dead_f1": 0.9304,
        "f1_H": 0.7890,
        "f1_A": 0.8062,
        "f1_D": 0.8899,
        "share_err": 0.0110,
        "gt_share_H": 0.4637,
        "pred_share_H": 0.4527,
        "notes": "Paso 5.1 con refinamiento cinemático. Spell F1 sube de 0.3980 a 0.4055 sobre 145.967 frames."
    },
    {
        "model": "model_viterbi",
        "dataset": "IDSSE (Bundesliga)",
        "match": "1. FC Köln vs. FC Bayern München (DFL-MAT-J03WMX)",
        "provider": "Sportec Solutions TRACAB (25 fps)",
        "date": "2026-11-08",
        "total_frames": 145967,
        "duration_minutes": 97.3,
        "macro_f1": 0.8319,
        "spell_f1_mean": 0.3981,
        "accuracy": 0.8426,
        "kappa": 0.7570,
        "alive_dead_f1": 0.9319,
        "f1_H": 0.7894,
        "f1_A": 0.8091,
        "f1_D": 0.8970,
        "share_err": 0.0110,
        "gt_share_H": 0.4637,
        "pred_share_H": 0.4527,
        "notes": "HMM Viterbi base de Fase 4 antes de refinamiento."
    }
]


def update_logs():
    print(f"Reading scoreboard from {SCOREBOARD_CSV}...")
    df_sb = pd.read_csv(SCOREBOARD_CSV)

    log_entries = []
    for _, row in df_sb.iterrows():
        name = row["submission"]
        desc = DESCRIPTIONS.get(name, {"phase": "Custom", "summary": "", "rank_official": int(row["rank"])})
        entry = {
            "rank": int(row["rank"]),
            "submission": name,
            "phase": desc["phase"],
            "summary": desc["summary"],
            "macro_f1": float(row["macro_f1"]),
            "spell_f1_mean": float(row["spell_f1_mean"]),
            "n_pred_spells": int(row["poss"]),
            "n_gt_spells": int(row["n_gt_spells"]),
            "mean_spell_duration_s": float(row["dur"]),
            "gt_mean_spell_duration_s": float(row["gt_mean_spell_duration_s"]),
            "accuracy": float(row["accuracy"]),
            "kappa": float(row["kappa"]),
            "alive_dead_f1": float(row["alive_dead_f1"]),
            "f1_H": float(row["f1_H"]),
            "f1_A": float(row["f1_A"]),
            "f1_D": float(row["f1_D"]),
            "precision_H": float(row["precision_H"]),
            "recall_H": float(row["recall_H"]),
            "precision_A": float(row["precision_A"]),
            "recall_A": float(row["recall_A"]),
            "precision_D": float(row["precision_D"]),
            "recall_D": float(row["recall_D"]),
            "share_err": float(row["share_err"]),
            "gt_home_share": float(row["gt_home_share"]),
            "pred_home_share": float(row["pred_home_share"]),
            "mean_latency_frames": float(row["mean_latency"]),
            "trans_f1_25": float(row["trans_f1_25"]),
            "trans_precision_25": float(row["trans_precision_25"]),
            "trans_recall_25": float(row["trans_recall_25"]),
            "fragmentation": float(row["fragmentation"]),
        }
        log_entries.append(entry)

    full_payload = {
        "updated_at": datetime.now().isoformat(),
        "competition": "More than a Hack 2027 — Phase 1: Ball Possession",
        "ground_truth_public_frames": 7500,
        "official_leaderboard": log_entries,
        "external_benchmarks": EXTERNAL_BENCHMARKS,
    }

    # 1. Write JSON
    with open(PERF_JSON, "w") as f:
        json.dump(full_payload, f, indent=2)
    print(f"[OK] Saved JSON performance log to: {PERF_JSON}")

    # 2. Write Markdown Dashboard
    md = []
    md.append("# Registro Histórico de Desempeño y Benchmark de Modelos")
    md.append("")
    md.append(f"**Última actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append("**Reto:** More than a Hack 2027 — Phase 1: Ball Possession (Metrica Sports / FC Barcelona / MWCapital)  ")
    md.append(f"**Archivo de datos estructurados:** [`evaluation/performance_log.json`](file://{PERF_JSON})  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Tabla Oficial de Clasificación (Ground Truth Público: FC Barcelona vs. Athletic Club)")
    md.append("")
    md.append("| Rank | Submission | Fase / Enfoque | `spellF1` (Rank) | `macroF1` | Exactitud | $\\kappa$ | Posesiones (GT=17) | Duración (GT=14.2s) | F1 D (Paradas) | `shareE` | `transF1` ($\\tau=25$) |")
    md.append("|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for e in log_entries:
        highlight = "**" if e["rank"] == 1 else ""
        md.append(f"| {highlight}{e['rank']}{highlight} | {highlight}`{e['submission']}`{highlight} | {e['phase']} | {highlight}{e['spell_f1_mean']:.4f}{highlight} | {highlight}{e['macro_f1']:.4f}{highlight} | {e['accuracy']*100:.1f}% | {e['kappa']:.4f} | {e['n_pred_spells']} | {e['mean_spell_duration_s']:.1f}s | {e['f1_D']:.4f} | {e['share_err']:.3f} | {e['trans_f1_25']:.4f} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Progresión Métrica a lo largo de las Fases del Proyecto")
    md.append("")
    md.append("| Hito / Versión | Descripción Clave | `spellF1` | `macroF1` | Recall D | Posesiones | Estado |")
    md.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")
    md.append("| **Baseline Inicial (`baseline_raw`)** | Asignación pura al más cercano frame a frame | 0.0378 | 0.5972 | 33.6% | 131 (2.0s) | Superado |")
    md.append("| **Filtro Suavizado (`smooth125`)** | Ventana móvil ciega de 5 segundos | 0.4242 | 0.5812 | 7.9% | 16 (18.5s) | Superado |")
    md.append("| **Fase 2 (`baseline_v2`)** | Histeresis 2.5m + colchón 5-NN + Set Pieces | 0.4235 | 0.7169 | 41.9% | 17 (15.8s) | Superado |")
    md.append("| **Fase 4 (`model_viterbi`)** | HMM Jerárquico + Viterbi Global + 10 features Fase 3 | **0.5371** | **0.7938** | **63.4%** | **18 (14.2s)** | **#1 Scoreboard** |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Pruebas de Generalización Fuera de Dominio (Benchmark Externo)")
    md.append("")
    md.append("Evaluación en el dataset público **IDSSE** (*Nature Scientific Data*, 2025) con tracking óptico Sportec Solutions TRACAB (25 fps) de la Bundesliga alemana:")
    md.append("")
    md.append("| Partido Evaluado | Frames | Duración | `macroF1` | Exactitud | $\\kappa$ | F1 Balón Parado ($D$) | Error Reparto Posesión |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for b in EXTERNAL_BENCHMARKS:
        md.append(f"| **{b['match']}** | {b['total_frames']:,} | {b['duration_minutes']:.1f} min | **{b['macro_f1']:.4f}** | **{b['accuracy']*100:.1f}%** | **{b['kappa']:.4f}** | **{b['f1_D']:.4f}** | **{b['share_err']*100:.2f}%** |")

    md.append("")
    md.append("### Conclusiones del Benchmark Externo")
    md.append("* El modelo funciona directamente sobre otro sistema de tracking y otra liga sin reentrenamiento.")
    md.append("* La detección de balón parado ($D$) alcanza un F1 de **0.8970**, confirmando la fiabilidad de la arquitectura jerárquica $DA$.")
    md.append("* La cuota de posesión acumulada tiene un error de apenas **1.1%** a lo largo de un partido completo de 97.3 minutos.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Instrucciones de Reproducción")
    md.append("")
    md.append("```bash")
    md.append("# 1. Generar la submission oficial #1")
    md.append(".venv/bin/python models/hmm_viterbi.py")
    md.append("")
    md.append("# 2. Evaluar en el scoreboard oficial")
    md.append("run-eval")
    md.append("")
    md.append("# 3. Ejecutar el benchmark sobre IDSSE Bundesliga")
    md.append(".venv/bin/python evaluation/benchmark_idsse.py")
    md.append("")
    md.append("# 4. Actualizar este registro de desempeño")
    md.append(".venv/bin/python evaluation/update_performance_log.py")
    md.append("```")

    with open(PERF_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[OK] Saved Markdown performance log to: {PERF_MD}")


if __name__ == "__main__":
    update_logs()
