"""EDA 01: Ground Truth Analysis and Hierarchical Task Decomposition.
Inspired by Borghesi et al. (2023) - Expert Systems with Applications.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
META_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public_meta.json")
REPORT_PATH = os.path.join(HERE, "reports", "report_01_ground_truth.md")

FPS = 25.0


def run_analysis():
    print("=" * 70)
    print("EDA 01: Ground Truth Analysis & Hierarchical Decomposition")
    print("=" * 70)

    # 1. Load Metadata
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    # 2. Load Ground Truth CSV
    df = pd.read_csv(GT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    total_frames = len(df)
    graded_mask = df["label"].isin(["H", "A", "D"])
    graded_df = df[graded_mask].copy().reset_index(drop=True)
    n_graded = len(graded_df)

    print(f"Total Match Frames:   {total_frames:,}")
    print(f"Graded (Public GT):   {n_graded:,} frames ({n_graded / FPS:.1f} s / {n_graded / (FPS * 60):.2f} min)")

    # 3. Class Distribution
    counts = graded_df["label"].value_counts()
    h_count = counts.get("H", 0)
    a_count = counts.get("A", 0)
    d_count = counts.get("D", 0)
    alive_count = h_count + a_count

    # 4. Hierarchical Decomposition (Borghesi et al., 2023)
    p_alive = alive_count / n_graded
    p_dead = d_count / n_graded
    p_h_given_alive = h_count / alive_count if alive_count > 0 else 0
    p_a_given_alive = a_count / alive_count if alive_count > 0 else 0

    print("\n--- Hierarchical Breakdown (Borghesi et al. framework) ---")
    print(f"1. Dead vs. Alive (DA):")
    print(f"   Alive (H + A): {alive_count:5d} frames ({p_alive*100:5.2f}%)")
    print(f"   Dead  (D):     {d_count:5d} frames ({p_dead*100:5.2f}%)")
    print(f"2. Possession given Alive (POSS | Alive):")
    print(f"   Home (H):      {h_count:5d} frames ({p_h_given_alive*100:5.2f}%)")
    print(f"   Away (A):      {a_count:5d} frames ({p_a_given_alive*100:5.2f}%)")

    # 5. Segment / Spells Extraction
    # Identify contiguous blocks
    labels = graded_df["label"].values
    frames = graded_df["frame"].values

    change_indices = np.where(labels[:-1] != labels[1:])[0] + 1
    split_indices = np.concatenate(([0], change_indices, [len(labels)]))

    segments = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        seg_label = labels[start_idx]
        seg_start_frame = frames[start_idx]
        seg_end_frame = frames[end_idx - 1]
        dur_frames = end_idx - start_idx
        dur_seconds = dur_frames / FPS
        segments.append({
            "segment_id": i + 1,
            "label": seg_label,
            "start_frame": seg_start_frame,
            "end_frame": seg_end_frame,
            "dur_frames": dur_frames,
            "dur_seconds": dur_seconds,
            "is_possession": seg_label in ["H", "A"]
        })

    seg_df = pd.DataFrame(segments)

    poss_df = seg_df[seg_df["is_possession"]].copy().reset_index(drop=True)
    dead_df = seg_df[~seg_df["is_possession"]].copy().reset_index(drop=True)

    print(f"\nTotal Segments: {len(seg_df)}")
    print(f"Possessions:    {len(poss_df)} ({len(poss_df[poss_df['label']=='H'])} H, {len(poss_df[poss_df['label']=='A'])} A)")
    print(f"Dead Stoppages: {len(dead_df)}")

    # 6. Duration Statistics
    poss_dur = poss_df["dur_seconds"]
    h_dur = poss_df[poss_df["label"] == "H"]["dur_seconds"]
    a_dur = poss_df[poss_df["label"] == "A"]["dur_seconds"]
    d_dur = dead_df["dur_seconds"]

    def stats_summary(s):
        return {
            "count": len(s),
            "mean": s.mean() if len(s) else 0,
            "std": s.std() if len(s) > 1 else 0,
            "min": s.min() if len(s) else 0,
            "p25": s.quantile(0.25) if len(s) else 0,
            "median": s.median() if len(s) else 0,
            "p75": s.quantile(0.75) if len(s) else 0,
            "p90": s.quantile(0.90) if len(s) else 0,
            "max": s.max() if len(s) else 0,
        }

    p_stats = stats_summary(poss_dur)
    d_stats = stats_summary(d_dur)

    print("\n--- Possession Duration (seconds) ---")
    print(f"  Mean: {p_stats['mean']:.2f}s | Median: {p_stats['median']:.2f}s | Min: {p_stats['min']:.2f}s | Max: {p_stats['max']:.2f}s | Std: {p_stats['std']:.2f}s")
    print(f"  H Mean: {h_dur.mean():.2f}s (n={len(h_dur)}) | A Mean: {a_dur.mean():.2f}s (n={len(a_dur)})")

    print("\n--- Dead Stoppage Duration (seconds) ---")
    print(f"  Mean: {d_stats['mean']:.2f}s | Median: {d_stats['median']:.2f}s | Min: {d_stats['min']:.2f}s | Max: {d_stats['max']:.2f}s")

    # 7. Transitions & Markov Matrix
    # Frame-by-frame Markov transition matrix P(S_t | S_{t-1})
    states = ["H", "A", "D"]
    state_to_idx = {s: i for i, s in enumerate(states)}
    trans_counts = np.zeros((3, 3), dtype=int)

    for i in range(len(labels) - 1):
        s_prev = labels[i]
        s_next = labels[i + 1]
        trans_counts[state_to_idx[s_prev], state_to_idx[s_next]] += 1

    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    # Spell-level transitions
    spell_transitions = []
    for i in range(len(segments) - 1):
        from_label = segments[i]["label"]
        to_label = segments[i + 1]["label"]
        spell_transitions.append(f"{from_label} -> {to_label}")
    spell_trans_counts = pd.Series(spell_transitions).value_counts()

    print("\n--- Spell-Level Transitions ---")
    for trans, count in spell_trans_counts.items():
        print(f"  {trans:10s} : {count}")

    # 8. Generate Markdown Report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as rep:
        rep.write("# Reporte EDA 01: Análisis del Ground Truth y Descomposición Jerárquica\n\n")
        rep.write("*Basado en el framework de Borghesi et al. (2023), Expert Systems with Applications.*\n\n")
        rep.write("## 1. Resumen General del Conjunto Calificado\n\n")
        rep.write(f"- **Frames Calificados:** {n_graded:,} (equivalente a {n_graded/FPS:.1f} s / 5.0 minutos a 25 fps).\n")
        rep.write(f"- **Rango de Video:** Frames `{graded_df['frame'].min()}` a `{graded_df['frame'].max()}`.\n")
        rep.write(f"- **Total de Segmentos:** {len(seg_df)} ({len(poss_df)} posesiones activas, {len(dead_df)} detenciones de balón muerto).\n\n")

        rep.write("## 2. Descomposición Jerárquica ($DA$ y $POSS$)\n\n")
        rep.write("| Nivel de Decisión | Clase / Estado | Frames | % Nivel | % Total Calificado |\n")
        rep.write("|---|---|---|---|---|\n")
        rep.write(f"| **1. Dead vs Alive ($DA$)** | **Alive** (`H` + `A`) | {alive_count:,} | {p_alive*100:.2f}% | {p_alive*100:.2f}% |\n")
        rep.write(f"| | **Dead** (`D`) | {d_count:,} | {p_dead*100:.2f}% | {p_dead*100:.2f}% |\n")
        rep.write(f"| **2. Posesión ($POSS \\mid \\text{{Alive}}$)** | **Home** (`H` - Barcelona) | {h_count:,} | {p_h_given_alive*100:.2f}% | {h_count/n_graded*100:.2f}% |\n")
        rep.write(f"| | **Away** (`A` - Athletic) | {a_count:,} | {p_a_given_alive*100:.2f}% | {a_count/n_graded*100:.2f}% |\n\n")

        rep.write("> [!IMPORTANT]\n")
        rep.write("> **Implicación directa para el modelado:** La probabilidad a priori de juego activo es del 80.4%, y cuando el juego está vivo el Barcelona domina el 57.3% del tiempo en este tramo. Un clasificador que intente resolver todo en una sola capa confunde los estados de parada con posesión defensiva.\n\n")

        rep.write("## 3. Caracterización de las 17 Posesiones Reales\n\n")
        rep.write("| Métrica | Posesiones Globales | Posesiones Home (`H`) | Posesiones Away (`A`) | Balón Muerto (`D`) |\n")
        rep.write("|---|---|---|---|---|\n")
        rep.write(f"| **Número de Spells** | {len(poss_df)} | {len(h_dur)} | {len(a_dur)} | {len(d_dur)} |\n")
        rep.write(f"| **Duración Media (s)** | **{p_stats['mean']:.2f} s** | {h_dur.mean():.2f} s | {a_dur.mean():.2f} s | {d_stats['mean']:.2f} s |\n")
        rep.write(f"| **Desviación Estándar** | {p_stats['std']:.2f} s | {h_dur.std():.2f} s | {a_dur.std():.2f} s | {d_stats['std']:.2f} s |\n")
        rep.write(f"| **Mínima Duración** | {p_stats['min']:.2f} s ({int(p_stats['min']*FPS)} frames) | {h_dur.min():.2f} s | {a_dur.min():.2f} s | {d_stats['min']:.2f} s |\n")
        rep.write(f"| **Mediana (p50)** | {p_stats['median']:.2f} s | {h_dur.median():.2f} s | {a_dur.median():.2f} s | {d_stats['median']:.2f} s |\n")
        rep.write(f"| **Percentil 90 (p90)** | {p_stats['p90']:.2f} s | {h_dur.quantile(0.9):.2f} s | {a_dur.quantile(0.9):.2f} s | {d_stats['p90']:.2f} s |\n")
        rep.write(f"| **Máxima Duración** | {p_stats['max']:.2f} s | {h_dur.max():.2f} s | {a_dur.max():.2f} s | {d_stats['max']:.2f} s |\n\n")

        rep.write("### Tabla Cronológica de Posesiones y Detenciones en el Ground Truth\n\n")
        rep.write("| # | Tipo | Label | Frame Inicio | Frame Fin | Duración (frames) | Duración (s) |\n")
        rep.write("|---|---|---|---|---|---|---|\n")
        for _, row in seg_df.iterrows():
            tipo = "Posesión" if row["is_possession"] else "Balón Muerto"
            rep.write(f"| {row['segment_id']} | {tipo} | `{row['label']}` | {row['start_frame']} | {row['end_frame']} | {row['dur_frames']} | {row['dur_seconds']:.2f} s |\n")

        rep.write("\n## 4. Dinámica Temporal y Matriz de Transiciones de Estado\n\n")
        rep.write("### A. Matriz de Transición Frame a Frame $P(S_t \\mid S_{t-1})$\n\n")
        rep.write("| Estado Anterior ($S_{t-1}$) | $P(H_t)$ | $P(A_t)$ | $P(D_t)$ | Interpretación |\n")
        rep.write("|---|---|---|---|---|\n")
        for i, s_from in enumerate(states):
            rep.write(f"| **{s_from}** | {trans_prob[i, 0]:.5f} | {trans_prob[i, 1]:.5f} | {trans_prob[i, 2]:.5f} | Inercia de permanencia: {trans_prob[i, i]*100:.2f}% |\n")

        rep.write("\n> [!TIP]\n")
        rep.write("> **Regla fundamental para Viterbi / HMM:** La probabilidad diagonal es superior al **98.8%** en los tres estados. Un cambio directo $H \\to A$ ocurre solo con probabilidad **0.00087** por frame. Esto fundamenta el uso de penalización de transición para maximizar `spellF1`.\n\n")

        rep.write("### B. Tipos de Transiciones entre Bloques (Spell-Level)\n\n")
        rep.write("| Transición | Frecuencia | % Transiciones | Descripción |\n")
        rep.write("|---|---|---|---|\n")
        for trans, count in spell_trans_counts.items():
            rep.write(f"| `{trans}` | {count} | {count / len(spell_transitions)*100:.1f}% | Transición observada en el partido |\n")

        rep.write("\n---\n*Generado automáticamente por `eda_01_ground_truth.py`.*")

    print(f"\n[OK] Report written successfully to: {REPORT_PATH}")


if __name__ == "__main__":
    run_analysis()
